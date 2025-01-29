import torch

from utils import damped_pair_vg, init_Q_exprs, precond_grad_kron_math, update_precond_kron_math_


class KronPSGD(torch.optim.Optimizer):
    """
    Kronecker-factored Preconditioned Stochastic Gradient Descent optimizer.
    
    Args:
        params: Iterable of parameters to optimize
        max_size: Maximum size for triangular matrices (default: inf)
        max_skew: Maximum dimension ratio before switching to diagonal (default: 1.0)
        init_scale: Initial preconditioner scale (default: None, computed automatically)
        lr: Learning rate for parameters (default: 0.01)
        lr_precond: Learning rate for preconditioner (default: 0.1)
        momentum: Momentum factor (default: 0.0)
        grad_clip: Maximum gradient norm (default: None)
        precond_update_prob: Probability of updating preconditioner (default: 1.0)
        step_normalizer: Method for normalizing step size ('1st' or '2nd')
        exact_hvp: Whether to use exact Hessian-vector products
        precond_type: Type of preconditioner ("Newton" or "whitening")
    """
    def __init__(self, params,
                 max_size=float("inf"),
                 max_skew=1.0,
                 init_scale=None,
                 lr=None,
                 lr_precond=None,
                 momentum=0.0,
                 grad_clip=None,
                 precond_update_prob=1.0,
                 step_normalizer='2nd',
                 exact_hvp=True,
                 precond_type="Newton"):
        
        if lr_precond is None:
            lr_precond = 0.1 if step_normalizer == '2nd' else 0.01
            
        if lr is None:
            lr = 0.01 if precond_type == "Newton" else 0.001
            
        defaults = dict(max_size=max_size,
                       max_skew=max_skew,
                       init_scale=init_scale,
                       lr=lr,
                       lr_precond=lr_precond,
                       momentum=momentum if 0 < momentum < 1 else 0.0,
                       grad_clip=grad_clip,
                       precond_update_prob=precond_update_prob,
                       step_normalizer=step_normalizer,
                       exact_hvp=exact_hvp,
                       precond_type=precond_type)
        
        super().__init__(params, defaults)
        
        # Initialize state
        self.state['Q_exprs'] = None
        self.state['momentum_buffer'] = None
        
        # Get parameters
        self._params = [p for group in self.param_groups 
                       for p in group['params'] if p.requires_grad]
        
        # Compute numerical constants
        self._tiny = max([torch.finfo(p.dtype).tiny for p in self._params])
        self._delta = max([torch.finfo(p.dtype).eps for p in self._params]) ** 0.5
        
        if init_scale is None:
            print("Warning: Auto-scaling preconditioner. Manual scaling recommended!")

    @torch.no_grad()
    def step(self, closure):
        """Perform one optimization step."""
        group = self.param_groups[0]
        
        # Get parameters from group
        max_size = group['max_size']
        max_skew = group['max_skew']
        lr = group['lr']
        lr_precond = group['lr_precond']
        momentum = group['momentum']
        grad_clip = group['grad_clip']
        precond_prob = group['precond_update_prob']
        step_normalizer = group['step_normalizer']
        exact_hvp = group['exact_hvp']
        precond_type = group['precond_type']
        
        # Newton-type preconditioner update
        if ((precond_type == "Newton") and 
            (torch.rand([]) < precond_prob or self.state['Q_exprs'] is None)):
            
            if exact_hvp:
                # Exact Hessian-vector products
                with torch.enable_grad():
                    # First evaluation to get gradients
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params, create_graph=True)
                    
                    # Generate random vectors and compute Hessian-vector products
                    vs = [torch.randn_like(p) for p in self._params]
                    Hvs = torch.autograd.grad(grads, self._params, vs)
            else:
                # Approximate Hessian-vector products using finite differences
                with torch.enable_grad():
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params)
                
                # Compute perturbation vectors
                vs = [self._delta * torch.randn_like(p) for p in self._params]
                
                # Apply perturbations
                for param, v in zip(self._params, vs):
                    param.add_(v)
                    
                # Evaluate perturbed gradients
                with torch.enable_grad():
                    perturbed_returns = closure()
                    perturbed_loss = perturbed_returns if isinstance(perturbed_returns, torch.Tensor) else perturbed_returns[0]
                    perturbed_grads = torch.autograd.grad(perturbed_loss, self._params)
                
                # Compute approximate Hvp using finite differences
                Hvs = [pg - g for pg, g in zip(perturbed_grads, grads)]

            # Initialize or update preconditioner
            if self.state['Q_exprs'] is None:
                # Initialize with automatic scaling based on gradient and Hvp magnitudes
                self.state['Q_exprs'] = [
                    init_Q_exprs(
                        h, 
                        (torch.mean((torch.abs(v))**2))**(1/4) * (torch.mean((torch.abs(h))**4))**(-1/8),
                        max_size, 
                        max_skew
                    ) for v, h in zip(vs, Hvs)
                ]
            
            # Update preconditioner factors
            for Q_exprs, v, h in zip(self.state['Q_exprs'], vs, Hvs):
                update_precond_kron_math_(*Q_exprs, v, h, lr_precond, step_normalizer, self._tiny)
        else:
            # Only compute gradients if not updating preconditioner
            with torch.enable_grad():
                closure_returns = closure()
                loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                grads = torch.autograd.grad(loss, self._params)
            vs = None

        # Handle gradient whitening preconditioner
        if ((precond_type != "Newton") and 
            (torch.rand([]) < precond_prob or self.state['Q_exprs'] is None)):
            
            if self.state['Q_exprs'] is None:
                # Initialize preconditioner for whitening with automatic scaling
                self.state['Q_exprs'] = [
                    init_Q_exprs(
                        g,
                        (torch.mean((torch.abs(g))**4))**(-1/8),
                        max_size,
                        max_skew
                    ) for g in grads
                ]
            
            # Update whitening preconditioner using damped pairs
            for Q_exprs, g in zip(self.state['Q_exprs'], grads):
                v, g_damped = damped_pair_vg(g)
                update_precond_kron_math_(*Q_exprs, v, g_damped, lr_precond, step_normalizer, self._tiny)

        # Apply momentum if enabled
        if momentum > 0:
            if self.state['momentum_buffer'] is None:
                self.state['momentum_buffer'] = [(1 - momentum)*g for g in grads]
            else:
                for mbuf, g in zip(self.state['momentum_buffer'], grads):
                    mbuf.mul_(momentum).add_(g, alpha=1 - momentum)
            
            pre_grads = [
                precond_grad_kron_math(*Q_exprs, m) 
                for Q_exprs, m in zip(self.state['Q_exprs'], self.state['momentum_buffer'])
            ]
        else:
            # No momentum - just precondition gradients directly
            self.state['momentum_buffer'] = None
            pre_grads = [
                precond_grad_kron_math(*Q_exprs, g)
                for Q_exprs, g in zip(self.state['Q_exprs'], grads)
            ]

        # Apply gradient clipping if enabled
        if grad_clip is not None:
            # Compute total gradient norm
            grad_norm = torch.sqrt(
                torch.abs(sum(torch.sum(g*g.conj()) for g in pre_grads))
            ) + self._tiny
            
            # Scale learning rate if needed
            actual_lr = lr * min(grad_clip/grad_norm, 1.0)
        else:
            actual_lr = lr

        # Update parameters
        if exact_hvp or vs is None or precond_type != "Newton":
            # Standard update
            for param, grad in zip(self._params, pre_grads):
                param.sub_(actual_lr * grad)
        else:
            # Include perturbation removal when using finite differences
            for param, grad, v in zip(self._params, pre_grads, vs):
                param.sub_(actual_lr * grad + v)

        return closure_returns