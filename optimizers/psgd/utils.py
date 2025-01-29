import torch
import opt_einsum

def damped_pair_vg(g, damp=2**(-13)):
    """
    Instead of return (v, g), it returns pair
        (v, g + sqrt(eps)*mean(abs(g))*v)
    such that the covariance matrix of the modified g is lower bound by
        eps * (mean(abs(g)))**2 * I
    This should damp the preconditioner to encourage numerical stability.
    The default amount of damping is 2**(-13), slightly smaller than sqrt(eps('single')). 
    
    If v is integrated out, let's just use the modified g; 
    If hvp is used, recommend to use L2 regularization to lower bound the Hessian, although this method also works. 

    Please check example
        https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_with_finite_precision_arithmetic.py
    for the rationale to set default damping level to 2**(-13). 
    """
    v = torch.randn_like(g)
    return (v, g + damp*torch.mean(torch.abs(g))*v)

def init_Q_exprs(t, Scale, max_size, max_skew):
    """
    Initialize preconditioner Q and contraction expressions for a tensor t.
    
    Args:
        t: Input tensor
        Scale: Initial scale for preconditioner (if None, computed automatically)
        max_size: Maximum size for triangular matrices before switching to diagonal
        max_skew: Maximum ratio between dimensions before switching to diagonal
    
    Returns:
        [Q, (exprA, exprGs, exprP)]:
            Q: List of preconditioner matrices
            exprA: Expression for computing A
            exprGs: List of expressions for computing gradients
            exprP: Expression for computing preconditioned gradient
    """
    shape = t.shape
    if len(shape) == 0:  # Scalar case
        scale = Scale if Scale else (1/(t*t.conj()))**0.25
        Q = [scale * torch.ones_like(t)]
        exprA = opt_einsum.contract_expression(",->", Q[0].shape, t.shape)
        exprP = opt_einsum.contract_expression(",,->", Q[0].shape, Q[0].shape, t.shape)
        exprGs = [opt_einsum.contract_expression(",->", t.shape, t.shape)]
        return [Q, (exprA, exprGs, exprP)]

    if len(shape) > 26:
        raise ValueError("Tensor dimension exceeds 26 (Einstein notation limit)")
        
    scale = Scale ** (1/len(shape)) if Scale else 1.0
    
    Q = []
    exprGs = []
    piece1A, piece2A, piece3A = [], "", ""  # For exprA
    piece1P, piece2P, piece3P, piece4P = [], [], "", ""  # For exprP
    min_tri_size, min_tri_loc = float("inf"), 0
    
    # Process each dimension
    for i, size in enumerate(shape):
        if size == 1 or size > max_size or size**2 > max_skew * t.numel():
            # Use diagonal matrix for this dimension
            Q.append(scale * torch.ones(size, dtype=t.dtype, device=t.device))
            
            # Build expression pieces
            piece1A.append(opt_einsum.get_symbol(i))
            piece2A = piece2A + opt_einsum.get_symbol(i)
            piece3A = piece3A + opt_einsum.get_symbol(i)
            
            piece1P.append(opt_einsum.get_symbol(i + 26))
            piece2P.append(opt_einsum.get_symbol(i + 26))
            piece3P = piece3P + opt_einsum.get_symbol(i + 26)
            piece4P = piece4P + opt_einsum.get_symbol(i + 26)
            
            # Generate gradient expressions
            piece1 = "".join([opt_einsum.get_symbol(i+26) if j==i else opt_einsum.get_symbol(j) 
                            for j in range(len(shape))])
            subscripts = f"{piece1},{piece1}->{opt_einsum.get_symbol(i+26)}"
            exprGs.append(opt_einsum.contract_expression(subscripts, t.shape, t.shape))
        else:
            # Use triangular matrix
            Q.append(scale * torch.eye(size, dtype=t.dtype, device=t.device))
            if size < min_tri_size:
                min_tri_size, min_tri_loc = size, i
            
            piece1A.append(opt_einsum.get_symbol(i) + opt_einsum.get_symbol(i + 26))
            piece2A = piece2A + opt_einsum.get_symbol(i + 26)
            piece3A = piece3A + opt_einsum.get_symbol(i)
            
            a, b, c = (opt_einsum.get_symbol(i), 
                      opt_einsum.get_symbol(i + 26),
                      opt_einsum.get_symbol(i + 805))
            piece1P.append(a + b)
            piece2P.append(a + c)
            piece3P = piece3P + c
            piece4P = piece4P + b
            
            # Generate gradient expressions for triangular case
            piece1 = "".join([opt_einsum.get_symbol(i+26) if j==i else opt_einsum.get_symbol(j) 
                            for j in range(len(shape))])
            piece2 = "".join([opt_einsum.get_symbol(i+805) if j==i else opt_einsum.get_symbol(j) 
                            for j in range(len(shape))])
            subscripts = f"{piece1},{piece2}->{opt_einsum.get_symbol(i+26)}{opt_einsum.get_symbol(i+805)}"
            exprGs.append(opt_einsum.contract_expression(subscripts, t.shape, t.shape))

    # Automatically determine scale if not provided
    if Scale is None:
        if min_tri_size**2 <= t.numel():
            # Whiten using smallest triangular matrix
            t1 = t.transpose(min_tri_loc, 0)
            t1 = t1.reshape(min_tri_size, -1)
            D, U = torch.linalg.eigh(t1 @ t1.H)
            D = torch.clamp(D, min=1e-4*torch.max(D))
            _, R = torch.linalg.qr(U @ (U.H * torch.pow(D[:,None], -1/4)))
            Q[min_tri_loc] = R * (R.diag().real.sign())[:,None]
        else:
            # Normalize using vector norm
            Q[min_tri_loc] *= (torch.linalg.vector_norm(t))**(-1/2)

    # Build final expressions
    subscripts = ",".join(piece1A) + "," + piece2A + "->" + piece3A
    exprA = opt_einsum.contract_expression(subscripts, *[q.shape for q in Q], t.shape)
    
    subscripts = ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
    exprP = opt_einsum.contract_expression(subscripts, *[q.shape for q in Q], *[q.shape for q in Q], t.shape)

    return [Q, (exprA, tuple(exprGs), exprP)]

def update_precond_kron_math_(Q, exprs, V, G, step, step_normalizer, tiny):
    """
    Update Kronecker product preconditioner Q with (vector, hessian-vector product) pair (V, G).
    
    Args:
        Q: List of preconditioner matrices
        exprs: Tuple of (exprA, exprGs, exprP) expressions
        V: Random vector (can be None if integrated out)
        G: Gradient or Hessian-vector product
        step: Learning rate for preconditioner update
        step_normalizer: Method for normalizing step size ('1st' or '2nd')
        tiny: Small constant for numerical stability
    """
    def triangular_inv(A):
        """Compute inverse of triangular matrix"""
        I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        return torch.linalg.solve_triangular(A, I, upper=True)
    
    def solve_triangular_right(X, A):
        """Solve X @ A^(-1)"""
        if X.dim() > 1:
            return torch.linalg.solve_triangular(A, X, upper=True, left=False)
        return torch.linalg.solve_triangular(A, X[None,:], upper=True, left=False)[0]

    order = G.dim()
    
    # Balance dynamic range of Q factors
    if order > 1 and torch.rand([]) < 0.01:
        norms = [torch.max(torch.abs(q)) for q in Q]
        gmean = (torch.cumprod(torch.stack(norms), dim=0)[-1])**(1/order)
        for i, q in enumerate(Q):
            q.mul_(gmean/norms[i])

    exprA, exprGs, _ = exprs
    A = exprA(*Q, G)

    if V is not None:
        invQhinvQ, trace_invQhinvQ = None, None
        p = list(range(order))
        conjB = torch.permute(V.conj(), p[1:] + p[:1])
        
        for i, q in enumerate(Q):
            conjB = conjB/q if q.dim()<2 else solve_triangular_right(conjB, q)
            if i < order - 1:
                conjB = torch.transpose(conjB, i, order - 1)
    else:
        conjB = None
        invQ = [1/q if q.dim()<2 else triangular_inv(q) for q in Q]
        invQhinvQ = [q.conj()*q if q.dim()<2 else q.H@q for q in invQ]
        trace_invQhinvQ = [torch.sum(q) if q.dim()<2 else torch.trace(q) for q in invQhinvQ]

    # Update each Q factor
    for i, q in enumerate(Q):
        term1 = exprGs[i](A, A.conj())
        if conjB is not None:
            term2 = exprGs[i](conjB.conj(), conjB)
        else:
            term2 = 1.0
            for j, trace in enumerate(trace_invQhinvQ):
                term2 = term2 * (trace if i!=j else invQhinvQ[i])
        
        if step_normalizer == "2nd":
            if q.dim() < 2:
                q.sub_(step/(torch.max(torch.abs(term1 + term2)) + tiny) * 
                       (term1 - term2) * q)
            else:
                q.sub_(step/(torch.linalg.norm(term1 + term2) + tiny) * 
                       torch.triu(term1 - term2) @ q)
        else:
            if q.dim() < 2:
                q.sub_(step/(torch.max(torch.abs(term1 - term2)) + tiny) * 
                       (term1 - term2) * q)
            else:
                grad = torch.triu(term1 - term2)
                q.sub_(step/(torch.linalg.norm(grad) + tiny) * grad @ q)

def precond_grad_kron_math(Q, exprs, G):
    """Apply Kronecker product preconditioner to gradient"""
    return exprs[-1](*[q.conj() for q in Q], *Q, G)