import math
import torch
from torch import nn
from torch.autograd import Function
import linear_attention_cuda

class LinearAttentionFunction(Function):
    @staticmethod
    def forward(ctx, queries, keys, values):
        # Save inputs for backward pass
        ctx.save_for_backward(queries, keys, values)
        
        # Run CUDA kernel
        output = linear_attention_cuda.forward(queries, keys, values)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        queries, keys, values = ctx.saved_tensors
        # Implement backward pass (not shown for brevity)
        return grad_queries, grad_keys, grad_values

class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Apply optimized linear attention
        attn = LinearAttentionFunction.apply(q, k, v)
        
        x = attn.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x