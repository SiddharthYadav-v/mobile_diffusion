import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads : int, d_embd : int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        self.in_proj = nn.Linear(d_embd, 3 * d_embd, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embd, d_embd, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embd // n_heads

    def forward(self, x : torch.Tensor, causal_mask : bool = False) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        intermediate_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim = -1)

        q = q.view(intermediate_shape).transpose(1, 2)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)

        wei = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(wei, dtype = torch.bool).triu(1)
            wei = wei.masked_fill(mask, -torch.inf)
        wei /= math.sqrt(self.d_head)
        wei = F.softmax(wei, dim = -1)

        output = wei @ v
        output = output.transpose(1, 2)
        output = output.reshape(batch_size, seq_len, dim)
        output = self.out_proj(output)
        
        return output
    
class CrossAttention(nn.Module):
    
    def __init__(self, n_heads : int, d_embd : int, d_cross : int, in_proj_bias : bool = True, out_proj_bias : bool = True):
        super().__init__()
        self.q_proj = nn.Linear(d_embd, d_embd, bias = in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embd, bias = in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embd, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embd, d_embd, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embd // n_heads
        
    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        intermediate_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        
        q = q.view(intermediate_shape).transpose(1, 2)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)
        
        wei = q @ k.transpose(-1, -2)
        wei = wei / math.sqrt(self.d_head)
        wei = F.softmax(wei, dim = -1)
        output = wei @ v
        
        output = output.transpose(1, 2).contiguous()
        output = output.view(x.shape)
        
        output = self.out_proj(output)
        
        return output