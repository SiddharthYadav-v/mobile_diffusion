import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention

class CLIPEmbedding(nn.Module):
    
    def __init__(self, n_vocab : int, n_embd : int, n_tokens : int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.positional_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))
        
    def forward(self, tokens : torch.LongTensor) -> torch.FloatTensor:
        x = self.token_embedding(tokens)
        x += self.positional_embedding
        
        return x

class CLIPLayer(nn.Module):
    
    def __init__(self, n_head : int, n_embd : int):
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        residue = x
        x = self.attention(x, causal_mask = True)
        x += residue
        x = self.layer_norm1(x)
        
        residue = x
        x = self.linear1(x)
        x = x * F.sigmoid(1.702 * x)
        x = self.linear2(x)
        x += residue
        x = self.layer_norm2(x)
        
        return x

class CLIP(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])
        
        self.layer_norm = nn.LayerNorm(768)
        
    def forward(self, tokens : torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        state = self.embedding(tokens)
        
        for layer in self.layers:
            state = layer(state)
        output = self.layer_norm(state)
        
        return output