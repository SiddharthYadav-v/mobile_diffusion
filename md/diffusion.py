import torch
from torch import nn
from torch.nn import functional as F

from .decoder import SelfAttentionBlock

import math

from tqdm import trange

class ForwardDiffusion(nn.Module):
    
    def __init__(self, beta_1 : float, beta_T : float, T : int):
        super().__init__()
        betas = torch.linspace(beta_1, beta_T, T).requires_grad_(False)
        alphas = (1.0 - betas).requires_grad_(False)
        alpha_bars = torch.cumprod(alphas, -1).requires_grad_(False)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        
    def forward(self, X : torch.Tensor, t : torch.Tensor, noise : torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(X.shape[0], 1, 1, 1).to(X.device)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bars[t]).view(X.shape[0], 1, 1, 1).to(X.device)
        return sqrt_alpha_bar * X + sqrt_one_minus_alpha_bar * noise


class TimeEncoding(nn.Module):
    
    def __init__(self, time_dim : int, d_model : int, d_time : int):
        super().__init__()
        self.time_dim = time_dim
        self.d_model = d_model
        positional_encodings = torch.zeros(time_dim, d_model)
        positions = torch.arange(0, time_dim).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        positional_encodings[:, 0::2] = torch.sin(positions * div_term)
        positional_encodings[:, 1::2] = torch.cos(positions * div_term)
        self.temb = nn.Sequential(
            nn.Embedding.from_pretrained(positional_encodings),
            nn.Linear(d_model, d_time),
            nn.SiLU(),
            nn.Linear(d_time, d_time)
        )
        
    def forward(self, t : torch.Tensor) -> torch.Tensor:
        return self.temb(t)

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels : int, out_channels : int, d_time : int):
        super().__init__()
        self.first_block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        )
        self.time_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_time, out_channels)
        )
        self.second_block = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        )
        self.group_norm = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU()
        )
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, X : torch.Tensor, temb : torch.Tensor) -> torch.Tensor:
        temb = self.time_embedding(temb)
        residue = X
        X = self.first_block(X) + temb.unsqueeze(-1).unsqueeze(-1)
        X = self.second_block(X)
        X = self.group_norm(X)
        return X + self.residual_layer(residue)

class DownSample(nn.Module):
    
    def __init__(self, in_channels : int, out_channels : int):
        super().__init__()
        self.f_ = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1)
        )

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        return self.f_(X)

class SwitchSequential(nn.Sequential):
    
    def forward(self, X : torch.Tensor, temb : torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, ResidualBlock):
                X = layer(X, temb)
            else:
                X = layer(X)
        return X

class UpSample(nn.Module):
    
    def __init__(self, in_channels : int, out_channels : int):
        super().__init__()
        self.f_ = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = "nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
        )
        
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        return self.f_(X)

class UNET(nn.Module):
    
    def __init__(self, in_channels : int, time_dim : int, d_model : int, d_time : int):
        super().__init__()
        self.time_embedding = TimeEncoding(time_dim, d_model, d_time)
        self.head = nn.Conv2d(in_channels, 256, kernel_size = 3, stride = 1, padding = 1)
        self.encoders = nn.ModuleList([
            SwitchSequential(ResidualBlock(256, 256, d_time)),
            DownSample(256, 256),
            SwitchSequential(ResidualBlock(256, 512, d_time), SelfAttentionBlock(512)),
            DownSample(512, 512),
            SwitchSequential(ResidualBlock(512, 1024, d_time), SelfAttentionBlock(1024)),
            DownSample(1024, 1024),
            SwitchSequential(ResidualBlock(1024, 1024, d_time), SelfAttentionBlock(1024)),
            DownSample(1024, 1024)
        ])
        self.bottle_neck = SwitchSequential(
            ResidualBlock(1024, 1024, d_time),
            SelfAttentionBlock(1024),
            ResidualBlock(1024, 1024, d_time)
        )
        self.decoders = nn.ModuleList([
            UpSample(1024, 1024),
            SwitchSequential(ResidualBlock(2048, 1024, d_time), SelfAttentionBlock(1024)),
            UpSample(1024, 1024),
            SwitchSequential(ResidualBlock(2048, 1024, d_time), SelfAttentionBlock(1024)),
            UpSample(1024, 1024),
            SwitchSequential(ResidualBlock(1536, 768, d_time), SelfAttentionBlock(768)),
            UpSample(768, 768),
            SwitchSequential(ResidualBlock(1024, 512, d_time))
        ])
        self.tail = nn.Sequential(
            nn.GroupNorm(32, 512),
            nn.Conv2d(512, in_channels, kernel_size = 3, stride = 1, padding = 1)
        )
        
    def forward(self, X : torch.Tensor, t : torch.LongTensor) -> torch.Tensor:
        temb = self.time_embedding(t)
        X = self.head(X)
        enc_outputs = []
        for layer in self.encoders:
            if isinstance(layer, SwitchSequential):
                X = layer(X, temb)
                enc_outputs.append(X.clone())
            else:
                X = layer(X)
        X = self.bottle_neck(X, temb)
        for layer in self.decoders:
            if isinstance(layer, UpSample):
                X = layer(X)
            else:
                X = layer(torch.cat([enc_outputs.pop(), X], dim = 1), temb)
        X = self.tail(X)
        return X

class DDPM(nn.Module):
    
    def __init__(self, in_channels, time_dim, d_model, d_time):
        super().__init__()
        self.time_steps = time_dim
        self.forw = ForwardDiffusion(1e-4, 0.02, time_dim)
        self.back = UNET(in_channels, time_dim, d_model, d_time)

    def forward(self, X, t, noise):
        X = self.forw(X, t, noise)
        eps = self.back(X, t)
        return eps

    def sample(self, X, device):
        res = []
        for t in trange(300, 0, -1):
            z = torch.randn(*X.shape) if t > 1 else torch.zeros(*X.shape)
            alpha_t = self.forw.alphas[t - 1].to(device)
            alpha_bar_t = self.forw.alpha_bars[t - 1].to(device)
            beta_t = self.forw.betas[t - 1].to(device)
            prev_alpha_bar_t = self.forw.alpha_bars[t - 2].to(device) if t > 1 else self.forw.alphas[0].to(device)
            beta_bar_t = ((1 - prev_alpha_bar_t)/(1 - alpha_bar_t)) * beta_t
            sigma_t = torch.sqrt(beta_bar_t)
            time_tensor = (torch.ones(X.shape[0]) * (t - 1)).to(device).long()
            X = 1 / torch.sqrt(alpha_t) * (X - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * self.back(X, time_tensor)) + sigma_t * z.to(device)
            res.append(X)
        return res