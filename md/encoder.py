import torch
from torch import nn
from torch.nn import functional as F
from .decoder import SelfAttentionBlock, ResidualBlock

class Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size = 3, padding = 1),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 0),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 0),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 0),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            SelfAttentionBlock(512),
            ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size = 3, padding = 1),
            nn.Conv2d(8, 8, kernel_size = 1, padding = 0)
        )
        self.kl = 0

    def forward(self, x : torch.Tensor, noise : torch.Tensor) -> torch.Tensor:
        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        mean, log_var = torch.chunk(x, 2, dim = 1)
        log_var = torch.clamp(log_var, -30, 20)
        var = log_var.exp()
        std_dev = var.sqrt()
        x = mean + std_dev * noise
        
        self.kl = -1 * (0.5 + torch.log(std_dev) - std_dev.pow(2) - mean.pow(2)).mean()
        
        x *= 0.18215
        
        return x
