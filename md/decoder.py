import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention

class SelfAttentionBlock(nn.Module):

    def __init__(self, channels : int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(4, channels)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        residue = x
        x = self.group_norm(x)
        
        n, c, h, w = x.shape
        x = x.view(n, c, h * w)
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view(n, c, h, w)

        return x + residue

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)

        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 0)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        residue = x

        x = self.group_norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.group_norm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + self.residual_layer(residue)
    
class Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size = 1, padding = 0),
            nn.Conv2d(4, 512, kernel_size = 3, padding = 1),
            ResidualBlock(512, 512),
            SelfAttentionBlock(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size = 3, padding = 1)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x /= 0.18215
        
        for module in self:
            x = module(x)
        
        return x
