import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention, CrossAttention
import math

class ForwardProcess(nn.Module):
    def __init__(self, betas):
        super().__init__()
        alphas = [1 - beta for beta in betas]
        alpha_bars = []
        for alpha in alphas:
            if len(alpha_bars) == 0:
                alpha_bars.append(alpha)
            else:
                alpha_bars.append(alpha_bars[-1] * alpha)
        alphas = torch.tensor(alphas).requires_grad_(False)
        alpha_bars = torch.tensor(alpha_bars).requires_grad_(False)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("betas", torch.tensor(betas))

    def forward(self, X, t, device = "cpu"):
        noise = torch.randn(*X.shape).to(device).requires_grad_(False)
        return torch.sqrt(self.alpha_bars[t]).view(X.shape[0], 1, 1, 1) * X + torch.sqrt(1 - self.alpha_bars[t]).view(X.shape[0], 1, 1, 1) * noise, noise

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
        positional_encodings = positional_encodings
        self.temb = nn.Sequential(
            nn.Embedding.from_pretrained(positional_encodings),
            nn.Linear(d_model, d_time),
            nn.SiLU(),
            nn.Linear(d_time, d_time)
        )
        
    def forward(self, t):
        return self.temb(t)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_time, self_attn, cross_attn, cond_dim):
        super().__init__()
        self.first_block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )
        self.time_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(d_time, out_channels, 1, 1, 0)
        )
        self.second_block = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.res = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.proj = nn.Conv2d(cond_dim, out_channels, 1, 1, 0)
        self.self_attn = self_attn
        self.cross_attn = cross_attn

    def forward(self, X, cond, temb):
        temb = self.time_embedding(temb.view(*temb.shape, 1, 1))
        res = self.res(X)
        out = self.first_block(X) + temb
        out = res + self.second_block(out)
        out = out + self.self_attn(out)
        cond = cond.view(*cond.shape, 1, 1)
        cond = self.proj(cond)
        if self.cross_attn is not None:
            return out + self.cross_attn(out, cond)
        return out

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, act):
        super().__init__()
        self.f_ = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            act
        )

    def forward(self, X):
        return self.f_(X)

class DownBlock(nn.Module):
    def __init__(self, res_block, down_block):
        super().__init__()
        self.res_block = res_block
        self.down_block = down_block

    def forward(self, X, cond, temb):
        X = self.res_block(X, cond, temb)
        return self.down_block(X)



class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, act):
        super().__init__()
        self.f_ = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.GroupNorm(32, out_channels),
            act
        )
        
    def forward(self, X):
        return self.f_(X)

class UpBlock(nn.Module):
    def __init__(self, up_block, res_block):
        super().__init__()
        self.up_block = up_block
        self.res_block = res_block

    def forward(self, X, Y, cond, temb):
        X = self.up_block(X)
        return self.res_block(torch.cat([X, Y], dim = 1), cond, temb)

class UNET(nn.Module):
    def __init__(self, time_dim, d_model, d_time, init, in_channels, cond_dim):
        super().__init__()
        self.time_embedding = TimeEncoding(time_dim, d_model, d_time)
        self.conditional_embedding = nn.Embedding(cond_dim, cond_dim)
        self.head = nn.Conv2d(in_channels, init, 3, 1, 1)
        self.down_blocks = []
        mult = 1
        for i in range(3):
            self.down_blocks.append(DownBlock(
                ResidualBlock(init * mult, init * mult, d_time, SelfAttention(init * mult, init * mult) if i >= 1 else nn.Identity(), CrossAttention(init * mult, init * mult) if i == 2 else None, cond_dim),
                DownSample(init * mult, init * mult * 2, nn.SiLU())
            ))
            mult *= 2
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.bottleneck = nn.ModuleList([
            ResidualBlock(init * mult, init * mult, d_time, SelfAttention(init * mult, init * mult), CrossAttention(init * mult, init * mult), cond_dim),
            ResidualBlock(init * mult, init * mult, d_time, SelfAttention(init * mult, init * mult), CrossAttention(init * mult, init * mult), cond_dim)
        ])
        self.up_blocks = []
        for i in range(3):
            self.up_blocks.append(UpBlock(
                UpSample(init * mult, init * mult // 2, nn.SiLU()),
                ResidualBlock(init * mult, init * mult // 2, d_time, SelfAttention(init * mult // 2, init * mult // 2) if i <= 1 else nn.Identity(), CrossAttention(init * mult // 2, init * mult // 2) if i == 0 else None, cond_dim)
            ))
            mult //= 2
        self.up_blocks = nn.ModuleList(self.up_blocks)
        self.tail = nn.Conv2d(init, in_channels, 3, 1, 1)

    def forward(self, X, c, t):
        t = self.time_embedding(t)
        c = self.conditional_embedding(c)
        X = self.head(X)
        Y = [None for _ in range(len(self.down_blocks))]
        for i in range(len(self.down_blocks)):
            Y[i] = X.clone()
            X = self.down_blocks[i](X, c, t)
        for block in self.bottleneck:
            X = block(X, c, t)
        for i in range(len(self.up_blocks)):
            X = self.up_blocks[i](X, Y[len(self.down_blocks) - 1 - i], c, t)
        return self.tail(X)

class DDPM(nn.Module):
    def __init__(self, time_dim, d_model, d_time, init, in_channels, cond_dim):
        super().__init__()
        m = (0.02 - 1e-4) / (time_dim - 1)
        c = 1e-4 - m
        betas = [m * x + c for x in range(1, time_dim + 1)]
        self.time_steps = time_dim
        self.forw = ForwardProcess(betas)
        self.back = UNET(time_dim, d_model, d_time, init, in_channels, cond_dim)

    def forward(self, X, c, t, device):
        noisy_img, noise = self.forw(X, t, device)
        eps = self.back(noisy_img, c, t)
        return eps, noise

    def sample(self, shape, c, device):
        X = torch.randn(shape).to(device)
        for t in range(self.time_steps, 0, -1):
            print ("\r" + str(t), end = "")
            z = torch.randn(shape) if t > 1 else torch.zeros(shape)
            alpha_t = self.forw.alphas[t - 1].to(device)
            alpha_bar_t = self.forw.alpha_bars[t - 1].to(device)
            beta_t = self.forw.betas[t - 1].to(device)
            prev_alpha_bar_t = self.forw.alpha_bars[t - 2] if t > 1 else self.forw.alphas[0]
            beta_bar_t = ((1 - prev_alpha_bar_t)/(1 - alpha_bar_t)) * beta_t
            sigma_t = torch.sqrt(beta_bar_t)
            time_tensor = (torch.ones(shape[0]) * (t - 1)).to(device).int()
            X = 1 / torch.sqrt(alpha_t) * (X - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * self.back(X, c, time_tensor)) + sigma_t * z.to(device)
        X -= torch.min(X)
        X /= torch.max(X)
        return X