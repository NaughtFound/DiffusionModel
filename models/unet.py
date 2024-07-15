import torch
from torch import nn


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        time_dim: int = 256
    ) -> None:
        super().__init__()

        self.time_dim = time_dim

        self.inc = DoubleConv(in_channels, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

        self.down = nn.ModuleList([
            Down(64, 128),
            Down(128, 256),
            Down(256, 256)
        ])

        self.up = nn.ModuleList([
            Up(512, 128),
            Up(256, 64),
            Up(128, 64),
        ])

        self.down_attention = nn.ModuleList([
            SelfAttention(128, 32),
            SelfAttention(256, 16),
            SelfAttention(256, 8),
        ])

        self.up_attention = nn.ModuleList([
            SelfAttention(128, 16),
            SelfAttention(64, 32),
            SelfAttention(64, 64),
        ])

        self.bottle_neck = nn.ModuleList([
            DoubleConv(256, 512),
            DoubleConv(512, 512),
            DoubleConv(512, 256),
        ])

    def pos_encoding(self, t: torch.Tensor, channels: int) -> torch.Tensor:
        inv_freq = 1.0 / (
            1e+4 ** (
                torch.arange(0, channels, 2, device=self.device).float() / channels)
        )

        pos_enc_a = torch.sin(t.repeat(1, channels//2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels//2) * inv_freq)

        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1).to(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x = self.inc(x)
        x_l = [x]

        for i in range(len(self.down)):
            x_i = self.down[i](x_l[-1], t)
            x_i = self.down_attention[i](x_i)
            x_l.append(x_i)

        x_l[-1] = self.bottle_neck(x_l[-1])

        x = x_l[-1]
        x_l = x_l[::-1]

        for i in range(len(self.up)):
            x = self.up[i](x, x_l[i+1], t)
            x = self.up_attention[i](x)

        return self.out(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()


class SelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
