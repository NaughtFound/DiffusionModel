import torch
from torch import nn
import modules as m


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        time_dim: int = 256,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim

        self.inc = m.DoubleConv(in_channels, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

        self.down = nn.ModuleList(
            [
                m.Down(64, 128),
                m.Down(128, 256),
                m.Down(256, 256),
            ]
        )

        self.up = nn.ModuleList(
            [
                m.Up(512, 128),
                m.Up(256, 64),
                m.Up(128, 64),
            ]
        )

        self.down_attention = nn.ModuleList(
            [
                m.SelfAttention(128, 32),
                m.SelfAttention(256, 16),
                m.SelfAttention(256, 8),
            ]
        )

        self.up_attention = nn.ModuleList(
            [
                m.SelfAttention(128, 16),
                m.SelfAttention(64, 32),
                m.SelfAttention(64, 64),
            ]
        )

        self.bottle_neck = nn.Sequential(
            m.DoubleConv(256, 512),
            m.DoubleConv(512, 512),
            m.DoubleConv(512, 256),
        )

    def pos_encoding(self, t: torch.Tensor, channels: int) -> torch.Tensor:
        inv_freq = 1.0 / (
            1e4 ** (torch.arange(0, channels // 2, device=t.device).float() / channels)
        )

        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)

        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1).to(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        return t

    def encode_decode(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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
            x = self.up[i](x, x_l[i + 1], t)
            x = self.up_attention[i](x)

        return self.out(x)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self.time_encoding(t)

        return self.encode_decode(x, t)
