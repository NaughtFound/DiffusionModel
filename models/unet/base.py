import torch
from torch import nn
from . import modules as m


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 64,
        out_channels: int = 3,
        time_dim: int = 256,
        emb_dim: int = 256,
        features: list[int] = [128, 256],
        neck_features: list[int] = [512],
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.emb_dim = emb_dim

        self.inc = m.DoubleConv(in_channels, mid_channels)
        self.out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

        self.down = nn.ModuleList()
        self.down_attention = nn.ModuleList()
        self.bottle_neck = nn.Sequential()
        self.up = nn.ModuleList()
        self.up_attention = nn.ModuleList()

        f_before = mid_channels
        for f in features:
            self.down.append(m.Down(f_before, f, emb_dim))
            self.down_attention.append(m.SelfAttention(f))
            f_before = f

        self.down.append(m.Down(features[-1], features[-1], emb_dim))
        self.down_attention.append(m.SelfAttention(features[-1]))

        n_before = features[-1]

        for n in neck_features:
            self.bottle_neck.append(m.DoubleConv(n_before, n))
            n_before = n

        self.bottle_neck.append(m.DoubleConv(n_before, features[-1]))

        for f in reversed(features):
            self.up.append(m.Up(f * 2, f // 2, emb_dim))
            self.up_attention.append(m.SelfAttention(f // 2))

        self.up.append(m.Up(f, mid_channels))
        self.up_attention.append(m.SelfAttention(mid_channels))

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
