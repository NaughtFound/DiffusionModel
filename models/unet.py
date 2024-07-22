import torch
from torch import nn
import torch.nn.functional as F


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

        self.bottle_neck = nn.Sequential(
            DoubleConv(256, 512),
            DoubleConv(512, 512),
            DoubleConv(512, 256),
        )

    def pos_encoding(self, t: torch.Tensor, channels: int) -> torch.Tensor:
        inv_freq = 1.0 / (
            1e+4 ** (
                torch.arange(0, channels//2, device=t.device).float() / channels)
        )

        pos_enc_a = torch.sin(t.repeat(1, channels//2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels//2) * inv_freq)

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
            x = self.up[i](x, x_l[i+1], t)
            x = self.up_attention[i](x)

        return self.out(x)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self.time_encoding(t)

        return self.encode_decode(x, t)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, residual: bool = False) -> None:
        super().__init__()

        self.residual = residual

        if not mid_channels:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(
                num_groups=1,
                num_channels=mid_channels
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(
                num_groups=1,
                num_channels=out_channels
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return F.gelu(x + self.conv(x))
        else:
            return self.conv(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, size: int) -> None:
        super().__init__()

        self.channels = channels
        self.size = size

        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=4,
            batch_first=True
        )

        self.ln = nn.LayerNorm(
            normalized_shape=[channels]
        )

        self.ff = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=[channels]
            ),
            nn.Linear(
                in_features=channels,
                out_features=channels
            ),
            nn.GELU(),
            nn.Linear(
                in_features=channels,
                out_features=channels
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.channels, self.size**2).swapaxes(1, 2)

        x_ln = self.ln(x)

        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff(attention_value) + attention_value

        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256) -> None:
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True
        )

        self.conv = nn.Sequential(
            DoubleConv(
                in_channels=in_channels,
                out_channels=in_channels,
                residual=True
            ),
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                mid_channels=in_channels//2
            )
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                in_features=emb_dim,
                out_features=out_channels
            )
        )

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        x = torch.cat([skip_x, x], dim=1)

        x = self.conv(x)

        emb = self.emb_layer(t)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1])

        return x + emb


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels=in_channels,
                out_channels=in_channels,
                residual=True
            ),
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
            )
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                in_features=emb_dim,
                out_features=out_channels
            )
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        emb = self.emb_layer(t)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1])

        return x + emb
