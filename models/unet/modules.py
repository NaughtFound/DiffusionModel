import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        residual: bool = False,
    ) -> None:
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
                bias=False,
            ),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            nn.GELU(),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return F.gelu(x + self.conv(x))
        else:
            return self.conv(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_dim: int,
        self_attention: bool = False,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.emb_dim = emb_dim
        self.self_attention = self_attention

        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            kdim=emb_dim,
            vdim=emb_dim,
            num_heads=4,
            batch_first=True,
        )

        if self_attention:
            self.ln = nn.LayerNorm(normalized_shape=[channels])
        else:
            self.ln_q = nn.LayerNorm(normalized_shape=[channels])
            self.ln_kv = nn.LayerNorm(normalized_shape=[emb_dim])

        self.ff = nn.Sequential(
            nn.LayerNorm(normalized_shape=[channels]),
            nn.Linear(in_features=channels, out_features=channels),
            nn.GELU(),
            nn.Linear(in_features=channels, out_features=channels),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        batch_size = q.size(0)
        size = int(math.sqrt(q.numel() // self.channels // batch_size))

        q = q.view(batch_size, self.channels, -1).swapaxes(1, 2)
        kv = kv.view(batch_size, self.emb_dim, -1).swapaxes(1, 2)

        if self.self_attention:
            q_ln = self.ln(q)
            kv_ln = self.ln(kv)
        else:
            q_ln = self.ln_q(q)
            kv_ln = self.ln_kv(kv)

        attention_value, _ = self.mha(q_ln, kv_ln, kv_ln)
        attention_value = attention_value + q
        attention_value = self.ff(attention_value) + attention_value

        return attention_value.swapaxes(2, 1).view(
            -1,
            self.channels,
            size,
            size,
        )


class SelfAttention(CrossAttention):
    def __init__(self, channels):
        super().__init__(channels, channels, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x, x)


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int = 256,
    ) -> None:
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv = nn.Sequential(
            DoubleConv(
                in_channels=in_channels,
                out_channels=in_channels,
                residual=True,
            ),
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                mid_channels=in_channels // 2,
            ),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_dim, out_features=out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip_x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)

        x = torch.cat([skip_x, x], dim=1)

        x = self.conv(x)

        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + emb


class Down(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int = 256,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels=in_channels,
                out_channels=in_channels,
                residual=True,
            ),
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
            ),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_dim, out_features=out_channels),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + emb
