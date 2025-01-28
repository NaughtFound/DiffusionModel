import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SelfAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        size: int,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.size = size

        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=4, batch_first=True
        )

        self.ln = nn.LayerNorm(normalized_shape=[channels])

        self.ff = nn.Sequential(
            nn.LayerNorm(normalized_shape=[channels]),
            nn.Linear(in_features=channels, out_features=channels),
            nn.GELU(),
            nn.Linear(in_features=channels, out_features=channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.channels, self.size**2).swapaxes(1, 2)

        x_ln = self.ln(x)

        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff(attention_value) + attention_value

        return attention_value.swapaxes(2, 1).view(
            -1, self.channels, self.size, self.size
        )


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
                in_channels=in_channels, out_channels=in_channels, residual=True
            ),
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                mid_channels=in_channels // 2,
            ),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(), nn.Linear(in_features=emb_dim, out_features=out_channels)
        )

    def forward(
        self, x: torch.Tensor, skip_x: torch.Tensor, t: torch.Tensor
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
                in_channels=in_channels, out_channels=in_channels, residual=True
            ),
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
            ),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(), nn.Linear(in_features=emb_dim, out_features=out_channels)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + emb
