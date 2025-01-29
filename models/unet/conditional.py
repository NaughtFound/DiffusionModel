import torch
from torch import nn

from .base import UNet


class ConditionalUNet(UNet):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        mid_channels: int = 64,
        out_channels: int = 3,
        time_dim: int = 256,
        emb_dim: int = 256,
        features: list[int] = [128, 256],
        neck_features: list[int] = [512],
    ) -> None:
        super().__init__(
            in_channels,
            mid_channels,
            out_channels,
            time_dim,
            emb_dim,
            features,
            neck_features,
        )

        self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor = None,
    ) -> torch.Tensor:
        t = self.time_encoding(t)

        if y is not None:
            t += self.label_emb(y)

        return self.encode_decode(x, t)
