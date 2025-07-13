import torch
from torch import nn

from .base import UNet


class LabelConditionedUNet(UNet):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        mid_channels: int = 64,
        out_channels: int = 3,
        emb_dim: int = 256,
        features: list[int] = [128, 256],
        neck_features: list[int] = [512],
    ) -> None:
        super().__init__(
            in_channels,
            mid_channels,
            out_channels,
            emb_dim,
            features,
            neck_features,
        )

        self.label_emb = nn.Embedding(num_classes, emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor = None,
        only_encode: bool = False,
    ) -> torch.Tensor:
        t = self._time_encoding(t)

        if labels is not None:
            t_batch = t.expand(len(x), -1)
            t = t_batch + self.label_emb(labels)

        if only_encode:
            return self._encode(x, t)

        return self._encode_decode(x, t)
