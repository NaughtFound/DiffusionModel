from typing import Optional
import torch
from torch import nn

from models.common.cfg import HasCFGBackBone
from .base import UNet


class LabelConditionedUNet(UNet, HasCFGBackBone):
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
        y: Optional[torch.Tensor] = None,
        only_encode: bool = False,
        embed_y: bool = True,
    ) -> torch.Tensor:
        t = self._time_encoding(t)

        t = t.expand(len(x), -1)

        if embed_y and y is not None:
            y = self.label_emb(y)

        if y is not None:
            t = t + y

        if only_encode:
            return self._encode(x, t)

        return self._encode_decode(x, t)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
        y_null: Optional[torch.Tensor] = None,
        only_encode: bool = False,
        fast_cfg: bool = True,
    ):
        y_emb = self.label_emb(y)

        return super().forward_with_cfg(
            x,
            t,
            y_emb,
            cfg_scale,
            y_null,
            fast_cfg=fast_cfg,
            only_encode=only_encode,
            embed_y=False,
        )
