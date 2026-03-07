import torch

from models.common.cfg import HasCFGBackBone

from . import modules as m
from .base import UNet


class ConditionalUNet(UNet, HasCFGBackBone):
    def _build_attention(self) -> None:
        for f in self.features:
            self.down_attention.append(m.CrossAttention(f, self.emb_dim))

        self.down_attention.append(m.CrossAttention(self.features[-1], self.emb_dim))

        for f in reversed(self.features):
            self.up_attention.append(m.CrossAttention(f // 2, self.emb_dim))

        self.up_attention.append(m.CrossAttention(self.mid_channels, self.emb_dim))

    def _encode(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> list[torch.Tensor]:
        x = self.inc(x)
        x_l = [x]

        for i in range(len(self.down)):
            x_i = self.down[i](x_l[-1], t)
            x_i = self.down_attention[i](x_i, y)
            x_l.append(x_i)

        x_l[-1] = self.bottle_neck(x_l[-1])

        return x_l

    def _decode(
        self,
        x_l: list[torch.Tensor],
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x = x_l[-1]
        x_l.reverse()

        for i in range(len(self.up)):
            x = self.up[i](x, x_l[i + 1], t)
            x = self.up_attention[i](x, y)

        return self.out(x)

    def _encode_decode(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x_l = self._encode(x, t, y)
        return self._decode(x_l, t, y)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(len(x))

        t = self._time_encoding(t)

        if y is None:
            y = torch.zeros((len(x), self.emb_dim), device=x.device)

        return self._encode_decode(x, t, y)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
        y_null: torch.Tensor | None = None,
        *,
        fast_cfg: bool = True,
    ) -> torch.Tensor:
        return super().forward_with_cfg(
            x,
            t,
            y,
            cfg_scale,
            y_null,
            fast_cfg=fast_cfg,
        )
