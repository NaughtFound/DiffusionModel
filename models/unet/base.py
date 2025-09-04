import torch
from torch import nn

from models.common.mixin import ModelMixin
from . import modules as m


class UNet(nn.Module, ModelMixin):
    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 64,
        out_channels: int = 3,
        emb_dim: int = 256,
        features: list[int] = [128, 256],
        neck_features: list[int] = [512],
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.features = features
        self.neck_features = neck_features
        self.emb_dim = emb_dim

        self.inc = m.DoubleConv(in_channels, mid_channels)
        self.out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

        self.down = nn.ModuleList()
        self.down_attention = nn.ModuleList()
        self.bottle_neck = nn.Sequential()
        self.up = nn.ModuleList()
        self.up_attention = nn.ModuleList()

        self._build_down()
        self._build_up()
        self._build_attention()
        self._build_neck()

    def _build_down(self):
        f_before = self.mid_channels
        for f in self.features:
            self.down.append(m.Down(f_before, f, self.emb_dim))
            f_before = f

        self.down.append(m.Down(self.features[-1], self.features[-1], self.emb_dim))

    def _build_up(self):
        for f in reversed(self.features):
            self.up.append(m.Up(f * 2, f // 2, self.emb_dim))

        self.up.append(m.Up(f, self.mid_channels))

    def _build_attention(self):
        for f in self.features:
            self.down_attention.append(m.SelfAttention(f))

        self.down_attention.append(m.SelfAttention(self.features[-1]))

        for f in reversed(self.features):
            self.up_attention.append(m.SelfAttention(f // 2))

        self.up_attention.append(m.SelfAttention(self.mid_channels))

    def _build_neck(self):
        n_before = self.features[-1]
        for n in self.neck_features:
            self.bottle_neck.append(m.DoubleConv(n_before, n))
            n_before = n

        self.bottle_neck.append(m.DoubleConv(n_before, self.features[-1]))

    def pos_encoding(self, t: torch.Tensor, channels: int) -> torch.Tensor:
        inv_freq = 1.0 / (
            1e4 ** (torch.arange(0, channels // 2, device=t.device).float() / channels)
        )

        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)

        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def _time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1).to(torch.float)
        t = self.pos_encoding(t, self.emb_dim)

        return t

    def _encode(self, x: torch.Tensor, t: torch.Tensor) -> list[torch.Tensor]:
        x = self.inc(x)
        x_l = [x]

        for i in range(len(self.down)):
            x_i = self.down[i](x_l[-1], t)
            x_i = self.down_attention[i](x_i)
            x_l.append(x_i)

        x_l[-1] = self.bottle_neck(x_l[-1])

        return x_l

    def _decode(self, x_l: list[torch.Tensor], t: torch.Tensor) -> torch.Tensor:
        x = x_l[-1]
        x_l = x_l[::-1]

        for i in range(len(self.up)):
            x = self.up[i](x, x_l[i + 1], t)
            x = self.up_attention[i](x)

        return self.out(x)

    def _encode_decode(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_l = self._encode(x, t)
        x = self._decode(x_l, t)

        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        only_encode: bool = False,
    ) -> torch.Tensor:
        t = self._time_encoding(t)

        if only_encode:
            return self._encode(x, t)

        return self._encode_decode(x, t)
