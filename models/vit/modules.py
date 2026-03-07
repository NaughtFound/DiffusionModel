import math
from collections import OrderedDict

import numpy as np
import torch
from timm.layers.attention import Attention
from timm.layers.mlp import Mlp
from torch import nn


class TanhGELU(nn.GELU):
    def __init__(self) -> None:
        super().__init__(approximate="tanh")


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class TimeStepEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=frequency_embedding_size,
                out_features=hidden_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=hidden_size,
                out_features=hidden_size,
                bias=True,
            ),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def time_step_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: int = 10000,
    ) -> torch.Tensor:
        half = dim // 2
        freq = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freq[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.time_step_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedding(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()

        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_embeddings=num_classes + use_cfg_embedding,
            embedding_dim=hidden_size,
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(
        self,
        labels: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(
        self,
        labels: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (self.training and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        **block_kwargs,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(
            normalized_shape=hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )

        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs,
        )

        self.norm2 = nn.LayerNorm(
            normalized_shape=hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=TanhGELU,
            drop=0,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                in_features=hidden_size,
                out_features=6 * hidden_size,
                bias=True,
            ),
        )

    def _modulate(
        self,
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        x = x + gate_msa.unsqueeze(1) * self.attn(
            self._modulate(self.norm1(x), shift_msa, scale_msa)
        )
        return x + gate_mlp.unsqueeze(1) * self.mlp(
            self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        )


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.norm_final = nn.LayerNorm(
            normalized_shape=hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=patch_size * patch_size * out_channels,
            bias=True,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                in_features=hidden_size,
                out_features=2 * hidden_size,
                bias=True,
            ),
        )

    def _modulate(
        self,
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)

        x = self._modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class PositionalEmbedding:
    @staticmethod
    def get_1d_sin_cos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
        assert embed_dim % 2 == 0

        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega

        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)

        return np.concatenate([emb_sin, emb_cos], axis=1)

    @staticmethod
    def get_2d_sin_cos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
        assert embed_dim % 2 == 0

        emb_h = PositionalEmbedding.get_1d_sin_cos_pos_embed_from_grid(
            embed_dim // 2,
            grid[0],
        )
        emb_w = PositionalEmbedding.get_1d_sin_cos_pos_embed_from_grid(
            embed_dim // 2,
            grid[1],
        )

        return np.concatenate([emb_h, emb_w], axis=1)

    @staticmethod
    def get_2d_sin_cos_pos_embed(
        embed_dim: int,
        grid_size: int,
        extra_tokens: int = 0,
        *,
        cls_token: bool = False,
    ) -> np.ndarray:
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = PositionalEmbedding.get_2d_sin_cos_pos_embed_from_grid(
            embed_dim,
            grid,
        )

        if cls_token and extra_tokens > 0:
            pos_embed = np.concatenate(
                [np.zeros([extra_tokens, embed_dim]), pos_embed],
                axis=0,
            )

        return pos_embed


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.res_block = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_block(x)
