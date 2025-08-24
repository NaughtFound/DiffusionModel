import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp


class TanhGELU(nn.GELU):
    def __init__(self):
        super().__init__(approximate="tanh")


class TimeStepEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
    ):
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
    def time_step_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freq = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freq[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor):
        t_freq = self.time_step_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)

        return t_emb


class LabelEmbedding(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        dropout_prob: float,
    ):
        super().__init__()

        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_embeddings=num_classes + use_cfg_embedding,
            embedding_dim=hidden_size,
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: torch.Tensor = None):
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)

        return labels

    def forward(self, labels: torch.Tensor, force_drop_ids: torch.Tensor = None):
        use_dropout = self.dropout_prob > 0
        if (self.training and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)

        return embeddings


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        **block_kwargs,
    ):
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
    ):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        x = x + gate_msa.unsqueeze(1) * self.attn(
            self._modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
    ):
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
    ):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)

        x = self._modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x


class PositionalEmbedding:
    @staticmethod
    def get_1d_sin_cos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray):
        assert embed_dim % 2 == 0

        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega

        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)

        return emb

    @staticmethod
    def get_2d_sin_cos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray):
        assert embed_dim % 2 == 0

        emb_h = PositionalEmbedding.get_1d_sin_cos_pos_embed_from_grid(
            embed_dim // 2,
            grid[0],
        )
        emb_w = PositionalEmbedding.get_1d_sin_cos_pos_embed_from_grid(
            embed_dim // 2,
            grid[1],
        )

        emb = np.concatenate([emb_h, emb_w], axis=1)

        return emb

    @staticmethod
    def get_2d_sin_cos_pos_embed(
        embed_dim: int,
        grid_size: int,
        cls_token: bool = False,
        extra_tokens: int = 0,
    ):
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
