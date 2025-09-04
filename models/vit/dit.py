from typing import Optional
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

from models.common.mixin import ModelMixin
from models.common.cfg import HasCFGBackBone
from . import modules as m


class DiT(nn.Module, ModelMixin, HasCFGBackBone):
    def __init__(
        self,
        in_channels: int = 4,
        input_size: int = 32,
        patch_size: int = 2,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.learn_sigma = learn_sigma

        self.x_embed = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            bias=True,
        )
        self.t_embed = m.TimeStepEmbedding(hidden_size=hidden_size)

        num_patches = self.x_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size),
            requires_grad=False,
        )

        self.blocks = nn.ModuleList(
            [
                m.DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = m.FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=self.out_channels,
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = m.PositionalEmbedding.get_2d_sin_cos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embed.num_patches**0.5),
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embed.proj.bias, 0)

        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor):
        c = self.out_channels
        p = self.x_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))

        return imgs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ):
        if t.dim() == 0:
            t = t.expand(len(x))

        x = self.x_embed(x) + self.pos_embed
        t = self.t_embed(t)

        if y is not None:
            t = t + y

        for block in self.blocks:
            x = block(x, t)

        x = self.final_layer(x, t)
        x = self.unpatchify(x)

        return x
