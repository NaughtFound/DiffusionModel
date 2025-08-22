from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


class VectorQuantizer(nn.Module):
    def __init__(self, n_emb: int, emb_dim: int, beta: float):
        super().__init__()
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_emb, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_emb, 1.0 / self.n_emb)

    def _calc_encodings(self, z: torch.Tensor):
        z_flattened = z.view(-1, self.emb_dim)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_emb).to(
            z_flattened.device
        )
        min_encodings.scatter_(1, min_encoding_indices, 1)

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return min_encoding_indices, min_encodings, perplexity

    def _calc_loss(self, z: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        q_latent_loss = torch.mean((z_q - z.detach()) ** 2)
        e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
        loss = q_latent_loss + self.beta * e_latent_loss

        return loss

    def forward(self, z: torch.Tensor):
        z = z.permute(0, 2, 3, 1).contiguous()

        min_encoding_indices, min_encodings, perplexity = self._calc_encodings(z)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        z_q = z + (z_q - z).detach()

        loss = self._calc_loss(z, z_q)

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return {
            "loss": loss,
            "z_q": z_q,
            "perplexity": perplexity,
            "encodings": min_encodings,
            "encodings_idx": min_encoding_indices,
        }


class ResidualLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res_h_dim: int,
    ):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=res_h_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=res_h_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

        self.skip_proj = None

        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip_proj is not None:
            x = self.skip_proj(x)

        x = x + self.res_block(x)

        return x


class ResidualStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res_h_dim: int,
        n_res_layers: int,
        hidden_dim: int = None,
    ):
        super().__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.Sequential()

        if hidden_dim is None:
            hidden_dim = out_channels

        for _ in range(n_res_layers - 1):
            self.stack.append(
                ResidualLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    res_h_dim=res_h_dim,
                )
            )
            in_channels = hidden_dim

        self.stack.append(
            ResidualLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                res_h_dim=res_h_dim,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stack(x)
        x = F.relu(x)

        return x


class PatchGANDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_filters: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()

        kernel_size = 4
        padding = 1

        sequence = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=n_filters,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mul = 1
        nf_mul_prev = 1
        for n in range(1, n_layers):
            nf_mul_prev = nf_mul
            nf_mul = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    n_filters * nf_mul_prev,
                    n_filters * nf_mul,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=n_filters * nf_mul),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mul_prev = nf_mul
        nf_mul = min(2**n_layers, 8)

        sequence += [
            nn.Conv2d(
                in_channels=n_filters * nf_mul_prev,
                out_channels=n_filters * nf_mul,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=n_filters * nf_mul),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(
                in_channels=n_filters * nf_mul,
                out_channels=1,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
        ]

        self.main = nn.Sequential(*sequence)

    def forward(self, input: torch.Tensor):
        return self.main(input)

    def init_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        lpips_model_path: str,
        disc_start: int,
        log_var_init: float = 0.0,
        kl_weight: float = 1.0,
        pixel_loss_weight: float = 1.0,
        disc_n_layers: int = 3,
        disc_in_channels: int = 3,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_conditional: bool = False,
        disc_loss: Literal["hinge", "vanilla"] = "hinge",
    ):
        super().__init__()

        self.kl_weight = kl_weight
        self.pixel_weight = pixel_loss_weight
        self.perceptual_loss = LPIPS(model_path=lpips_model_path).eval()
        self.perceptual_weight = perceptual_weight

        self.log_var = nn.Parameter(torch.ones(size=()) * log_var_init)

        self.discriminator = PatchGANDiscriminator(
            in_channels=disc_in_channels,
            n_layers=disc_n_layers,
        )

        self.discriminator.init_weights()

        self.discriminator_iter_start = disc_start
        self.disc_loss = disc_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def _calculate_adaptive_weight(
        self,
        nll_loss: torch.Tensor,
        g_loss: torch.Tensor,
        last_layer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if last_layer is not None:
            nll_grads = torch.autograd.grad(
                nll_loss,
                last_layer,
                retain_graph=True,
            )[0]
            g_grads = torch.autograd.grad(
                g_loss,
                last_layer,
                retain_graph=True,
            )[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss,
                self.last_layer[0],
                retain_graph=True,
            )[0]
            g_grads = torch.autograd.grad(
                g_loss,
                self.last_layer[0],
                retain_graph=True,
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight

        return d_weight

    def _calc_loss(
        self,
        logits_real: torch.Tensor,
        logits_fake: torch.Tensor,
    ) -> torch.Tensor:
        if self.disc_loss == "vanilla":
            d_loss = 0.5 * (
                torch.mean(torch.nn.functional.softplus(-logits_real))
                + torch.mean(torch.nn.functional.softplus(logits_fake))
            )

            return d_loss

        if self.disc_loss == "hinge":
            loss_real = torch.mean(F.relu(1.0 - logits_real))
            loss_fake = torch.mean(F.relu(1.0 + logits_fake))
            d_loss = 0.5 * (loss_real + loss_fake)

            return d_loss

    def _adopt_weight(
        self,
        weight: torch.Tensor,
        global_step: int,
        threshold: int = 0,
        value: float = 0.0,
    ):
        if global_step < threshold:
            weight = value
        return weight

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        posteriors: DiagonalGaussianDistribution,
        optimizer_idx: int,
        global_step: int,
        last_layer: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(),
                reconstructions.contiguous(),
            )

            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.log_var) + self.log_var

        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss

        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        if optimizer_idx == 0:
            if cond is None:
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                logits_fake = self.discriminator(
                    torch.cat(
                        (
                            reconstructions.contiguous(),
                            cond,
                        ),
                        dim=1,
                    )
                )
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self._calculate_adaptive_weight(
                        nll_loss,
                        g_loss,
                        last_layer=last_layer,
                    )
                except RuntimeError:
                    if self.training:
                        raise ValueError("training is True")

                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = self._adopt_weight(
                self.disc_factor,
                global_step,
                threshold=self.discriminator_iter_start,
            )

            loss = (
                weighted_nll_loss
                + self.kl_weight * kl_loss
                + d_weight * disc_factor * g_loss
            )

            return loss

        if optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat(
                        (
                            inputs.contiguous().detach(),
                            cond,
                        ),
                        dim=1,
                    )
                )
                logits_fake = self.discriminator(
                    torch.cat(
                        (
                            reconstructions.contiguous().detach(),
                            cond,
                        ),
                        dim=1,
                    )
                )

            disc_factor = self._adopt_weight(
                self.disc_factor,
                global_step,
                threshold=self.discriminator_iter_start,
            )

            d_loss = disc_factor * self._calc_loss(
                logits_real=logits_real,
                logits_fake=logits_fake,
            )

            return d_loss
