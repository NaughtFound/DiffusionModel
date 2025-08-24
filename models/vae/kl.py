from typing import Literal, Optional
import torch
from diffusers.models import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from .base import VAE
from .modules import LPIPSWithDiscriminator


class VAE_KL_Params:
    in_channels: int
    out_channels: int
    latent_channels: int
    pretrained_model_name_or_path: Optional[str]
    lpips_model_path: Optional[str]
    disc_start: int
    log_var_init: float
    kl_weight: float
    pixel_loss_weight: float
    disc_n_layers: int
    disc_in_channels: int
    disc_factor: float
    disc_weight: float
    perceptual_weight: float
    disc_conditional: bool
    disc_loss: Literal["hinge", "vanilla"]

    def __init__(self, device: torch.device):
        self.device = device

        self.in_channels = 3
        self.out_channels = 3
        self.latent_channels = 4
        self.pretrained_model_name_or_path = None
        self.lpips_model_path = None
        self.disc_start = 50001
        self.log_var_init = 0.0
        self.kl_weight = 1.0
        self.pixel_loss_weight = 1.0
        self.disc_n_layers = 3
        self.disc_in_channels = 3
        self.disc_factor = 1.0
        self.disc_weight = 1.0
        self.perceptual_weight = 1.0
        self.disc_conditional = False
        self.disc_loss = "hinge"


class VAE_KL(VAE):
    def __init__(self, args: VAE_KL_Params):
        super().__init__()

        self.args = args

        if args.pretrained_model_name_or_path is not None:
            self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path)
        else:
            self.vae = AutoencoderKL(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                latent_channels=args.latent_channels,
            )

        self.loss = LPIPSWithDiscriminator(
            lpips_model_path=args.lpips_model_path,
            disc_start=args.disc_start,
            log_var_init=args.log_var_init,
            kl_weight=args.kl_weight,
            pixel_loss_weight=args.pixel_loss_weight,
            disc_n_layers=args.disc_n_layers,
            disc_in_channels=args.disc_in_channels,
            disc_factor=args.disc_factor,
            disc_weight=args.disc_weight,
            perceptual_weight=args.perceptual_weight,
            disc_conditional=args.disc_conditional,
            disc_loss=args.disc_loss,
        )

    def encode(self, x: torch.Tensor, return_dist: bool = False):
        enc_dist = self.vae.encode(x, return_dict=False)[0]

        if not isinstance(enc_dist, DiagonalGaussianDistribution):
            raise TypeError("encoded x is not DiagonalGaussianDistribution")

        if return_dist:
            return enc_dist

        return enc_dist.sample()

    def decode(self, z: torch.Tensor):
        dec_tensor = self.vae.decode(z, return_dict=False)[0]

        if not isinstance(dec_tensor, torch.Tensor):
            raise TypeError("decoded z is not Tensor")

        return dec_tensor

    def sample(self, n: int):
        raise NotImplementedError("sampling is not implemented for kl.")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)

        return x_hat

    def calc_loss(self, x: torch.Tensor, optimizer_idx: int, global_step: int):
        last_layer = self.vae.decoder.conv_out.weight

        z_dist = self.encode(x, return_dist=True)
        x_hat = self.decode(z_dist.sample())

        loss = self.loss(
            inputs=x,
            reconstructions=x_hat,
            posteriors=z_dist,
            optimizer_idx=optimizer_idx,
            global_step=global_step,
            last_layer=last_layer,
        )

        return loss
