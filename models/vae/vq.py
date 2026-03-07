import torch
from torch import nn

from models.common.params import ModelParams

from . import modules as m
from .base import VAE


class VAEVQParams(ModelParams):
    in_channels: int
    latent_channels: int
    hidden_dim: int
    n_embeddings: int
    res_h_dim: int
    n_res_layers: int
    beta: float

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)

        self.in_channels = 3
        self.latent_channels = 64
        self.hidden_dim = 128
        self.n_embeddings = 512
        self.res_h_dim = 32
        self.n_res_layers = 2
        self.beta = 0.25


class VAEVQEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        n_res_layers: int,
        res_h_dim: int,
    ) -> None:
        super().__init__()

        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels // 2,
                kernel_size=kernel,
                stride=stride,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels // 2,
                out_channels=hidden_channels,
                kernel_size=kernel,
                stride=stride,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel - 1,
                stride=stride - 1,
                padding=1,
            ),
            m.ResidualStack(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                res_h_dim=res_h_dim,
                n_res_layers=n_res_layers,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x)


class VAEVQDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_res_layers: int,
        res_h_dim: int,
    ) -> None:
        super().__init__()

        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel - 1,
                stride=stride - 1,
                padding=1,
            ),
            m.ResidualStack(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                res_h_dim=res_h_dim,
                n_res_layers=n_res_layers,
            ),
            nn.ConvTranspose2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels // 2,
                kernel_size=kernel,
                stride=stride,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=hidden_channels // 2,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inverse_conv_stack(x)


class VAEVQ(VAE):
    def __init__(self, args: VAEVQParams) -> None:
        super().__init__()

        self.args = args

        self.encoder = VAEVQEncoder(
            in_channels=self.args.in_channels,
            hidden_channels=self.args.hidden_dim,
            n_res_layers=self.args.n_res_layers,
            res_h_dim=self.args.res_h_dim,
        )
        self.pre_quantizer = nn.Conv2d(
            in_channels=self.args.hidden_dim,
            out_channels=self.args.latent_channels,
            kernel_size=1,
            stride=1,
        )
        self.quantizer = m.VectorQuantizer(
            n_emb=self.args.n_embeddings,
            emb_dim=self.args.latent_channels,
            beta=self.args.beta,
        )
        self.decoder = VAEVQDecoder(
            in_channels=self.args.latent_channels,
            hidden_channels=self.args.hidden_dim,
            out_channels=self.args.in_channels,
            n_res_layers=self.args.n_res_layers,
            res_h_dim=self.args.res_h_dim,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(x)
        return self.pre_quantizer(z_e)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def sample(self, n: int) -> torch.Tensor:
        msg = "sampling is not implemented for vq."
        raise NotImplementedError(msg)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encode(x)
        vq_dict = self.quantizer(z_e)

        z_q = vq_dict.get("z_q", torch.zeros_like(z_e))

        return self.decode(z_q)

    def calc_loss(self, x: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        z_e = self.encode(x)
        vq_dict = self.quantizer(z_e)

        z_q = vq_dict.get("z_q", torch.zeros_like(z_e))

        x_hat = self.decode(z_q)

        recon_loss = torch.mean((x_hat - x) ** 2 / var)
        embed_loss = vq_dict.get("loss", torch.zeros_like(recon_loss))

        return recon_loss + embed_loss
