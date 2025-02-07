import torch
from torch import nn

from .ddpm import Diffusion_DDPM


class Diffusion_CFG(Diffusion_DDPM):
    def __init__(
        self,
        noise_predictor: nn.Module,
        T: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        img_size: int = 64,
        in_channels: int = 3,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            noise_predictor,
            T,
            beta_start,
            beta_end,
            img_size,
            in_channels,
            device,
        )

    def predict_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        if cfg_scale == 0:
            return super().predict_noise(x_t, t)

        if cfg_scale == 1:
            return self.noise_predictor(x_t, t, labels)

        unconditional_noise = self.noise_predictor(x_t, t, None)
        conditional_noise = self.noise_predictor(x_t, t, labels)

        return torch.lerp(unconditional_noise, conditional_noise, cfg_scale)

    @torch.no_grad()
    def sample(
        self,
        n: int,
        labels: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        x_t = torch.randn(
            (n, self.in_channels, self.img_size, self.img_size),
            device=self.device,
        )

        for i in reversed(range(1, self.T + 1)):
            t = (torch.ones(n, device=x_t.device) * i).long()

            z = torch.randn_like(x_t) if i > 1 else torch.zeros_like(x_t)

            sigma_t = torch.sqrt(self.sigma_theta(t - 1))

            eps_theta = self.predict_noise(x_t, t - 1, labels, cfg_scale)

            x_t = self.mu_theta(x_t, t - 1, eps_theta) + sigma_t * z

        return x_t

    def calc_loss(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        mse = nn.MSELoss()

        x_t, noise = self.forward(x_0, t)
        noise_pred = self.predict_noise(x_t, t, labels, cfg_scale)

        loss = mse(noise, noise_pred)

        return loss
