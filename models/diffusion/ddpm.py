import torch
from torch import nn
from .base import Diffusion


class Diffusion_DDPM(Diffusion):
    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        img_size: int = 64,
        in_channels: int = 3,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.in_channels = in_channels

        self.device = device

        self.beta = self.beta_scheduler()
        self.alpha = 1 - self.beta

        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def beta_scheduler(self):
        return torch.linspace(
            self.beta_start,
            self.beta_end,
            self.T,
            device=self.device,
        )

    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        b = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]

        eps = torch.rand_like(x_0)

        return a * x_0 + b * eps, eps

    def t(self, n: int):
        return torch.randint(1, self.T, (n,), device=self.device)

    def predict_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        return model.forward(x_t, t)

    def mu_theta(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps_theta: torch.Tensor,
    ) -> torch.Tensor:
        alpha_t = self.alpha[t][:, None, None, None]
        alpha_hat_t = self.alpha_hat[t][:, None, None, None]
        beta_t = self.beta[t][:, None, None, None]

        return (
            1
            / torch.sqrt(alpha_t)
            * (x_t - beta_t / torch.sqrt(1 - alpha_hat_t) * eps_theta)
        )

    def sigma_theta(self, t: torch.Tensor) -> torch.Tensor:
        # alpha_t = self.alpha[t][:, None, None, None]
        # alpha_hat_t = self.alpha_hat[t][:, None, None, None]

        beta_t = self.beta[t][:, None, None, None]

        # return (1-alpha_hat_t)/(1-alpha_t) * beta_t

        return beta_t

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int) -> torch.Tensor:
        model.eval()
        x_t = torch.randn(
            (n, self.in_channels, self.img_size, self.img_size), device=self.device
        )

        for i in reversed(range(1, self.T + 1)):
            t = (torch.ones(n, device=x_t.device) * i).long()

            z = torch.randn_like(x_t) if i > 1 else torch.zeros_like(x_t)

            sigma_t = torch.sqrt(self.sigma_theta(t - 1))

            eps_theta = self.predict_noise(x_t, t - 1, model)

            x_t = self.mu_theta(x_t, t - 1, eps_theta) + sigma_t * z

        x_0 = (x_t.clamp(-1, 1) + 1) / 2
        x_0 = (x_0 * 255).to(torch.uint8)

        return x_0
