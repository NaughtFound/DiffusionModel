import torch
from torch import nn


class SimpleDiffusion(nn.Module):
    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        img_size: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()

        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size

        self.beta = self.beta_scheduler()
        self.alpha = 1 - self.beta

        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def beta_scheduler(self):
        return torch.linspace(self.beta_start, self.beta_end, self.T)

    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        b = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]

        eps = torch.rand_like(x_0)

        return a * x_0 + b*eps, eps

    def t(self, n: int):
        return torch.randint(1, self.T, (n,))

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int):
        model.eval()
        x_t = torch.randn((n, 3, *self.img_size), device=self.device)

        for i in range(self.T, 1, -1):
            t = (torch.ones(n, device=x_t.device)*i).long()

            alpha_t = self.alpha[t][:, None, None, None]
            alpha_hat_t = self.alpha_hat[t][:, None, None, None]

            beta_t = self.beta[t][:, None, None, None]

            eps_theta = model.forward(x_t, t)

            z = torch.randn_like(x_t) if i > 1 else torch.zeros_like(x_t)

            sigma_t = torch.sqrt(beta_t)

            x_t = 1/torch.sqrt(alpha_t) * (x_t - beta_t /
                                           torch.sqrt(1-alpha_hat_t) * eps_theta) + sigma_t*z

        x_0 = (x_t.clamp(-1, 1) + 1)/2
        x_0 = (x_0*255).to(torch.uint8)

        return x_0
