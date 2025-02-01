import torch
import torch.nn as nn
import torchsde
from models.diffusion.base import Diffusion


class SDE_DDPM_Params:
    eps_theta: nn.Module
    beta_start: float
    beta_end: float
    input_size: tuple[int, int, int]

    def __init__(self, device: torch.device):
        self.device = device

        self.eps_theta = None
        self.beta_start = 1e-4
        self.beta_end = 0.02

        self.input_size = (1, 28, 28)


class SDE_DDPM_Forward(nn.Module):
    def __init__(self, args: SDE_DDPM_Params):
        super().__init__()

        self.args = args

    def _beta(self, t: torch.Tensor):
        b_s = self.args.beta_start
        b_e = self.args.beta_end

        return b_s + t * (b_e - b_s)

    def _alpha_hat(self, t: torch.Tensor) -> torch.Tensor:
        alpha = 1 - self._beta(t)
        return torch.cumprod(alpha, dim=0)

    def s_theta(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.args.eps_theta(x=x, t=t)

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * self._beta(t) * x

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self._beta(t))

    def forward(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.sqrt(self._alpha_hat(t))[:, None, None, None]
        b = torch.sqrt(1.0 - self._alpha_hat(t))[:, None, None, None]

        eps = torch.rand_like(x_0)

        return a * x_0 + b * eps, eps


class SDE_DDPM_Reverse(nn.Module):
    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(
        self,
        forward_sde: SDE_DDPM_Forward,
        args: SDE_DDPM_Params,
    ):
        super().__init__()

        self.forward_sde = forward_sde
        self.args = args

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, *self.args.input_size)

        f1 = self.forward_sde.f(-t, x)
        f2 = self.forward_sde.g(-t, x) ** 2 * self.forward_sde.s_theta(-t, x)

        f = -(f1 - f2)

        return f.flatten(1)

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, *self.args.input_size)

        g = self.forward_sde.g(-t, x)
        return -g.flatten(1)

    @torch.no_grad()
    def forward(self, n: int, dt: float = 1e-2) -> torch.Tensor:
        t = torch.tensor([-1, 0], device=self.args.device)
        x_t = torch.randn(size=(n, *self.args.input_size), device=self.args.device)

        x_s = torchsde.sdeint(self, x_t.flatten(1), t, dt=dt).view(len(t), *x_t.size())

        return x_s


class SDE_DDPM(Diffusion):
    def __init__(self, args: SDE_DDPM_Params):
        super().__init__()

        self.args = args

        self.f_sde = SDE_DDPM_Forward(args)
        self.r_sde = SDE_DDPM_Reverse(self.f_sde, args)

    def t(self, n: int):
        return torch.rand((n,), device=self.args.device)

    def forward(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ):
        return self.f_sde.forward(x_0, t)

    def sample(self, n: int):
        return self.r_sde.forward(n)[-1]

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor):
        return self.f_sde.s_theta(t, x_t)
