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

    def _indefinite_int(self, t: torch.Tensor) -> torch.Tensor:
        b_s = self.args.beta_start
        b_e = self.args.beta_end

        return b_s * t + 0.5 * t**2 * (b_e - b_s)

    def analytical_mean(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean_coef = (-0.5 * (self._indefinite_int(t) - self._indefinite_int(0))).exp()
        mean = x_0 * mean_coef
        return mean

    def analytical_var(self, t: torch.Tensor) -> torch.Tensor:
        analytical_var = 1 - (-self._indefinite_int(t) + self._indefinite_int(0)).exp()
        return analytical_var

    @torch.no_grad()
    def analytical_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean = self.analytical_mean(x_0, t)
        var = self.analytical_var(x_0, t)
        return mean + torch.randn_like(mean) * var.sqrt()

    @torch.no_grad()
    def analytical_score(
        self,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        mean = self.analytical_mean(x_0, t)
        var = self.analytical_var(x_0, t)
        return -(x_t - mean) / var.clamp_min(1e-5)

    def s_theta(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.args.eps_theta(x=x, t=t)

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * self._beta(t) * x

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self._beta(t)).expand_as(x)

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

        x_0 = (x_s.clamp(-1, 1) + 1) / 2
        x_0 = (x_0 * 255).to(torch.uint8)

        return x_0


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

    def eval(self):
        self.f_sde.eval()
        self.args.eps_theta.eval()

    def train(self):
        self.f_sde.train()
        self.args.eps_theta.train()

    def calc_loss(self, x_0: torch.Tensor, t: torch.Tensor):
        score_pred = self.forward(x_0, t)

        x_t = self.f_sde.analytical_sample(x_0, t)
        lambda_t = self.f_sde.analytical_var(t)
        score_true = self.f_sde.analytical_score(x_t, x_0, t)

        loss = lambda_t * ((score_pred - score_true) ** 2)

        return loss.flatten(1).sum(dim=1)
