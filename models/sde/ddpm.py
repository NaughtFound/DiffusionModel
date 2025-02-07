import torch
import torch.nn as nn
import torchsde
from models.diffusion.base import Diffusion
from utils import fill_tail_dims


class SDE_DDPM_Params:
    eps_theta: nn.Module
    beta_start: float
    beta_end: float
    t0: float
    t1: float
    input_size: tuple[int, int, int]

    def __init__(self, device: torch.device):
        self.device = device

        self.eps_theta = None
        self.beta_start = 1e-4
        self.beta_end = 0.02

        self.t0 = 0
        self.t1 = 1

        self.input_size = (1, 28, 28)


class SDE_DDPM_Forward(nn.Module):
    def __init__(self, args: SDE_DDPM_Params):
        super().__init__()

        self.args = args

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        b_s = self.args.beta_start
        b_e = self.args.beta_end

        return b_s + t * (b_e - b_s)

    def _indefinite_int(self, t: torch.Tensor) -> torch.Tensor:
        b_s = self.args.beta_start
        b_e = self.args.beta_end

        return b_s * t + 0.5 * t**2 * (b_e - b_s)

    def analytical_mean(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        int_t = self._indefinite_int(t)
        int_t_0 = self._indefinite_int(self.args.t0)

        mean_coef = (-0.5 * (int_t - int_t_0)).exp()
        mean = x_0 * fill_tail_dims(mean_coef, x_0)

        return mean

    def analytical_var(self, t: torch.Tensor) -> torch.Tensor:
        int_t = self._indefinite_int(t)
        int_t_0 = self._indefinite_int(self.args.t0)

        var = 1 - (-int_t + int_t_0).exp()

        return var

    @torch.no_grad()
    def analytical_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean = self.analytical_mean(x_0, t)
        var = self.analytical_var(t)
        return mean + torch.randn_like(mean) * fill_tail_dims(var.sqrt(), mean)

    @torch.no_grad()
    def analytical_score(
        self,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        mean = self.analytical_mean(x_0, t)
        var = self.analytical_var(t)
        return -(x_t - mean) / fill_tail_dims(var, mean).clamp_min(1e-5)

    def s_theta(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.args.eps_theta(x=x, t=t)

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * self._beta(t) * x

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return fill_tail_dims(self._beta(t).sqrt(), x).expand_as(x)


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
        t = torch.tensor([-self.args.t1, -self.args.t0], device=self.args.device)
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
        r = torch.rand((n,), device=self.args.device)
        return self.args.t0 + (self.args.t1 - self.args.t0) * r

    def forward(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ):
        return self.f_sde.analytical_sample(x_0, t)

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
        x_t = self.f_sde.analytical_sample(x_0, t)
        lambda_t = self.f_sde.analytical_var(t)

        score_pred = self.f_sde.s_theta(t, x_t)
        score_true = self.f_sde.analytical_score(x_t, x_0, t)

        loss = fill_tail_dims(lambda_t, x_t) * ((score_pred - score_true) ** 2)

        return loss.mean()
