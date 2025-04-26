import torch
import torch.nn as nn
import torchsde
import torchdiffeq
from models.diffusion.base import Diffusion
from utils import fill_tail_dims


class DDPM_Params:
    eps_theta: nn.Module
    beta_min: float
    beta_max: float
    t0: float
    t1: float
    input_size: tuple[int, int, int]

    def __init__(self, device: torch.device):
        self.device = device

        self.eps_theta = None
        self.beta_min = 1e-4
        self.beta_max = 0.02

        self.t0 = 1e-5
        self.t1 = 1

        self.input_size = (1, 28, 28)


class DDPM_Forward(nn.Module):
    def __init__(self, args: DDPM_Params):
        super().__init__()

        self.args = args

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        b_min = self.args.beta_min
        b_max = self.args.beta_max

        return b_min + t * (b_max - b_min)

    def _indefinite_int(self, t: torch.Tensor) -> torch.Tensor:
        b_min = self.args.beta_min
        b_max = self.args.beta_max

        return b_min * t + 0.5 * t**2 * (b_max - b_min)

    def analytical_mean(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        int_t = self._indefinite_int(t)
        int_t_0 = self._indefinite_int(self.args.t0)

        mean_coef = (-0.5 * (int_t - int_t_0)).exp()
        mean = x_0 * fill_tail_dims(mean_coef, x_0)

        return mean

    def analytical_var(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        int_t = self._indefinite_int(t)
        int_t_0 = self._indefinite_int(self.args.t0)

        var = 1 - (-int_t + int_t_0).exp()

        return var

    @torch.no_grad()
    def analytical_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean = self.analytical_mean(x_0, t)
        var = self.analytical_var(x_0, t)
        return mean + torch.randn_like(mean) * fill_tail_dims(var.sqrt(), mean)

    @torch.no_grad()
    def analytical_score(
        self,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        mean = self.analytical_mean(x_0, t)
        var = self.analytical_var(x_0, t)
        return -(x_t - mean) / fill_tail_dims(var, mean).clamp_min(1e-5)

    def s_theta(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.args.eps_theta(x=x, t=t)

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * self._beta(t) * x

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return fill_tail_dims(self._beta(t).sqrt(), x).expand_as(x)

    @torch.no_grad()
    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.analytical_sample(x_0, t)


class DDPM_Reverse(nn.Module):
    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(
        self,
        forward_sde: DDPM_Forward,
        args: DDPM_Params,
    ):
        super().__init__()

        self.forward_sde = forward_sde
        self.args = args

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, *self.args.input_size)

        f1 = self.forward_sde.f(-t, x)
        f2 = self.forward_sde.g(-t, x) ** 2 * self.forward_sde.s_theta(-t, x)

        f = f1 - f2

        return -f.flatten(1)

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, *self.args.input_size)

        g = self.forward_sde.g(-t, x)
        return -g.flatten(1)

    def prob_flow_ode(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, *self.args.input_size)

        f = self.forward_sde.f(-t, x)
        g = self.forward_sde.g(-t, x)

        flow = f - 0.5 * g**2 * self.forward_sde.s_theta(-t, x)

        return -flow.flatten(1)

    @torch.no_grad()
    def forward(
        self,
        x_t: torch.Tensor,
        dt: float = 1e-2,
        use_sde: bool = True,
    ) -> torch.Tensor:
        t = torch.tensor(
            [-self.args.t1, -self.args.t0],
            device=self.args.device,
            dtype=torch.float32,
        )

        if use_sde:
            x_0 = torchsde.sdeint(
                self,
                x_t.flatten(1),
                t,
                dt=dt,
            ).view(len(t), *x_t.size())
        else:
            x_0 = torchdiffeq.odeint(
                self.prob_flow_ode,
                x_t.flatten(1),
                t,
                method="rk4",
                options={"step_size": dt},
            ).view(len(t), *x_t.size())

        return x_0


class DDPM(Diffusion):
    def __init__(self, args: DDPM_Params):
        super().__init__()

        self.args = args

        self.f_sde = DDPM_Forward(args)
        self.r_sde = DDPM_Reverse(self.f_sde, args)

    def t(self, n: int):
        r = torch.rand((n,), device=self.args.device)
        return self.args.t0 + (self.args.t1 - self.args.t0) * r

    def forward(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ):
        return self.f_sde(x_0, t)

    def reverse(self, x_t: torch.Tensor, use_sde: bool = True):
        return self.r_sde(x_t, use_sde=use_sde)[-1]

    def sample(self, n: int, use_sde: bool = True):
        x_t = torch.randn(size=(n, *self.args.input_size), device=self.args.device)

        return self.r_sde(x_t, use_sde=use_sde)[-1]

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
        lambda_t = self.f_sde.analytical_var(x_0, t)

        score_pred = self.f_sde.s_theta(t, x_t)
        score_true = self.f_sde.analytical_score(x_t, x_0, t)

        loss = fill_tail_dims(lambda_t, x_t) * ((score_pred - score_true) ** 2)

        return loss.mean()
