import torch
import torchdiffeq
from .ddpm import DDPM_Params, DDPM_Forward, DDPM_Reverse, DDPM


class DDIM_Params(DDPM_Params):
    sigma_max: float
    sigma_min: float

    def __init__(self, device: torch.device):
        super().__init__(device)

        self.sigma_max = 50
        self.sigma_min = 0.01


class DDIM_Forward(DDPM_Forward):
    def __init__(self, args: DDIM_Params):
        super().__init__(args)

        self.args = args

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        s_min = self.args.sigma_min
        s_max = self.args.sigma_max

        return s_min * (s_max / s_min) ** t

    @torch.no_grad()
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        s_min = self.args.sigma_min
        s_max = self.args.sigma_max
        s_t = self._sigma(t)

        s_min_max = torch.tensor(s_max / s_min, device=self.args.device)

        x = x / (s_t**2 + 1).sqrt()

        score_pred = self.s_theta(t, x) / s_t
        g = s_t * (2 * s_min_max.log()).sqrt()
        f = 0.5 * g**2 * score_pred

        return f.flatten(1)

    @torch.no_grad()
    def ode_forward(self, x_0: torch.Tensor, dt: float = 1e-2) -> torch.Tensor:
        t = torch.tensor([self.args.t0, self.args.t1], device=self.args.device)

        x_o = torchdiffeq.odeint(self, x_0, t, options={"step_size": dt})

        return x_o


class DDIM_Reverse(DDPM_Reverse):
    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(
        self,
        forward_sde: DDIM_Forward,
        args: DDIM_Params,
    ):
        super().__init__(forward_sde, args)

        self.forward_sde = forward_sde
        self.args = args

    @torch.no_grad()
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sde(-t, x)

    @torch.no_grad()
    def ode_forward(self, x_t: torch.Tensor, dt: float = 1e-2) -> torch.Tensor:
        t = torch.tensor([-self.args.t1, -self.args.t0], device=self.args.device)

        x_o = torchdiffeq.odeint(self, x_t, t, options={"step_size": dt})

        return x_o


class DDIM(DDPM):
    def __init__(self, args: DDIM_Params):
        super().__init__(args)

        self.args = args

        self.f_sde = DDIM_Forward(args)
        self.r_sde = DDIM_Reverse(self.f_sde, args)
