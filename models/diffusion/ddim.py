import torch
import torchdiffeq
from utils import fill_tail_dims
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

    def g_ode(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        s_min = self.args.sigma_min
        s_max = self.args.sigma_max

        s_min_max = torch.tensor(s_max / s_min, device=self.args.device)

        g = self._sigma(t) * (2 * s_min_max.log()).sqrt()

        return fill_tail_dims(g, x).expand_as(x)

    def prob_flow_ode(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, *self.args.input_size)

        g = self.g_ode(t, x)
        s_t = self._sigma(t)

        x_score = x / (s_t**2 + 1).sqrt()

        score_pred = self.s_theta(t, x_score) / s_t

        flow = 0.5 * g**2 * score_pred

        return -flow.flatten(1)

    @torch.no_grad()
    def forward(
        self,
        x_0: torch.Tensor,
        t: float = None,
        dt: float = 1e-2,
    ) -> torch.Tensor:
        if t is None:
            t = self.args.t1

        t = torch.tensor(
            [self.args.t0, t],
            device=self.args.device,
            dtype=torch.float32,
        )

        x_t = torchdiffeq.odeint(
            self.prob_flow_ode,
            x_0.flatten(1),
            t,
            method="rk4",
            options={"step_size": dt},
        ).view(len(t), *x_0.size())

        return x_t


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

    def prob_flow_ode(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        flow = self.forward_sde.prob_flow_ode(-t, x)

        return -flow


class DDIM(DDPM):
    def __init__(self, args: DDIM_Params):
        super().__init__(args)

        self.args = args

        self.f_sde = DDIM_Forward(args)
        self.r_sde = DDIM_Reverse(self.f_sde, args)

    def forward(self, x_0: torch.Tensor, t: float = None):
        x_t = self.f_sde(x_0, t=t)
        return x_t[-1]

    def reverse(self, x_t: torch.Tensor):
        x_0 = self.r_sde(x_t, use_sde=False)

        return x_0[-1]
