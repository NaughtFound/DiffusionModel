import torch
import torch.nn as nn
from utils import fill_tail_dims
from .ddpm import SDE_DDPM_Forward, SDE_DDPM_Reverse, SDE_DDPM


class SDE_SMLD_Params:
    eps_theta: nn.Module
    sigma_max: float
    sigma_min: float
    t0: float
    t1: float
    input_size: tuple[int, int, int]

    def __init__(self, device: torch.device):
        self.device = device

        self.eps_theta = None
        self.sigma_max = 50
        self.sigma_min = 0.01

        self.t0 = 0
        self.t1 = 1

        self.input_size = (1, 28, 28)


class SDE_SMLD_Forward(SDE_DDPM_Forward):
    def __init__(self, args: SDE_SMLD_Params):
        super().__init__(args)

        self.args = args

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        s_min = self.args.sigma_min
        s_max = self.args.sigma_max

        return s_min * (s_max / s_min) ** t

    def analytical_mean(self, x_0: torch.Tensor) -> torch.Tensor:
        return x_0

    def analytical_var(self, t: torch.Tensor) -> torch.Tensor:
        return self._sigma(t) ** 2

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        s_min = self.args.sigma_min
        s_max = self.args.sigma_max

        g = self._sigma(t) * (2 * torch.log(s_max / s_min)).sqrt()

        return fill_tail_dims(g, x).expand_as(x)


class SDE_SMLD_Reverse(SDE_DDPM_Reverse):
    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(
        self,
        forward_sde: SDE_SMLD_Forward,
        args: SDE_SMLD_Params,
    ):
        super().__init__()

        self.forward_sde = forward_sde
        self.args = args


class SDE_SMLD(SDE_DDPM):
    def __init__(self, args: SDE_SMLD_Params):
        super().__init__()

        self.args = args

        self.f_sde = SDE_SMLD_Forward(args)
        self.r_sde = SDE_SMLD_Reverse(self.f_sde, args)
