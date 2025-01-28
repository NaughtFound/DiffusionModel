import torch
import torch.nn as nn


class SDE_DDPM_Params:
    eps_theta: nn.Module
    beta_start: float
    beta_end: float
    T: int

    def __init__(self, device: torch.device):
        self.device = device

        self.eps_theta = None
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.T = 1000

        self.beta = self.beta_scheduler()

    def beta_scheduler(self):
        return torch.linspace(
            self.beta_start,
            self.beta_end,
            self.T,
            device=self.device,
        )


class SDE_DDPM_Forward(nn.Module):
    def __init__(self, args: SDE_DDPM_Params):
        super().__init__()

        self.args = args

    def _beta(self, t: torch.Tensor):
        return self.args.beta[t]

    def s_theta(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.args.eps_theta(x=x, t=t)

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * self._beta(t) * x

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self._beta(t))


class SDE_DDPM_Reverse(nn.Module):
    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(
        self,
        forward_sde: SDE_DDPM_Forward,
    ):
        super().__init__()

        self.forward_sde = forward_sde

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        f1 = self.forward_sde.f(-t, x)
        f2 = self.forward_sde.g(-t, x) ** 2 * self.forward_sde.s_theta(-t, x)

        return -(f1 - f2)

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g = self.forward_sde.g(-t, x)
        return -g
