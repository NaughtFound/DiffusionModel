import torch
import torch.nn as nn


class SDE_DDPM_Forward(nn.Module):
    def __init__(self):
        super().__init__()

    def _beta(self, t: torch.Tensor):
        pass

    def s_theta(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -0.5 * self._beta(t) * y

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        f1 = self.forward_sde.f(-t, y)
        f2 = self.forward_sde.g(-t, y) ** 2 * self.forward_sde.s_theta(-t, y)

        return -(f1 - f2)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        g = self.forward_sde.g(-t, y)
        return -g
