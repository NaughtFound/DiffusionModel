import torch
from torch import nn

from utils.args import KWargs, with_kwargs

from .ddpm import DDPM, DDPMForward, DDPMParams, DDPMReverse


class LDMParams(DDPMParams):
    tau_theta: nn.Module | None

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)

        self.tau_theta = None


class LDMForward(DDPMForward):
    def __init__(self, args: LDMParams) -> None:
        super().__init__(args)

        self.args = args

    @with_kwargs
    def s_theta(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.args.tau_theta, nn.Module):
            y = self.args.tau_theta(y)

        return self.args.eps_theta(x=x, t=t, y=y)


class LDMReverse(DDPMReverse):
    def __init__(self, forward_sde: LDMForward, args: LDMParams) -> None:
        super().__init__(forward_sde, args)

        self.forward_sde = forward_sde
        self.args = args


class LDM(DDPM):
    def __init__(self, args: LDMParams) -> None:
        super().__init__(args)

        self.args = args

        self.f_sde = LDMForward(args)
        self.r_sde = LDMReverse(self.f_sde, args)

    def predict_noise(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        KWargs.insert(self.f_sde.s_theta, y=y)

        noise = super().predict_noise(x_t, t)

        KWargs.drop(self.f_sde.s_theta)

        return noise

    def reverse(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        *,
        use_sde: bool = True,
    ) -> torch.Tensor:
        KWargs.insert(self.f_sde.s_theta, y=y)

        x_0 = super().reverse(x_t, use_sde=use_sde)

        KWargs.drop(self.f_sde.s_theta)

        return x_0

    def sample(
        self,
        n: int,
        y: torch.Tensor,
        *,
        use_sde: bool = True,
    ) -> torch.Tensor:
        KWargs.insert(self.f_sde.s_theta, y=y)

        x_0 = super().sample(n, use_sde=use_sde)

        KWargs.drop(self.f_sde.s_theta)

        return x_0

    def calc_loss(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        KWargs.insert(self.f_sde.s_theta, y=y)

        loss = super().calc_loss(x_0, t)

        KWargs.drop(self.f_sde.s_theta)

        return loss

    def train(self, *, with_tau: bool = True) -> None:
        super().train()
        if with_tau and isinstance(self.args.tau_theta, nn.Module):
            self.args.tau_theta.train()

    def eval(self, *, with_tau: bool = True) -> None:
        super().eval()
        if with_tau and isinstance(self.args.tau_theta, nn.Module):
            self.args.tau_theta.eval()
