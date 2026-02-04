from typing import Optional
import torch
import torch.nn as nn
from utils.args import with_kwargs, KWargs
from .ddpm import DDPM, DDPM_Params, DDPM_Forward, DDPM_Reverse


class LDM_Params(DDPM_Params):
    tau_theta: Optional[nn.Module]

    def __init__(self, device: torch.device):
        super().__init__(device)

        self.tau_theta = None


class LDM_Forward(DDPM_Forward):
    def __init__(self, args: LDM_Params):
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


class LDM_Reverse(DDPM_Reverse):
    def __init__(self, forward_sde: LDM_Forward, args: LDM_Params):
        super().__init__(forward_sde, args)

        self.forward_sde = forward_sde
        self.args = args


class LDM(DDPM):
    def __init__(self, args: LDM_Params):
        super().__init__(args)

        self.args = args

        self.f_sde = LDM_Forward(args)
        self.r_sde = LDM_Reverse(self.f_sde, args)

    def predict_noise(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ):
        KWargs.insert(self.f_sde.s_theta, y=y)

        noise = super().predict_noise(x_t, t)

        KWargs.drop(self.f_sde.s_theta)

        return noise

    def reverse(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        use_sde: bool = True,
    ):
        KWargs.insert(self.f_sde.s_theta, y=y)

        x_0 = super().reverse(x_t, use_sde)

        KWargs.drop(self.f_sde.s_theta)

        return x_0

    def sample(
        self,
        n: int,
        y: torch.Tensor,
        use_sde: bool = True,
    ):
        KWargs.insert(self.f_sde.s_theta, y=y)

        x_0 = super().sample(n, use_sde)

        KWargs.drop(self.f_sde.s_theta)

        return x_0

    def calc_loss(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ):
        KWargs.insert(self.f_sde.s_theta, y=y)

        loss = super().calc_loss(x_0, t)

        KWargs.drop(self.f_sde.s_theta)

        return loss

    def train(self, with_tau: bool = True):
        super().train()
        if with_tau and isinstance(self.args.tau_theta, nn.Module):
            self.args.tau_theta.train()

    def eval(self, with_tau: bool = True):
        super().eval()
        if with_tau and isinstance(self.args.tau_theta, nn.Module):
            self.args.tau_theta.eval()
