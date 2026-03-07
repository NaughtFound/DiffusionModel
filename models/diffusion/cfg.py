import torch

from models.common.cfg import HasCFGBackBone
from utils.args import KWargs, with_kwargs

from .ddpm import DDPM, DDPMForward, DDPMParams, DDPMReverse


class CFGParams(DDPMParams):
    eps_theta: HasCFGBackBone
    fast_cfg: bool

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)

        self.fast_cfg = True


class CFGForward(DDPMForward):
    def __init__(self, args: CFGParams) -> None:
        super().__init__(args)

        self.args = args

    @with_kwargs
    def s_theta(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        if self.training:
            return self.args.eps_theta(
                x=x,
                t=t,
                y=labels,
            )
        return self.args.eps_theta.forward_with_cfg(
            x=x,
            t=t,
            y=labels,
            cfg_scale=cfg_scale,
            fast_cfg=self.args.fast_cfg,
        )


class CFGReverse(DDPMReverse):
    def __init__(self, forward_sde: CFGForward, args: CFGParams) -> None:
        super().__init__(forward_sde, args)

        self.forward_sde = forward_sde
        self.args = args


class CFG(DDPM):
    def __init__(self, args: CFGParams) -> None:
        super().__init__(args)

        self.args = args

        self.f_sde = CFGForward(args)
        self.r_sde = CFGReverse(self.f_sde, args)

    def predict_noise(
        self,
        x_t: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        KWargs.insert(self.f_sde.s_theta, labels=labels, cfg_scale=cfg_scale)

        noise = super().predict_noise(x_t, t)

        KWargs.drop(self.f_sde.s_theta)

        return noise

    def reverse(
        self,
        x_t: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
        *,
        use_sde: bool = True,
    ) -> torch.Tensor:
        KWargs.insert(self.f_sde.s_theta, labels=labels, cfg_scale=cfg_scale)

        x_0 = super().reverse(x_t, use_sde=use_sde)

        KWargs.drop(self.f_sde.s_theta)

        return x_0

    def sample(
        self,
        n: int,
        labels: torch.Tensor,
        cfg_scale: float,
        *,
        use_sde: bool = True,
    ) -> torch.Tensor:
        KWargs.insert(self.f_sde.s_theta, labels=labels, cfg_scale=cfg_scale)

        x_0 = super().sample(n, use_sde=use_sde)

        KWargs.drop(self.f_sde.s_theta)

        return x_0

    def calc_loss(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        KWargs.insert(self.f_sde.s_theta, labels=labels, cfg_scale=cfg_scale)

        loss = super().calc_loss(x_0, t)

        KWargs.drop(self.f_sde.s_theta)

        return loss
