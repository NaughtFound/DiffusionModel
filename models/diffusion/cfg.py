import torch
from utils.args import with_kwargs, KWargs
from .ddpm import DDPM, DDPM_Params, DDPM_Forward, DDPM_Reverse
from models.modules import HasCFGBackBone


class CFG_Params(DDPM_Params):
    eps_theta: HasCFGBackBone


class CFG_Forward(DDPM_Forward):
    def __init__(self, args: CFG_Params):
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
        else:
            return self.args.eps_theta.forward_with_cfg(
                x=x,
                t=t,
                y=labels,
                cfg_scale=cfg_scale,
            )


class CFG_Reverse(DDPM_Reverse):
    def __init__(self, forward_sde: CFG_Forward, args: CFG_Params):
        super().__init__(forward_sde, args)

        self.forward_sde = forward_sde
        self.args = args


class CFG(DDPM):
    def __init__(self, args: CFG_Params):
        super().__init__(args)

        self.args = args

        self.f_sde = CFG_Forward(args)
        self.r_sde = CFG_Reverse(self.f_sde, args)

    def predict_noise(
        self,
        x_t: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor,
        cfg_scale: float,
    ):
        KWargs().insert(self.f_sde.s_theta, labels=labels, cfg_scale=cfg_scale)

        noise = super().predict_noise(x_t, t)

        KWargs().drop(self.f_sde.s_theta)

        return noise

    def reverse(
        self,
        x_t: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
        use_sde: bool = True,
    ):
        KWargs().insert(self.f_sde.s_theta, labels=labels, cfg_scale=cfg_scale)

        x_0 = super().reverse(x_t, use_sde)

        KWargs().drop(self.f_sde.s_theta)

        return x_0

    def sample(
        self,
        n: int,
        labels: torch.Tensor,
        cfg_scale: float,
        use_sde: bool = True,
    ):
        KWargs().insert(self.f_sde.s_theta, labels=labels, cfg_scale=cfg_scale)

        x_0 = super().sample(n, use_sde)

        KWargs().drop(self.f_sde.s_theta)

        return x_0

    def calc_loss(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
    ):
        KWargs().insert(self.f_sde.s_theta, labels=labels, cfg_scale=cfg_scale)

        loss = super().calc_loss(x_0, t)

        KWargs().drop(self.f_sde.s_theta)

        return loss
