import torch
from utils import fill_tail_dims
from utils.args import with_kwargs, KWargs
from .ddpm import DDPM, DDPM_Params, DDPM_Forward, DDPM_Reverse


class CFG_Params(DDPM_Params):
    pass


class CFG_Forward(DDPM_Forward):
    def __init__(self, args: CFG_Params):
        super().__init__(args)

    @with_kwargs
    def s_theta(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        if cfg_scale == 0:
            return super().s_theta(t, x)

        if cfg_scale == 1:
            return self.args.eps_theta(x=x, t=t, labels=labels)

        unconditional_noise = self.args.eps_theta(x=x, t=t, labels=None)
        conditional_noise = self.args.eps_theta(x=x, t=t, labels=labels)

        return torch.lerp(unconditional_noise, conditional_noise, cfg_scale)


class CFG_Reverse(DDPM_Reverse):
    def __init__(self, forward_sde: CFG_Forward, args: CFG_Params):
        super().__init__(forward_sde, args)

        self.forward_sde = forward_sde
        self.args = args

    @torch.no_grad()
    def forward(
        self,
        x_t: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
        dt: float = 1e-2,
        use_sde: bool = True,
    ) -> torch.Tensor:
        KWargs().insert(self.forward_sde.s_theta, labels=labels, cfg_scale=cfg_scale)

        x_0 = super().forward(x_t, dt, use_sde)

        KWargs().drop(self.forward_sde.s_theta)

        return x_0


class CFG(DDPM):
    def __init__(self, args: CFG_Params):
        super().__init__(args)

        self.args = args

        self.f_sde = CFG_Forward(args)
        self.r_sde = CFG_Reverse(self.f_sde, args)

    def reverse(
        self,
        x_t: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
        use_sde: bool = True,
    ):
        return self.r_sde(x_t, labels, cfg_scale, use_sde=use_sde)[-1]

    def sample(
        self,
        n: int,
        labels: torch.Tensor,
        cfg_scale: float,
        use_sde: bool = True,
    ):
        x_t = torch.randn(size=(n, *self.args.input_size), device=self.args.device)

        return self.r_sde(x_t, labels, cfg_scale, use_sde=use_sde)[-1]

    def predict_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
    ):
        return self.f_sde.s_theta(t, x_t, labels, cfg_scale)

    def calc_loss(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
    ):
        x_t = self.f_sde.analytical_sample(x_0, t)
        lambda_t = self.f_sde.analytical_var(x_0, t)

        score_pred = self.f_sde.s_theta(t, x_t, labels, cfg_scale)
        score_true = self.f_sde.analytical_score(x_t, x_0, t)

        loss = fill_tail_dims(lambda_t, x_t) * ((score_pred - score_true) ** 2)

        return loss.mean()
