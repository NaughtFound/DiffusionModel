import torch
import torch.nn as nn
import torchsde
from utils import fill_tail_dims
from utils.args import with_kwargs, KWargs
from .ddpm import SDE_DDPM, SDE_DDPM_Params, SDE_DDPM_Forward, SDE_DDPM_Reverse


class SDE_LDM_Params(SDE_DDPM_Params):
    encoder: nn.Module


class SDE_LDM_Forward(SDE_DDPM_Forward):
    def __init__(self, args: SDE_LDM_Params):
        super().__init__(args)

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


class SDE_LDM_Reverse(SDE_DDPM_Reverse):
    def __init__(self, forward_sde: SDE_LDM_Forward, args: SDE_LDM_Params):
        super().__init__(forward_sde, args)

        self.forward_sde = forward_sde
        self.args = args

    @with_kwargs
    def f(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        x = x.view(-1, *self.args.input_size)

        score = self.forward_sde.s_theta(-t, x, labels, cfg_scale)

        f1 = self.forward_sde.f(-t, x)
        f2 = self.forward_sde.g(-t, x) ** 2 * score

        f = -(f1 - f2)

        return f.flatten(1)

    @torch.no_grad()
    def forward(
        self,
        n: int,
        labels: torch.Tensor,
        cfg_scale: float,
        dt: float = 1e-2,
    ) -> torch.Tensor:
        t = torch.tensor([-self.args.t1, -self.args.t0], device=self.args.device)
        x_t = torch.randn(size=(n, *self.args.input_size), device=self.args.device)

        KWargs().insert(self.f, labels=labels, cfg_scale=cfg_scale)

        x_s = torchsde.sdeint(self, x_t.flatten(1), t, dt=dt).view(len(t), *x_t.size())

        x_0 = (x_s.clamp(-1, 1) + 1) / 2
        x_0 = (x_0 * 255).to(torch.uint8)

        return x_0


class SDE_LDM(SDE_DDPM):
    def __init__(self, args: SDE_LDM_Params):
        super().__init__(args)

        self.args = args

        self.f_sde = SDE_LDM_Forward(args)
        self.r_sde = SDE_LDM_Reverse(self.f_sde, args)

    def sample(
        self,
        n: int,
        labels: torch.Tensor,
        cfg_scale: float,
    ):
        return self.r_sde.forward(n, labels, cfg_scale)[-1]

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
        lambda_t = self.f_sde.analytical_var(t)

        score_pred = self.f_sde.s_theta(t, x_t, labels, cfg_scale)
        score_true = self.f_sde.analytical_score(x_t, x_0, t)

        loss = fill_tail_dims(lambda_t, x_t) * ((score_pred - score_true) ** 2)

        return loss.mean()
