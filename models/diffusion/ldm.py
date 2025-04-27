import torch
import torch.nn as nn
from utils import fill_tail_dims
from utils.args import with_kwargs, KWargs
from .ddpm import DDPM, DDPM_Params, DDPM_Forward, DDPM_Reverse


class LDM_Params(DDPM_Params):
    tau_theta: nn.Module


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
        y = self.args.tau_theta(y)

        return self.args.eps_theta(x=x, t=t, y=y)


class LDM_Reverse(DDPM_Reverse):
    def __init__(self, forward_sde: LDM_Forward, args: LDM_Params):
        super().__init__(forward_sde, args)

        self.forward_sde = forward_sde
        self.args = args

    @torch.no_grad()
    def forward(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        dt: float = 1e-2,
        use_sde: bool = True,
    ) -> torch.Tensor:
        KWargs().insert(self.forward_sde.s_theta, y=y)

        x_0 = super().forward(x_t, dt, use_sde)

        KWargs().drop(self.forward_sde.s_theta)

        return x_0


class LDM(DDPM):
    def __init__(self, args: LDM_Params):
        super().__init__(args)

        self.args = args

        self.f_sde = LDM_Forward(args)
        self.r_sde = LDM_Reverse(self.f_sde, args)

    def reverse(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        use_sde: bool = True,
    ):
        return self.r_sde(x_t, y, use_sde=use_sde)[-1]

    def sample(
        self,
        n: int,
        y: torch.Tensor,
        use_sde: bool = True,
    ):
        x_t = torch.randn(size=(n, *self.args.input_size), device=self.args.device)

        return self.r_sde(x_t, y, use_sde=use_sde)[-1]

    def predict_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ):
        return self.f_sde.s_theta(t, x_t, y)

    def calc_loss(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ):
        x_t = self.f_sde.analytical_sample(x_0, t)
        lambda_t = self.f_sde.analytical_var(x_0, t)

        score_pred = self.f_sde.s_theta(t, x_t, y)
        score_true = self.f_sde.analytical_score(x_t, x_0, t)

        loss = fill_tail_dims(lambda_t, x_t) * ((score_pred - score_true) ** 2)

        return loss.mean()

    def train(self, with_tau: bool = True):
        super().train()
        if with_tau:
            self.args.tau_theta.train()

    def eval(self, with_tau: bool = True):
        super().eval()
        if with_tau:
            self.args.tau_theta.eval()
