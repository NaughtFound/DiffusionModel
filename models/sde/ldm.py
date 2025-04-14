import torch
import torch.nn as nn
import torchsde
from utils import fill_tail_dims
from utils.args import with_kwargs, KWargs
from .ddpm import SDE_DDPM, SDE_DDPM_Params, SDE_DDPM_Forward, SDE_DDPM_Reverse


class SDE_LDM_Params(SDE_DDPM_Params):
    tau_theta: nn.Module


class SDE_LDM_Forward(SDE_DDPM_Forward):
    def __init__(self, args: SDE_LDM_Params):
        super().__init__(args)

        self.args = args

    def s_theta(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        y = self.args.tau_theta(y)

        return self.args.eps_theta(x=x, t=t, y=y)


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
        y: torch.Tensor,
    ) -> torch.Tensor:
        x = x.view(-1, *self.args.input_size)

        score = self.forward_sde.s_theta(-t, x, y)

        f1 = self.forward_sde.f(-t, x)
        f2 = self.forward_sde.g(-t, x) ** 2 * score

        f = -(f1 - f2)

        return f.flatten(1)

    @torch.no_grad()
    def forward(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        dt: float = 1e-2,
    ) -> torch.Tensor:
        t = torch.tensor([-self.args.t1, -self.args.t0], device=self.args.device)

        KWargs().insert(self.f, y=y)

        x_s = torchsde.sdeint(self, x_t.flatten(1), t, dt=dt).view(len(t), *x_t.size())

        KWargs().drop(self.f)

        return x_s


class SDE_LDM(SDE_DDPM):
    def __init__(self, args: SDE_LDM_Params):
        super().__init__(args)

        self.args = args

        self.f_sde = SDE_LDM_Forward(args)
        self.r_sde = SDE_LDM_Reverse(self.f_sde, args)

    def sample(
        self,
        n: int,
        y: torch.Tensor,
    ):
        x_t = torch.randn(size=(n, *self.args.input_size), device=self.args.device)

        return self.r_sde(x_t, y)[-1]

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
        lambda_t = self.f_sde.analytical_var(t)

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
