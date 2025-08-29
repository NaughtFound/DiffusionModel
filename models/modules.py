from typing import Any, Callable, Optional, Protocol
import torch


class HasCFGBackBone(Protocol):
    in_channels: int

    __call__: Callable[..., Any]

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
        y_null: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if y_null is None:
            y_null = torch.zeros_like(y)

        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        y_in = torch.cat([y_null, y], dim=0)

        x_hat = self.__call__(x=x_in, t=t_in, y=y_in, **kwargs)

        eps, rest = x_hat[:, : self.in_channels], x_hat[:, self.in_channels :]

        u_eps, c_eps = eps.chunk(2, dim=0)

        g_eps = torch.lerp(u_eps, c_eps, cfg_scale)

        if rest.shape[1] > 0:
            u_res, _ = rest.chunk(2, dim=0)
            return torch.cat([g_eps, u_res], dim=1)

        return g_eps
