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
        fast_cfg: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if y_null is None and fast_cfg:
            y_null = torch.zeros_like(y)

        if t.dim() == 0:
            t = t.expand(len(x))

        if fast_cfg:
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)
            y_in = torch.cat([y_null, y], dim=0)

            x_hat = self.__call__(x=x_in, t=t_in, y=y_in, **kwargs)

            eps, rest = x_hat[:, : self.in_channels], x_hat[:, self.in_channels :]

            u_eps, c_eps = eps.chunk(2, dim=0)

            g_eps = torch.lerp(u_eps, c_eps, cfg_scale)

            if rest.shape[1] > 0:
                u_res, _ = rest.chunk(2, dim=0)
                g_eps = torch.cat([g_eps, u_res], dim=1)

        else:
            u_x_hat = self.__call__(x=x, t=t, y=y_null, **kwargs)
            c_x_hat = self.__call__(x=x, t=t, y=y, **kwargs)

            c_eps = c_x_hat[:, : self.in_channels]
            u_eps = u_x_hat[:, : self.in_channels]
            u_res = u_x_hat[:, self.in_channels :]

            g_eps = torch.lerp(u_eps, c_eps, cfg_scale)

            if u_res.shape[1] > 0:
                g_eps = torch.cat([g_eps, u_res], dim=1)

        return g_eps
