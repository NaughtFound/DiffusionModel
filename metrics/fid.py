from typing import Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg
from .base import Metric, MetricMeta


class FIDMeta(MetricMeta):
    inception: nn.Module
    forward_method: Callable[[torch.Tensor], torch.Tensor]


class FID(Metric):
    def __init__(self, meta: FIDMeta):
        super().__init__(meta)

        self.meta = meta

    @torch.no_grad()
    def _calc_features(
        self, dataloader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        list_rf = []
        list_ff = []

        for batch in dataloader:
            real_images = batch.to(self.meta.device)
            fake_images = self.meta.forward_method(real_images)

            rf = self.meta.inception(real_images)[0]
            ff = self.meta.inception(fake_images)[0]

            if rf.size(2) != 1 or rf.size(3) != 1:
                rf = adaptive_avg_pool2d(rf, output_size=(1, 1))

            if ff.size(2) != 1 or ff.size(3) != 1:
                ff = adaptive_avg_pool2d(ff, output_size=(1, 1))

            rf = rf.squeeze(3).squeeze(2)
            list_rf.append(rf)

            ff = ff.squeeze(3).squeeze(2)
            list_ff.append(ff)

        list_rf = torch.cat(list_rf, dim=0)
        list_ff = torch.cat(list_ff, dim=0)

        r_mu = list_rf.mean(0)
        r_sigma = list_rf.T.cov()

        f_mu = list_ff.mean(0)
        f_sigma = list_ff.T.cov()

        return r_mu, r_sigma, f_mu, f_sigma

    def calc(self, dataloader: DataLoader):
        r_mu, r_sigma, f_mu, f_sigma = self._calc_features(dataloader)

        cov_prod = linalg.sqrtm((r_sigma @ f_sigma).cpu().numpy())
        cov_prod = torch.tensor(cov_prod, dtype=torch.float32, device=self.meta.device)

        fid_score = torch.norm(r_mu - f_mu) ** 2 + torch.trace(
            r_sigma + f_sigma - 2 * cov_prod
        )

        return fid_score
