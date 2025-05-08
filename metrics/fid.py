from typing import Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.linalg
import numpy as np
from .base import Metric, MetricMeta


class FIDMeta(MetricMeta):
    inception: nn.Module
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]]
    forward_method: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self):
        super().__init__()

        self.transform = None


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
            if isinstance(batch, list):
                batch = batch[0]

            real_images = batch.to(self.meta.device)
            fake_images = self.meta.forward_method(real_images)

            if self.meta.transform is not None:
                real_images = self.meta.transform(real_images)
                fake_images = self.meta.transform(fake_images)

            rf = self.meta.inception(real_images)
            ff = self.meta.inception(fake_images)

            if isinstance(rf, list):
                rf = rf[0]
                ff = ff[0]

            list_rf.append(rf)
            list_ff.append(ff)

        list_rf = torch.cat(list_rf, dim=0)
        list_ff = torch.cat(list_ff, dim=0)

        r_mu = list_rf.mean(0)
        r_sigma = list_rf.T.cov()

        f_mu = list_ff.mean(0)
        f_sigma = list_ff.T.cov()

        return r_mu, r_sigma, f_mu, f_sigma

    def _fid_score(
        self,
        mu1: torch.Tensor,
        mu2: torch.Tensor,
        sigma1: torch.Tensor,
        sigma2: torch.Tensor,
        eps: float = 1e-6,
    ) -> float:
        mu1, mu2 = mu1.cpu(), mu2.cpu()
        sigma1, sigma2 = sigma1.cpu(), sigma2.cpu()

        diff = mu1 - mu2

        covmean, _ = scipy.linalg.sqrtm(sigma1.mm(sigma2), disp=False)

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        if not np.isfinite(covmean).all():
            tr_covmean = np.sum(
                np.sqrt(
                    ((np.diag(sigma1) * eps) * (np.diag(sigma2) * eps)) / (eps * eps)
                )
            )

        return float(
            diff.dot(diff).item()
            + torch.trace(sigma1)
            + torch.trace(sigma2)
            - 2 * tr_covmean
        )

    def calc(self, dataloader: DataLoader):
        r_mu, r_sigma, f_mu, f_sigma = self._calc_features(dataloader)

        return self._fid_score(mu1=r_mu, mu2=f_mu, sigma1=r_sigma, sigma2=f_sigma)
