from typing import Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.linalg
import numpy as np
from tqdm import tqdm
from .base import Metric, MetricMeta


class FIDMeta(MetricMeta):
    inception: nn.Module
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]]
    forward_method: Callable[[torch.Tensor], torch.Tensor]
    num_images: int
    num_features: int

    def __init__(self):
        super().__init__()

        self.transform = None


class _FIDPair:
    def __init__(self, num_features: int, device: torch.device):
        self.total = torch.zeros(
            num_features,
            dtype=torch.float32,
            device=device,
        )

        self.sigma = torch.zeros(
            (num_features, num_features),
            dtype=torch.float32,
            device=device,
        )

    def update(self, features: torch.Tensor):
        self.total += features
        self.sigma += torch.outer(features, features)

    def calc_mean(self, n: int):
        return self.total / n

    def calc_cov(self, n: int):
        sub_matrix = torch.outer(self.total, self.total)
        sub_matrix = sub_matrix / n

        return (self.sigma - sub_matrix) / (n - 1)


class FID(Metric):
    def __init__(self, meta: FIDMeta):
        super().__init__(meta)

        self.meta = meta

    @torch.no_grad()
    def _calc_features(self, dataloader: DataLoader) -> tuple[_FIDPair, _FIDPair]:
        r_pair = _FIDPair(self.meta.num_features, self.meta.device)
        f_pair = _FIDPair(self.meta.num_features, self.meta.device)

        for batch in tqdm(dataloader, desc="Calculating"):
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

            for f in rf:
                r_pair.update(f)

            for f in ff:
                f_pair.update(f)

        return r_pair, f_pair

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
        r_pair, f_pair = self._calc_features(dataloader)

        return self._fid_score(
            mu1=r_pair.calc_mean(self.meta.num_images),
            mu2=f_pair.calc_mean(self.meta.num_images),
            sigma1=r_pair.calc_cov(self.meta.num_images),
            sigma2=f_pair.calc_cov(self.meta.num_images),
        )
