import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg
from .base import Metric, MetricMeta


class FIDMeta(MetricMeta):
    inception: nn.Module
    batch_size: int


class FID(Metric):
    def __init__(self, meta: FIDMeta):
        super().__init__(meta)

        self.meta = meta

    @torch.no_grad()
    def _calc_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dataset = TensorDataset(images)
        dataloader = DataLoader(dataset, self.meta.batch_size)

        features = []

        for batch in dataloader:
            batch = batch.to(self.meta.device)
            batch_features = self.meta.inception(batch)[0]

            if batch_features.size(2) != 1 or batch_features.size(3) != 1:
                batch_features = adaptive_avg_pool2d(batch_features, output_size=(1, 1))

            batch_features = batch_features.squeeze(3).squeeze(2)
            features.append(batch_features)

        features = torch.cat(features, dim=0)

        mu = features.mean(0)
        sigma = features.T.cov()

        return mu, sigma

    def calc(self, real_images: torch.Tensor, fake_images: torch.Tensor):
        r_mu, r_sigma = self._calc_features(real_images)
        f_mu, f_sigma = self._calc_features(fake_images)

        cov_prod = linalg.sqrtm((r_sigma @ f_sigma).cpu().numpy())
        cov_prod = torch.tensor(cov_prod, dtype=torch.float32, device=self.meta.device)

        fid_score = torch.norm(r_mu - f_mu) ** 2 + torch.trace(
            r_sigma + f_sigma - 2 * cov_prod
        )

        return fid_score
