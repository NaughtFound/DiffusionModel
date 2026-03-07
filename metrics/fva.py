from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import Metric, MetricMeta


class FVAMeta(MetricMeta):
    transform: Callable[[torch.Tensor], torch.Tensor] | None
    forward_method: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]
    feature_extractor: nn.Module
    score_thresh: float | None

    def __init__(self) -> None:
        super().__init__()

        self.transform = None
        self.score_thresh = 0.5


class FVA(Metric):
    def __init__(self, meta: FVAMeta) -> None:
        super().__init__(meta)

        self.meta = meta

    @torch.no_grad()
    def calc(self, dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        scores = []
        dists = []

        cosine_similarity = nn.CosineSimilarity()

        for batch in tqdm(dataloader, desc="Calculating"):
            real_images, labels = self._unwind_batch(batch)

            fake_images = self.meta.forward_method(real_images, labels)

            if self.meta.transform is not None:
                real_images = self.meta.transform(real_images)
                fake_images = self.meta.transform(fake_images)

            rf = self.meta.feature_extractor(real_images)
            ff = self.meta.feature_extractor(fake_images)

            dist = cosine_similarity(rf, ff)
            dists.append(dist)
            scores.append((dist > self.meta.score_thresh).to(torch.float))

        dists = torch.cat(dists)
        scores = torch.cat(scores)

        return torch.mean(scores), torch.mean(dists)
