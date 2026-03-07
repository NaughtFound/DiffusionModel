from collections.abc import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import Metric, MetricMeta


class S3Meta(MetricMeta):
    transform: Callable[[torch.Tensor], torch.Tensor] | None
    forward_method: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]
    dist_extractor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __init__(self) -> None:
        super().__init__()

        self.transform = None


class S3(Metric):
    def __init__(self, meta: S3Meta) -> None:
        super().__init__(meta)

        self.meta = meta

    @torch.no_grad()
    def calc(self, dataloader: DataLoader) -> torch.Tensor:
        dists = []

        for batch in tqdm(dataloader, desc="Calculating"):
            real_images, labels = self._unwind_batch(batch)

            fake_images = self.meta.forward_method(real_images, labels)

            if self.meta.transform is not None:
                real_images = self.meta.transform(real_images)
                fake_images = self.meta.transform(fake_images)

            dist = self.meta.dist_extractor(real_images, fake_images)
            dists.append(dist)

        dists = torch.cat(dists)

        return torch.mean(dists)
