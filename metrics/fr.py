from typing import Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .base import Metric, MetricMeta


class FRMeta(MetricMeta):
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]]
    forward_method: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
    classifier: nn.Module

    def __init__(self):
        super().__init__()

        self.transform = None


class FR(Metric):
    def __init__(self, meta: FRMeta):
        super().__init__(meta)

        self.meta = meta

    @torch.no_grad()
    def calc(self, dataloader: DataLoader):
        scores = []

        for batch in tqdm(dataloader, desc="Calculating"):
            real_images, labels = self._unwind_batch(batch)

            fake_images = self.meta.forward_method(real_images, labels)

            if self.meta.transform is not None:
                real_images = self.meta.transform(real_images)
                fake_images = self.meta.transform(fake_images)

            rc = self.meta.classifier(real_images).argmax(dim=1)
            fc = self.meta.classifier(fake_images).argmax(dim=1)

            scores.append((rc != fc).to(torch.float))

        scores = torch.cat(scores)

        return torch.mean(scores)
