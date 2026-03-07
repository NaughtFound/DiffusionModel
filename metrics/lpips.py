from collections.abc import Callable
from typing import Literal

import lpips
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import Metric, MetricMeta


class LPIPSMeta(MetricMeta):
    net_type: Literal["alex", "vgg"]
    transform: Callable[[torch.Tensor], torch.Tensor] | None
    forward_method: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]

    def __init__(self) -> None:
        super().__init__()

        self.net_type = "alex"
        self.transform = None


class LPIPS(Metric):
    def __init__(self, meta: LPIPSMeta) -> None:
        super().__init__(meta)

        self.meta = meta

        self.model = lpips.LPIPS(net=self.meta.net_type, spatial=False)
        self.model.eval()
        self.model.to(self.meta.device)

    @torch.no_grad()
    def calc(self, dataloader: DataLoader) -> torch.Tensor:
        scores = []

        for batch in tqdm(dataloader, desc="Calculating"):
            real_images, labels = self._unwind_batch(batch)

            fake_images = self.meta.forward_method(real_images, labels)

            if self.meta.transform is not None:
                real_images = self.meta.transform(real_images)
                fake_images = self.meta.transform(fake_images)

            score = self.model(real_images, fake_images, normalize=False).squeeze()
            scores.append(score.mean())

        scores = torch.tensor(scores, device=self.meta.device)

        return torch.mean(scores)
