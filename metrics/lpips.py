from typing import Callable, Literal, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips
from .base import Metric, MetricMeta


class LPIPSMeta(MetricMeta):
    net_type: Optional[Literal["alex", "vgg"]]
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]]
    forward_method: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self):
        super().__init__()

        self.net_type = "alex"
        self.transform = None


class LPIPS(Metric):
    def __init__(self, meta: LPIPSMeta):
        super().__init__(meta)

        self.meta = meta

        self.model = lpips.LPIPS(net=self.meta.net_type, spatial=False)
        self.model.eval()
        self.model.to(self.meta.device)

    @torch.no_grad()
    def calc(self, dataloader: DataLoader):
        scores = []

        for batch in tqdm(dataloader, desc="Calculating"):
            if isinstance(batch, list):
                batch = batch[0]

            real_images = batch.to(self.meta.device)
            fake_images = self.meta.forward_method(real_images)

            if self.meta.transform is not None:
                real_images = self.meta.transform(real_images)
                fake_images = self.meta.transform(fake_images)

            score = self.model(real_images, fake_images, normalize=False).squeeze()
            scores.append(score.mean())

        scores = torch.tensor(scores, device=self.meta.device)

        return torch.mean(scores)
