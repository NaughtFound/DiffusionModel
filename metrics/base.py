from abc import ABC, abstractmethod
from typing import Any, Optional, get_type_hints

import torch


class MetricMeta:
    device: torch.device

    @classmethod
    def from_kwargs(cls, **kwargs):
        meta = cls()

        meta_kwargs = get_type_hints(cls).keys()

        for name, value in kwargs.items():
            if name in meta_kwargs:
                setattr(meta, name, value)

        return meta


class Metric(ABC):
    def __init__(self, meta: MetricMeta):
        super().__init__()

        self.meta = meta

    def _unwind_batch(self, batch: Any) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        labels = None
        if isinstance(batch, list):
            labels = batch[1]
            batch = batch[0]

        real_images = batch.to(self.meta.device)

        if labels is not None:
            labels = labels.to(self.meta.device)

        return real_images, labels

    @abstractmethod
    def calc(self, *args: Any, **kwargs: Any) -> Any:
        pass
