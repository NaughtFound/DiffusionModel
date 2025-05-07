from abc import ABC, abstractmethod
from typing import Any

import torch


class MetricMeta:
    device: torch.device


class Metric(ABC):
    def __init__(self, meta: MetricMeta):
        super().__init__()

        self.meta = meta

    @abstractmethod
    @torch.no_grad()
    def calc(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass
