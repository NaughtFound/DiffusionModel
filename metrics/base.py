from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class Metric(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model

    @abstractmethod
    def calc(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass
