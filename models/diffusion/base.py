from abc import ABC, abstractmethod
from typing import Any

import torch


class Diffusion(ABC):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_noise(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass
