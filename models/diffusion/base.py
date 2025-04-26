from abc import ABC, abstractmethod
from typing import Any

import torch


class Diffusion(ABC):
    @abstractmethod
    def t(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def reverse(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_noise(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def calc_loss(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def train(self):
        pass
