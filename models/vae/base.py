from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class VAE(nn.Module, ABC):
    @abstractmethod
    def encode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def calc_loss(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass
