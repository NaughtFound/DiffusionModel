from abc import ABC, abstractmethod
from typing import Any


class Diffusion(ABC):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Any:
        pass
