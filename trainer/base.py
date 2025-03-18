from abc import ABC, abstractmethod
from typing import Any
from argparse import Namespace, ArgumentParser
from torch.utils.data import DataLoader


class Trainer(ABC):
    @abstractmethod
    def create_model(self) -> Any:
        pass

    @abstractmethod
    def load_last_checkpoint(self) -> Any:
        pass

    @abstractmethod
    def create_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def pre_train(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def train(self) -> Any:
        pass

    @abstractmethod
    def train_step(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def save_step(self):
        pass

    @abstractmethod
    def post_train(self):
        pass

    @abstractmethod
    def get_arg_parser(self) -> ArgumentParser:
        pass

    @abstractmethod
    def create_default_args(self) -> Namespace:
        pass
