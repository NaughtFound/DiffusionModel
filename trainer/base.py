from abc import ABC, abstractmethod
from typing import Any
from argparse import Namespace, ArgumentParser
from utils.loader import DatasetLoader


class Trainer(ABC):
    @abstractmethod
    def create_model(self) -> Any:
        pass

    @abstractmethod
    def load_last_checkpoint(self) -> Any:
        pass

    def create_dataloader(self, loader: Any, args: Namespace) -> DatasetLoader:
        try:
            instance = loader(args)
            if not isinstance(instance, DatasetLoader):
                raise TypeError(
                    f"Created instance must be a DatasetLoader, got {type(instance)}"
                )
            return instance
        except Exception as e:
            raise RuntimeError(f"Failed to create loader instance: {str(e)}") from e

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
    def pre_inference(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def get_arg_parser(self) -> ArgumentParser:
        pass

    @abstractmethod
    def create_default_args(self) -> Namespace:
        pass
