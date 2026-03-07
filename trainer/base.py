import logging
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Any

from utils import setup_logging
from utils.loader import DatasetLoader


class Trainer(ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.args = self.create_default_args()

        for k, v in kwargs.items():
            setattr(self.args, k, v)

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s: %(message)s",
            level=logging.INFO,
            datefmt="%I:%M:%S",
        )

        setup_logging(self.args.run_name, self.args.prefix)

    def launch(self) -> None:
        parser = self.get_arg_parser()
        self.args = parser.parse_args()

        self.train()

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
                msg = f"Created instance must be a DatasetLoader, got {type(instance)}"
                raise TypeError(msg)

        except Exception as e:
            msg = f"Failed to create loader instance: {e!s}"
            raise RuntimeError(msg) from e
        else:
            return instance

    @abstractmethod
    def pre_train(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def train(self) -> Any:
        pass

    @abstractmethod
    def train_step(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def save_step(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def post_train(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def pre_inference(self, *args: Any, **kwargs: Any) -> None:
        pass

    @staticmethod
    @abstractmethod
    def get_arg_parser() -> ArgumentParser:
        pass

    @staticmethod
    @abstractmethod
    def create_default_args() -> Namespace:
        pass
