from abc import ABC, abstractmethod
from typing import Any
from argparse import Namespace, ArgumentParser
import logging
from utils import setup_logging
from utils.loader import DatasetLoader


class Trainer(ABC):
    def __init__(self, **kwargs):
        super().__init__()

        self.args = self.create_default_args()

        for k in kwargs:
            setattr(self.args, k, kwargs[k])

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s: %(message)s",
            level=logging.INFO,
            datefmt="%I:%M:%S",
        )

        setup_logging(self.args.run_name, self.args.prefix)

    def launch(self):
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

    @staticmethod
    @abstractmethod
    def get_arg_parser() -> ArgumentParser:
        pass

    @staticmethod
    @abstractmethod
    def create_default_args() -> Namespace:
        pass
