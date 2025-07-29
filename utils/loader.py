from typing import Any
from enum import Enum
from abc import ABC, abstractmethod
from argparse import Namespace
from torch.utils.data import DataLoader, Dataset


class ConfigKey(Enum):
    train = 0
    valid = 1
    test = 2


class DataloaderConfig:
    dataset: Dataset
    kwargs: dict[str, Any]

    def __init__(self, dataset: Dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs


class DatasetLoader(ABC):
    def __init__(self, args: Namespace):
        super().__init__()

        self.args = args

    @classmethod
    def from_kwargs(cls, **kwargs):
        args = Namespace(**kwargs)

        loader = cls(args)

        return loader

    @abstractmethod
    def get_dataloader_configs(self) -> dict[ConfigKey, DataloaderConfig]:
        pass

    def create_dataloaders(self) -> dict[ConfigKey, DataLoader]:
        configs = self.get_dataloader_configs()

        dataloaders = {}

        for k in configs:
            config = configs[k]
            dataloaders[k] = DataLoader(dataset=config.dataset, **config.kwargs)

        return dataloaders
