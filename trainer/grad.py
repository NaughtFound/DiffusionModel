import os
from abc import abstractmethod
from typing import Any
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import logging
from tqdm import tqdm
from utils import setup_logging
from utils.loader import ConfigKey, DatasetLoader
from .base import Trainer


class GradientTrainer(Trainer):
    @abstractmethod
    def load_last_checkpoint(self) -> tuple[nn.Module, optim.Optimizer, int]:
        pass

    @abstractmethod
    def pre_train(self, dataloader: DataLoader, model: nn.Module):
        pass

    @abstractmethod
    def train_step(self, model: nn.Module, batch: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def save_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        batch: Any,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def pre_inference(self, model: nn.Module):
        pass

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

    def train(self):
        args = self.args

        dataloader_class = self.create_dataloader(args.loader, args)
        dataloaders = dataloader_class.create_dataloaders()

        train_dataloader = dataloaders[ConfigKey.train]
        valid_dataloader = dataloaders.get(ConfigKey.valid)
        test_dataloader = dataloaders.get(ConfigKey.test)

        model, optimizer, last_epoch = self.load_last_checkpoint()

        model.train()

        self.pre_train(dataloader=train_dataloader, model=model)

        logger = SummaryWriter(os.path.join(args.prefix, "runs", args.run_name))

        len_train_data = len(train_dataloader)

        for epoch in range(last_epoch + 1, args.epochs):
            logging.info(f"Starting epoch {epoch+1}")

            for i, batch in enumerate(
                tqdm(
                    train_dataloader,
                    desc=f"Training [{epoch + 1}/{args.epochs}]",
                )
            ):
                loss = self.train_step(model=model, batch=batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logger.add_scalar(
                    "Loss",
                    loss.item(),
                    global_step=epoch * len_train_data + i,
                )

            if (epoch + 1) % args.save_freq == 0:
                self.save_step(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    batch=batch,
                )

        self.post_train()

    @classmethod
    def create_for_inference(cls, **kwargs):
        trainer = cls()
        args = trainer.args

        for k in kwargs:
            setattr(args, k, kwargs[k])

        model = trainer.load_last_checkpoint()[0]
        model.eval()

        trainer.pre_inference(model=model)

        return trainer

    def create_default_args(self):
        args = argparse.Namespace()
        args.prefix = "."
        args.run_name = ""
        args.model_type = ""
        args.epochs = 500
        args.lr = 3e-4
        args.device = "cuda"
        args.checkpoint = None
        args.save_freq = 5

        return args

    def get_arg_parser(self):
        parser = argparse.ArgumentParser()

        d_args = self.create_default_args()

        parser.add_argument("--prefix", type=str, default=d_args.prefix)
        parser.add_argument("--run_name", type=str, default=d_args.run_name)
        parser.add_argument("--model_type", type=str, default=d_args.model_type)
        parser.add_argument("--epochs", type=int, default=d_args.epochs)
        parser.add_argument("--lr", type=float, default=d_args.lr)
        parser.add_argument("--device", type=str, default=d_args.device)
        parser.add_argument("--checkpoint", type=str, default=d_args.checkpoint)
        parser.add_argument("--save_freq", type=int, default=d_args.save_freq)
        parser.add_argument("--loader", type=DatasetLoader, required=True)

        return parser
