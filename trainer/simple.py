import os
from abc import abstractmethod
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import logging
from tqdm import tqdm
from .base import Trainer


class SimpleTrainer(Trainer):
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

    def __init__(self):
        super().__init__()

        self.args = self.create_default_args()

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s: %(message)s",
            level=logging.INFO,
            datefmt="%I:%M:%S",
        )

    def launch(self):
        parser = self.get_arg_parser()
        self.args = parser.parse_args()

        self.train()

    def train(self):
        args = self.args

        dataloader = self.create_dataloader()

        model, optimizer, last_epoch = self.load_last_checkpoint()

        model.train()

        self.pre_train(dataloader=dataloader, model=model)

        logger = SummaryWriter(os.path.join(args.prefix, "runs", args.run_name))

        len_data = len(dataloader)

        for epoch in range(last_epoch, args.epochs):
            logging.info(f"Starting epoch {epoch+1}")

            for i, batch in enumerate(
                tqdm(
                    dataloader,
                    desc=f"Training [{epoch + 1}/{args.epochs}]",
                )
            ):
                loss = self.train_step(model=model, batch=batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logger.add_scalar("Loss", loss.item(), global_step=epoch * len_data + i)

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
