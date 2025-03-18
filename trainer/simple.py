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
    def pre_train(self, dataloader: DataLoader):
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

    def __init__(self):
        super().__init__()

        parser = self.get_arg_parser()
        self.args = parser.parse_args()

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s: %(message)s",
            level=logging.INFO,
            datefmt="%I:%M:%S",
        )

    def train(self):
        args = self.args

        dataloader = self.create_dataloader()

        model, optimizer, last_epoch = self.load_last_checkpoint()

        model.train()

        self.pre_train(dataloader=dataloader)

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
