import os
from abc import abstractmethod
from typing import Any, Optional, Union
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import logging
from tqdm import tqdm
from utils.loader import ConfigKey, DatasetLoader
from .base import Trainer


class GradientTrainer(Trainer):
    @abstractmethod
    def load_last_checkpoint(
        self,
    ) -> tuple[
        nn.Module,
        Union[optim.Optimizer, dict[str, optim.Optimizer]],
        int,
    ]:
        pass

    @abstractmethod
    def pre_train(self, dataloader: DataLoader, model: nn.Module):
        pass

    @abstractmethod
    def train_step(self, model: nn.Module, batch: Any, optim_name: str) -> torch.Tensor:
        pass

    @abstractmethod
    def save_step(
        self,
        model: nn.Module,
        optimizer: Union[optim.Optimizer, dict[str, optim.Optimizer]],
        epoch: int,
        batch: Any,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def pre_inference(self, model: nn.Module):
        pass

    def calc_loss(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        optimizer_dict: dict[str, optim.Optimizer],
        step_optimizer: bool = True,
        desc: Optional[str] = None,
    ) -> dict[str, float]:
        loss_dict = {}

        for batch in tqdm(dataloader, desc=desc):
            for optim_name, optimizer in optimizer_dict.items():
                loss = self.train_step(
                    model=model,
                    batch=batch,
                    optim_name=optim_name,
                )

                if optim_name not in loss_dict:
                    loss_dict[optim_name] = 0

                loss_dict[optim_name] += loss.item()

                if step_optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        return loss_dict

    def train(self):
        args = self.args

        dataloader_class = self.create_dataloader(args.loader, args)
        dataloaders = dataloader_class.create_dataloaders()

        train_dataloader = dataloaders[ConfigKey.train]
        valid_dataloader = dataloaders.get(ConfigKey.valid)
        test_dataloader = dataloaders.get(ConfigKey.test)

        model, optimizer, last_epoch = self.load_last_checkpoint()

        if isinstance(optimizer, optim.Optimizer):
            optimizer_dict = {
                optimizer.__class__.__name__: optimizer,
            }
        else:
            optimizer_dict = optimizer

        model.train()

        self.pre_train(dataloader=train_dataloader, model=model)

        train_logger = SummaryWriter(
            os.path.join(args.prefix, "runs_train", args.run_name)
        )

        len_train_data = len(train_dataloader)

        if valid_dataloader is not None:
            len_valid_data = len(valid_dataloader)
            valid_logger = SummaryWriter(
                os.path.join(args.prefix, "runs_valid", args.run_name)
            )

        for epoch in range(last_epoch + 1, args.epochs):
            logging.info(f"Starting epoch {epoch + 1}")

            loss_dict = self.calc_loss(
                dataloader=train_dataloader,
                model=model,
                optimizer_dict=optimizer_dict,
                step_optimizer=True,
                desc=f"Training [{epoch + 1}/{self.args.epochs}]",
            )

            for loss_name, loss in loss_dict.items():
                train_logger.add_scalar(
                    f"Loss {loss_name}",
                    loss / len_train_data,
                    global_step=epoch,
                )

            if (epoch + 1) % args.save_freq == 0:
                self.save_step(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    batch=next(iter(train_dataloader)),
                )

            if valid_dataloader is None:
                continue

            with torch.no_grad():
                model.eval()
                loss_dict = self.calc_loss(
                    dataloader=valid_dataloader,
                    model=model,
                    optimizer_dict=optimizer_dict,
                    step_optimizer=False,
                    desc=f"Validation [{epoch + 1}/{args.epochs}]",
                )
                model.train()

                for loss_name, loss in loss_dict.items():
                    valid_logger.add_scalar(
                        f"Loss {loss_name}",
                        loss / len_valid_data,
                        global_step=epoch,
                    )

        if test_dataloader is not None:
            len_test_data = len(test_dataloader)

            with torch.no_grad():
                model.eval()
                loss_dict = self.calc_loss(
                    dataloader=test_dataloader,
                    model=model,
                    optimizer_dict=optimizer_dict,
                    step_optimizer=False,
                    desc="Testing",
                )
                model.train()

                for loss_name, loss in loss_dict.items():
                    mean_test_loss = loss / len_test_data
                    logging.info(f"Test Mean Loss for {loss_name}: {mean_test_loss}")

        self.post_train()

    @classmethod
    def create_for_inference(cls, **kwargs):
        trainer = cls(**kwargs)

        model = trainer.load_last_checkpoint()[0]
        model.eval()

        trainer.pre_inference(model=model)

        return trainer

    @staticmethod
    def create_default_args():
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

    @staticmethod
    def get_arg_parser():
        parser = argparse.ArgumentParser()

        d_args = GradientTrainer.create_default_args()

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
