import os
from dataclasses import dataclass
from abc import abstractmethod
from typing import Any, Optional, Union
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
import logging
import mlflow
from tqdm import tqdm
from utils.loader import ConfigKey, DatasetLoader
from .base import Trainer


@dataclass(frozen=True)
class GradientTrainerState:
    model: nn.Module
    optimizer: Union[optim.Optimizer, dict[str, optim.Optimizer]]
    epoch: int
    run_id: Optional[str] = None


class GradientTrainer(Trainer):
    @abstractmethod
    def load_last_checkpoint(self) -> GradientTrainerState:
        pass

    @abstractmethod
    def pre_train(self, dataloader: DataLoader, model: nn.Module):
        pass

    @abstractmethod
    def train_step(self, model: nn.Module, batch: Any, optim_name: str) -> torch.Tensor:
        pass

    @abstractmethod
    def save_step(self, state: GradientTrainerState, batch: Any) -> torch.Tensor:
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

        mlflow.set_tracking_uri(os.path.join(args.prefix, "runs", args.run_name))

        dataloader_class = self.create_dataloader(args.loader, args)
        dataloaders = dataloader_class.create_dataloaders()

        train_dataloader = dataloaders[ConfigKey.train]
        valid_dataloader = dataloaders.get(ConfigKey.valid)
        test_dataloader = dataloaders.get(ConfigKey.test)

        last_state = self.load_last_checkpoint()

        model = last_state.model
        optimizer = last_state.optimizer
        last_epoch = last_state.epoch
        run_id = last_state.run_id

        if isinstance(optimizer, optim.Optimizer):
            optimizer_dict = {
                optimizer.__class__.__name__: optimizer,
            }
        else:
            optimizer_dict = optimizer

        model.train()

        self.pre_train(dataloader=train_dataloader, model=model)

        ema = None

        if args.use_ema:
            ema = AveragedModel(
                model=model,
                multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay),
                device=args.device,
                use_buffers=True,
            )
            ema.eval()

        len_train_data = len(train_dataloader)

        if valid_dataloader is not None:
            len_valid_data = len(valid_dataloader)

        active_run = mlflow.start_run(run_name=args.run_name, run_id=run_id)

        for epoch in range(last_epoch + 1, args.epochs):
            logging.info(f"Starting epoch {epoch + 1}")

            loss_dict = self.calc_loss(
                dataloader=train_dataloader,
                model=model,
                optimizer_dict=optimizer_dict,
                step_optimizer=True,
                desc=f"Training [{epoch + 1}/{self.args.epochs}]",
            )

            if args.use_ema and isinstance(ema, AveragedModel):
                if epoch >= args.ema_start and epoch % args.ema_update_freq == 0:
                    logging.info("Updating EMA model")
                    ema.update_parameters(model)

            for loss_name, loss in loss_dict.items():
                mlflow.log_metric(
                    key=f"train/{loss_name}",
                    value=loss / len_train_data,
                    step=epoch,
                )

            if (epoch + 1) % args.save_freq == 0:
                save_model = model
                if args.use_ema and isinstance(ema, AveragedModel):
                    save_model = ema.module

                self.save_step(
                    state=GradientTrainerState(
                        model=save_model,
                        optimizer=optimizer,
                        epoch=epoch,
                        run_id=active_run.info.run_id,
                    ),
                    batch=next(iter(train_dataloader)),
                )

            if valid_dataloader is None:
                continue

            with torch.no_grad():
                model.eval()

                valid_model = model
                if args.use_ema and isinstance(ema, AveragedModel):
                    valid_model = ema.module

                loss_dict = self.calc_loss(
                    dataloader=valid_dataloader,
                    model=valid_model,
                    optimizer_dict=optimizer_dict,
                    step_optimizer=False,
                    desc=f"Validation [{epoch + 1}/{args.epochs}]",
                )
                model.train()

                for loss_name, loss in loss_dict.items():
                    mlflow.log_metric(
                        key=f"valid/{loss_name}",
                        value=loss / len_valid_data,
                        step=epoch,
                    )

        if test_dataloader is not None:
            len_test_data = len(test_dataloader)

            with torch.no_grad():
                model.eval()

                test_model = model
                if args.use_ema and isinstance(ema, AveragedModel):
                    test_model = ema.module

                loss_dict = self.calc_loss(
                    dataloader=test_dataloader,
                    model=test_model,
                    optimizer_dict=optimizer_dict,
                    step_optimizer=False,
                    desc="Testing",
                )
                model.train()

                for loss_name, loss in loss_dict.items():
                    mean_test_loss = loss / len_test_data
                    logging.info(f"Test Mean Loss for {loss_name}: {mean_test_loss}")

        self.post_train()

        mlflow.end_run()

    def post_train(self):
        pass

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

        args.use_ema = True
        args.ema_decay = 0.99
        args.ema_start = 15
        args.ema_update_freq = 1

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

        parser.add_argument("--use_ema", type=bool, default=d_args.use_ema)
        parser.add_argument("--ema_decay", type=float, default=d_args.ema_decay)
        parser.add_argument(
            "--ema_start",
            type=int,
            default=d_args.ema_start,
        )
        parser.add_argument(
            "--ema_update_freq",
            type=int,
            default=d_args.ema_update_freq,
        )

        return parser
