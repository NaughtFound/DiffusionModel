import argparse
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import mlflow
import torch
from torch import nn, optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.loader import ConfigKey

from .base import Trainer


@dataclass
class GradientTrainerState[M: nn.Module]:
    model: M
    optimizer: optim.Optimizer | dict[str, optim.Optimizer]
    epoch: int
    run_id: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def set_param(self, key: str, value: Any) -> None:
        self.kwargs[key] = value

    def get_param(self, key: str) -> Any | None:
        return self.kwargs.get(key)


class GradientTrainer(Trainer):
    @abstractmethod
    def load_last_checkpoint(self) -> GradientTrainerState:
        pass

    def pre_train(self, state: GradientTrainerState) -> None:
        pass

    @abstractmethod
    def train_step(self, *, model: nn.Module, batch: Any, optim_name: str) -> torch.Tensor:
        pass

    @abstractmethod
    def save_step(self, state: GradientTrainerState) -> None:
        pass

    @abstractmethod
    def pre_inference(self, state: GradientTrainerState) -> None:
        pass

    def calc_loss(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        optimizer_dict: dict[str, optim.Optimizer],
        desc: str | None = None,
        *,
        step_optimizer: bool = True,
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

    def _get_optimizer_dict(
        self,
        optimizer: optim.Optimizer | dict[str, optim.Optimizer],
    ) -> dict[str, optim.Optimizer]:
        if isinstance(optimizer, optim.Optimizer):
            return {
                optimizer.__class__.__name__: optimizer,
            }

        return optimizer

    def _setup_ema(self, model: nn.Module) -> AveragedModel | None:
        if not self.args.use_ema:
            return None

        ema = AveragedModel(
            model=model,
            multi_avg_fn=get_ema_multi_avg_fn(self.args.ema_decay),
            device=self.args.device,
            use_buffers=True,
        )
        ema.eval()

        return ema

    def _update_ema(self, epoch: int, state: GradientTrainerState) -> None:
        ema = state.get_param("ema")
        if (
            isinstance(ema, AveragedModel)
            and epoch >= self.args.ema_start
            and epoch % self.args.ema_update_freq == 0
        ):
            logging.info("Updating EMA model")
            ema.update_parameters(state.model)

    def _save_checkpoint(
        self,
        epoch: int,
        state: GradientTrainerState,
        active_run: mlflow.ActiveRun,
    ) -> None:
        train_dataloader = state.get_param("train_dataloader")
        if not isinstance(train_dataloader, DataLoader):
            return

        ema = state.get_param("ema")

        if (epoch + 1) % self.args.save_freq == 0:
            save_model = state.model
            if isinstance(ema, AveragedModel):
                save_model = ema.module

            save_state = GradientTrainerState(
                model=save_model,
                optimizer=state.optimizer,
                epoch=epoch,
                run_id=active_run.info.run_id,
                kwargs={**state.kwargs},
            )

            save_state.set_param("batch", next(iter(train_dataloader)))

            self.save_step(save_state)

    def _run_training(
        self,
        epoch: int,
        state: GradientTrainerState,
        active_run: mlflow.ActiveRun,
    ) -> None:
        train_dataloader = state.get_param("train_dataloader")

        if not isinstance(train_dataloader, DataLoader):
            return

        len_train_data = len(train_dataloader)
        optimizer_dict = self._get_optimizer_dict(state.optimizer)

        loss_dict = self.calc_loss(
            dataloader=train_dataloader,
            model=state.model,
            optimizer_dict=optimizer_dict,
            step_optimizer=True,
            desc=f"Training [{epoch + 1}/{self.args.epochs}]",
        )

        self._update_ema(epoch=epoch, state=state)

        for loss_name, loss in loss_dict.items():
            mlflow.log_metric(
                key=f"train/{loss_name}",
                value=loss / len_train_data,
                step=epoch,
            )

        self._save_checkpoint(epoch=epoch, state=state, active_run=active_run)

    @torch.no_grad()
    def _run_validation(self, epoch: int, state: GradientTrainerState) -> None:
        valid_dataloader = state.get_param("valid_dataloader")
        if valid_dataloader is None:
            return

        len_valid_data = len(valid_dataloader)
        ema = state.get_param("ema")
        optimizer_dict = self._get_optimizer_dict(state.optimizer)

        state.model.eval()

        valid_model = state.model
        if isinstance(ema, AveragedModel):
            valid_model = ema.module

        loss_dict = self.calc_loss(
            dataloader=valid_dataloader,
            model=valid_model,
            optimizer_dict=optimizer_dict,
            step_optimizer=False,
            desc=f"Validation [{epoch + 1}/{self.args.epochs}]",
        )
        state.model.train()

        for loss_name, loss in loss_dict.items():
            mlflow.log_metric(
                key=f"valid/{loss_name}",
                value=loss / len_valid_data,
                step=epoch,
            )

    @torch.no_grad()
    def _run_testing(self, state: GradientTrainerState) -> None:
        test_dataloader = state.get_param("test_dataloader")
        if not isinstance(test_dataloader, DataLoader):
            return

        len_test_data = len(test_dataloader)
        ema = state.get_param("ema")
        optimizer_dict = self._get_optimizer_dict(state.optimizer)

        state.model.eval()

        test_model = state.model
        if isinstance(ema, AveragedModel):
            test_model = ema.module

        loss_dict = self.calc_loss(
            dataloader=test_dataloader,
            model=test_model,
            optimizer_dict=optimizer_dict,
            step_optimizer=False,
            desc="Testing",
        )
        state.model.train()

        for loss_name, loss in loss_dict.items():
            mean_test_loss = loss / len_test_data
            logging.info(f"Test Mean Loss for {loss_name}: {mean_test_loss}")

    def _finalize_train(
        self,
        state: GradientTrainerState,
        active_run: mlflow.ActiveRun,
    ) -> None:
        final_model = state.model
        ema = state.get_param("ema")

        if isinstance(ema, AveragedModel):
            final_model = ema.module

        final_state = GradientTrainerState(
            model=final_model,
            optimizer=state.optimizer,
            epoch=self.args.epochs - 1,
            run_id=active_run.info.run_id,
            kwargs={**state.kwargs},
        )

        self.post_train(final_state)

    def train(self) -> None:
        args = self.args

        mlflow.set_tracking_uri(Path(args.prefix) / "runs" / args.run_name)

        dataloader_class = self.create_dataloader(args.loader, args)
        dataloaders = dataloader_class.create_dataloaders()

        last_state: GradientTrainerState[nn.Module] = self.load_last_checkpoint()

        model = last_state.model
        last_epoch = last_state.epoch
        run_id = last_state.run_id

        ema = self._setup_ema(model)

        last_state.set_param("train_dataloader", dataloaders[ConfigKey.train])
        last_state.set_param("valid_dataloader", dataloaders.get(ConfigKey.valid))
        last_state.set_param("test_dataloader", dataloaders.get(ConfigKey.test))
        last_state.set_param("ema", ema)

        model.train()

        self.pre_train(state=last_state)

        with mlflow.start_run(run_name=args.run_name, run_id=run_id) as active_run:
            for epoch in range(last_epoch + 1, args.epochs):
                logging.info(f"Starting epoch {epoch + 1}")

                self._run_training(epoch=epoch, state=last_state, active_run=active_run)
                self._run_validation(epoch=epoch, state=last_state)

            self._run_testing(state=last_state)
            self._finalize_train(state=last_state, active_run=active_run)

    def post_train(self, state: GradientTrainerState) -> None:
        pass

    @classmethod
    def create_for_inference(cls, **kwargs) -> Self:
        trainer = cls(**kwargs)

        state = trainer.load_last_checkpoint()
        model = state.model
        run_id = state.run_id

        model.eval()

        trainer.pre_inference(state)

        trainer.args.run_id = run_id

        return trainer

    @staticmethod
    def create_default_args() -> argparse.Namespace:
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
    def get_arg_parser() -> argparse.ArgumentParser:
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
        parser.add_argument("--loader", required=True)

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
