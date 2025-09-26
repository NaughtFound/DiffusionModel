from typing import Any, Optional
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import logging
import utils
from models.unet.base import UNet
from models.diffusion.base import Diffusion
from models.diffusion.ddpm import DDPM, DDPM_Params
from trainer.grad import GradientTrainer, GradientTrainerState


class DDPMTrainer(GradientTrainer):
    def create_diffusion_model(self, eps_theta: nn.Module) -> Diffusion:
        args = self.args

        if args.model_type == "sde":
            params = DDPM_Params(args.device)
            params.eps_theta = eps_theta
            params.beta_min = args.beta_min
            params.beta_max = args.beta_max
            params.input_size = (args.in_channels, args.img_size, args.img_size)

            return DDPM(params)

    def create_model(self):
        return UNet.from_params(self.args)

    def load_last_checkpoint(self):
        args = self.args

        eps_theta = self.create_model().to(args.device)

        optimizer = optim.AdamW(eps_theta.parameters(), lr=args.lr)

        last_epoch = -1
        run_id = None

        if hasattr(args, "checkpoint") and args.checkpoint is not None:
            logging.info(f"Loading checkpoint {args.checkpoint}")
            last_epoch, run_id = utils.load_state_dict(
                eps_theta,
                optimizer,
                args.prefix,
                args.run_name,
                args.checkpoint,
                args.device,
            )

            eps_theta.to(args.device)

        return GradientTrainerState(
            model=eps_theta,
            optimizer=optimizer,
            epoch=last_epoch,
            run_id=run_id,
        )

    def pre_train(self, model: nn.Module, **kwargs):
        self.diffusion = self.create_diffusion_model(model)
        self.diffusion.train()

    def calc_loss(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        optimizer_dict: dict[str, optim.Optimizer],
        step_optimizer: bool = True,
        desc: Optional[str] = None,
    ):
        if not step_optimizer:
            self.diffusion.eval()

        loss = super().calc_loss(
            dataloader,
            model,
            optimizer_dict,
            step_optimizer,
            desc,
        )

        if not step_optimizer:
            self.diffusion.train()

        return loss

    def train_step(self, batch: Any, **kwargs) -> torch.Tensor:
        device = self.args.device

        images = batch[0].to(device)
        t = self.diffusion.t(images.shape[0])

        loss = self.diffusion.calc_loss(images, t)

        return loss

    def save_step(self, state: GradientTrainerState, batch: Any, **kwargs):
        args = self.args
        n = len(batch[0])

        logging.info(f"Sampling for epoch {state.epoch + 1}")
        self.diffusion.eval()
        sampled_images = self.diffusion.sample(n=n)
        self.diffusion.train()
        logging.info(f"Saving results for epoch {state.epoch + 1}")
        utils.save_images(
            sampled_images,
            args.prefix,
            args.run_name,
            f"{state.epoch + 1}.jpg",
        )
        utils.save_state_dict(
            model=state.model,
            optimizer=state.optimizer,
            epoch=state.epoch,
            prefix=args.prefix,
            run_name=args.run_name,
            file_name=f"ckpt-{state.epoch + 1}.pt",
            run_id=state.run_id,
        )

    def pre_inference(self, model: nn.Module, **kwargs):
        self.pre_train(model=model, **kwargs)

        self.diffusion.eval()

    @staticmethod
    def create_default_args():
        args = super(DDPMTrainer, DDPMTrainer).create_default_args()

        args.run_name = "DDPM_unconditional"
        args.model_type = "default"
        args.img_size = 64
        args.in_channels = 3
        args.T = 1000
        args.beta_min = 1e-4
        args.beta_max = 2e-2

        return args

    @staticmethod
    def get_arg_parser():
        parser = super(DDPMTrainer, DDPMTrainer).get_arg_parser()

        d_args = DDPMTrainer.create_default_args()

        parser.add_argument("--img_size", type=int, default=d_args.img_size)
        parser.add_argument("--in_channels", type=int, default=d_args.in_channels)
        parser.add_argument("--T", type=int, default=d_args.T)
        parser.add_argument("--beta_min", type=float, default=d_args.beta_min)
        parser.add_argument("--beta_max", type=float, default=d_args.beta_max)

        return parser
