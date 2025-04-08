from typing import Any
import argparse
import torch
from torch import optim, nn
from argparse import Namespace
import logging
import utils
from models.unet.base import UNet
from models.diffusion.base import Diffusion
from models.diffusion.ddpm import Diffusion_DDPM
from models.sde.ddpm import SDE_DDPM, SDE_DDPM_Params
from trainer.grad import GradientTrainer


class DDPMTrainer(GradientTrainer):
    def create_diffusion_model(self, eps_theta: nn.Module) -> Diffusion:
        args = self.args

        if args.model_type == "default":
            return Diffusion_DDPM(
                noise_predictor=eps_theta,
                T=args.T,
                beta_start=args.beta_start,
                beta_end=args.beta_end,
                img_size=args.img_size,
                in_channels=args.in_channels,
                device=args.device,
            )

        if args.model_type == "sde":
            params = SDE_DDPM_Params(args.device)
            params.eps_theta = eps_theta
            params.beta_start = args.beta_start
            params.beta_end = args.beta_end
            params.input_size = (args.in_channels, args.img_size, args.img_size)

            return SDE_DDPM(params)

    def create_model(self):
        args = self.args

        return UNet(
            in_channels=args.in_channels,
            out_channels=args.in_channels,
        )

    def load_last_checkpoint(self):
        args = self.args

        eps_theta = self.create_model().to(args.device)

        optimizer = optim.AdamW(eps_theta.parameters(), lr=args.lr)

        last_epoch = -1

        if hasattr(args, "checkpoint") and args.checkpoint is not None:
            logging.info(f"Loading checkpoint {args.checkpoint}")
            last_epoch = utils.load_state_dict(
                eps_theta,
                optimizer,
                args.prefix,
                args.run_name,
                args.checkpoint,
                args.device,
            )

            eps_theta.to(args.device)

        return eps_theta, optimizer, last_epoch

    def pre_train(self, model: nn.Module, **kwargs):
        self.diffusion = self.create_diffusion_model(model)
        self.diffusion.train()

    def train_step(self, batch: Any, **kwargs) -> torch.Tensor:
        device = self.args.device

        images = batch[0].to(device)
        t = self.diffusion.t(images.shape[0])

        loss = self.diffusion.calc_loss(images, t)

        return loss

    def save_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        batch: Any,
    ):
        args = self.args
        n = len(batch[0])

        logging.info(f"Sampling for epoch {epoch+1}")
        self.diffusion.eval()
        sampled_images = self.diffusion.sample(n=n)
        self.diffusion.train()
        logging.info(f"Saving results for epoch {epoch+1}")
        utils.save_images(
            sampled_images,
            args.prefix,
            args.run_name,
            f"{epoch+1}.jpg",
        )
        utils.save_state_dict(
            model,
            optimizer,
            epoch,
            args.prefix,
            args.run_name,
            f"ckpt-{epoch+1}.pt",
        )

    def post_train(self):
        pass

    def pre_inference(self, model: nn.Module, **kwargs):
        self.pre_train(model=model, **kwargs)

        self.diffusion.eval()

    def create_default_args(self):
        args = Namespace()
        args.prefix = "."
        args.run_name = "DDPM_unconditional"
        args.model_type = "default"
        args.epochs = 500
        args.batch_size = 12
        args.shuffle = True
        args.img_size = 64
        args.in_channels = 3
        args.T = 1000
        args.beta_start = 1e-4
        args.beta_end = 2e-2
        args.device = "cuda"
        args.lr = 3e-4
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
        parser.add_argument("--batch_size", type=int, default=d_args.batch_size)
        parser.add_argument("--shuffle", type=bool, default=d_args.shuffle)
        parser.add_argument("--img_size", type=int, default=d_args.img_size)
        parser.add_argument("--in_channels", type=int, default=d_args.in_channels)
        parser.add_argument("--T", type=int, default=d_args.T)
        parser.add_argument("--beta_start", type=float, default=d_args.beta_start)
        parser.add_argument("--beta_end", type=float, default=d_args.beta_end)
        parser.add_argument("--dataset_path", type=str, required=True)
        parser.add_argument("--device", type=str, default=d_args.device)
        parser.add_argument("--lr", type=float, default=d_args.lr)
        parser.add_argument("--checkpoint", type=str, default=d_args.checkpoint)
        parser.add_argument("--save_freq", type=int, default=d_args.save_freq)

        return parser
