from typing import Any
import argparse
import numpy as np
import torch
from torch import optim, nn
from argparse import Namespace
import logging
import utils
from models.unet.label_conditioned import LabelConditionedUNet
from models.diffusion.cfg import Diffusion_CFG
from models.diffusion.base import Diffusion
from models.sde.cfg import SDE_CFG_Params, SDE_CFG
from trainer.models.ddpm import DDPMTrainer
from utils.loader import DatasetLoader


class CFGTrainer(DDPMTrainer):
    def create_diffusion_model(self, eps_theta: nn.Module) -> Diffusion:
        args = self.args

        if args.model_type == "default":
            return Diffusion_CFG(
                noise_predictor=eps_theta,
                T=args.T,
                beta_start=args.beta_start,
                beta_end=args.beta_end,
                img_size=args.img_size,
                in_channels=args.in_channels,
                device=args.device,
            )

        if args.model_type == "sde":
            params = SDE_CFG_Params(args.device)
            params.eps_theta = eps_theta
            params.beta_start = args.beta_start
            params.beta_end = args.beta_end
            params.input_size = (args.in_channels, args.img_size, args.img_size)

            return SDE_CFG(params)

    def create_model(self):
        args = self.args

        return LabelConditionedUNet(
            in_channels=args.in_channels,
            out_channels=args.in_channels,
            num_classes=args.num_classes,
        )

    def train_step(self, batch: Any, **kwargs) -> torch.Tensor:
        args = self.args

        images = batch[0].to(args.device)
        labels = batch[1].to(args.device)
        t = self.diffusion.t(images.shape[0])

        if np.random.random() < args.alpha:
            labels = None

        loss = self.diffusion.calc_loss(images, t, labels, args.cfg_scale)

        return loss

    def save_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        **kwargs,
    ):
        args = self.args

        logging.info(f"Sampling for epoch {epoch+1}")
        self.diffusion.eval()
        labels = torch.arange(args.num_classes).long().to(args.device)
        sampled_images = self.diffusion.sample(
            n=args.num_classes,
            labels=labels,
            cfg_scale=args.cfg_scale,
        )
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

    def create_default_args(self):
        args = Namespace()
        args.prefix = "."
        args.run_name = "CFG"
        args.model_type = "default"
        args.epochs = 500
        args.img_size = 64
        args.in_channels = 3
        args.T = 1000
        args.beta_start = 1e-4
        args.beta_end = 2e-2
        args.device = "cuda"
        args.lr = 3e-4
        args.alpha = 0.1
        args.cfg_scale = 0.1
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
        parser.add_argument("--img_size", type=int, default=d_args.img_size)
        parser.add_argument("--in_channels", type=int, default=d_args.in_channels)
        parser.add_argument("--T", type=int, default=d_args.T)
        parser.add_argument("--beta_start", type=float, default=d_args.beta_start)
        parser.add_argument("--beta_end", type=float, default=d_args.beta_end)
        parser.add_argument("--loader", type=DatasetLoader, required=True)
        parser.add_argument("--device", type=str, default=d_args.device)
        parser.add_argument("--lr", type=float, default=d_args.lr)
        parser.add_argument("--num_classes", type=int, required=True)
        parser.add_argument("--alpha", type=float, default=d_args.alpha)
        parser.add_argument("--cfg_scale", type=float, default=d_args.cfg_scale)
        parser.add_argument("--checkpoint", type=str, default=d_args.checkpoint)
        parser.add_argument("--save_freq", type=int, default=d_args.save_freq)

        return parser
