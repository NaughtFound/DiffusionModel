from typing import Any
import argparse
import torch
from torch import optim, nn
from argparse import Namespace
import logging
import utils
from models.unet.conditional import ConditionalUNet
from models.diffusion.base import Diffusion
from models.vae.base import VAE
from models.sde.ldm import SDE_LDM_Params, SDE_LDM
from trainer.models.ddpm import DDPMTrainer
from . import vae


class LDMTrainer(DDPMTrainer):
    def create_diffusion_model(
        self,
        eps_theta: nn.Module,
        tau_theta: nn.Module,
    ) -> Diffusion:
        args = self.args

        if args.model_type == "default":
            raise NotImplementedError("default type is not implemented yet.")

        if args.model_type == "sde":
            params = SDE_LDM_Params(args.device)
            params.eps_theta = eps_theta
            params.tau_theta = tau_theta
            params.beta_start = args.beta_start
            params.beta_end = args.beta_end
            params.input_size = (args.z_channels, args.img_size, args.img_size)

            return SDE_LDM(params)

    def create_vae_model(self) -> VAE:
        args = self.args

        vae_trainer = vae.VAETrainer()

        vae_args = vae_trainer.args
        vae_args.checkpoint = args.vae_checkpoint
        vae_args.device = args.device
        vae_args.in_channels = args.in_channels
        vae_args.img_size = args.img_size

        vae_model = vae_trainer.load_last_checkpoint()[0]

        return vae_model

    def create_model(self):
        args = self.args

        eps_theta = ConditionalUNet(
            in_channels=args.z_channels,
            out_channels=args.z_channels,
        )

        tau_theta = nn.Embedding(args.num_classes, eps_theta.emb_dim)

        return nn.ModuleDict({"eps_theta": eps_theta, "tau_theta": tau_theta})

    def load_last_checkpoint(self):
        args = self.args

        model = self.create_model().to(args.device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        last_epoch = -1

        if hasattr(args, "checkpoint") and args.checkpoint is not None:
            logging.info(f"Loading checkpoint {args.checkpoint}")
            last_epoch = utils.load_state_dict(
                model,
                optimizer,
                args.prefix,
                args.run_name,
                args.checkpoint,
                args.device,
            )

            model.to(args.device)

        return model, optimizer, last_epoch

    def pre_train(self, model: nn.ModuleDict, **kwargs):
        self.diffusion = self.create_diffusion_model(**model)
        self.diffusion.train()

        self.vae = self.create_vae_model()
        self.vae.eval()

    def train_step(self, batch: Any, **kwargs) -> torch.Tensor:
        device = self.args.device

        images = batch[0].to(device)
        labels = batch[1].to(device)

        encoded_images = self.vae.encode(images)

        t = self.diffusion.t(encoded_images.shape[0])

        loss = self.diffusion.calc_loss(encoded_images, t, labels)

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
            y=labels,
        )
        self.diffusion.train()

        decoded_images = self.vae.decode(sampled_images)

        logging.info(f"Saving results for epoch {epoch+1}")
        utils.save_images(
            decoded_images,
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

    def pre_inference(self, model: nn.Module, **kwargs):
        self.pre_train(model=model, **kwargs)

        self.diffusion.eval()
        self.vae.eval()

    def create_default_args(self):
        args = Namespace()
        args.prefix = "."
        args.run_name = "LDM"
        args.model_type = "sde"
        args.epochs = 500
        args.batch_size = 12
        args.shuffle = True
        args.img_size = 64
        args.in_channels = 3
        args.z_channels = 32
        args.T = 1000
        args.beta_start = 1e-4
        args.beta_end = 2e-2
        args.device = "cuda"
        args.lr = 3e-4
        args.alpha = 0.1
        args.checkpoint = None
        args.vae_checkpoint = None
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
        parser.add_argument("--z_channels", type=int, default=d_args.z_channels)
        parser.add_argument("--T", type=int, default=d_args.T)
        parser.add_argument("--beta_start", type=float, default=d_args.beta_start)
        parser.add_argument("--beta_end", type=float, default=d_args.beta_end)
        parser.add_argument("--dataset_path", type=str, required=True)
        parser.add_argument("--device", type=str, default=d_args.device)
        parser.add_argument("--lr", type=float, default=d_args.lr)
        parser.add_argument("--num_classes", type=int, required=True)
        parser.add_argument("--alpha", type=float, default=d_args.alpha)
        parser.add_argument("--checkpoint", type=str, default=d_args.checkpoint)
        parser.add_argument("--vae_checkpoint", type=str, default=d_args.vae_checkpoint)
        parser.add_argument("--save_freq", type=int, default=d_args.save_freq)

        return parser
