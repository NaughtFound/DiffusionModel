from typing import Any
import torch
from torch import optim, nn
import logging
import utils
from models.unet.conditional import ConditionalUNet
from models.diffusion.base import Diffusion
from models.diffusion.ldm import LDM_Params, LDM
from models.vae.base import VAE
from trainer.models.ddpm import DDPMTrainer
from . import vae


class LDMTrainer(DDPMTrainer):
    def create_diffusion_model(
        self,
        eps_theta: nn.Module,
        tau_theta: nn.Module,
    ) -> Diffusion:
        args = self.args

        if args.model_type == "sde":
            params = LDM_Params(args.device)
            params.eps_theta = eps_theta
            params.tau_theta = tau_theta
            params.beta_min = args.beta_min
            params.beta_max = args.beta_max

            if args.use_vae:
                params.input_size = (
                    args.vae_hidden_dim,
                    args.latent_size,
                    args.latent_size,
                )
            else:
                params.input_size = (args.in_channels, args.img_size, args.img_size)

            return LDM(params)

    def create_vae_model(self) -> VAE:
        args = self.args

        vae_trainer = vae.VAETrainer(
            device=args.device,
            run_name=args.vae_run_name,
            model_type=args.vae_model_type,
            checkpoint=args.vae_checkpoint,
            in_channels=args.in_channels,
            img_size=args.img_size,
            hidden_dim=args.vae_hidden_dim,
            embedding_dim=args.vae_embedding_dim,
            n_embeddings=args.vae_n_embeddings,
            res_h_dim=args.vae_res_h_dim,
            n_res_layers=args.vae_n_res_layers,
            beta=args.vae_beta,
        )

        vae_model = vae_trainer.load_last_checkpoint()[0]

        return vae_model

    def create_model(self):
        args = self.args

        channels = args.in_channels

        if args.use_vae:
            channels = args.vae_embedding_dim

        eps_theta = ConditionalUNet(in_channels=channels, out_channels=channels)

        tau_theta = nn.Embedding(args.num_classes, eps_theta.emb_dim)

        return nn.ModuleDict({"eps_theta": eps_theta, "tau_theta": tau_theta})

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

        logging.info(f"Sampling for epoch {epoch + 1}")
        self.diffusion.eval()
        labels = torch.arange(args.num_classes).long().to(args.device)
        sampled_images = self.diffusion.sample(
            n=args.num_classes,
            y=labels,
        )
        self.diffusion.train()

        decoded_images = self.vae.decode(sampled_images)

        logging.info(f"Saving results for epoch {epoch + 1}")
        utils.save_images(
            decoded_images,
            args.prefix,
            args.run_name,
            f"{epoch + 1}.jpg",
        )
        utils.save_state_dict(
            model,
            optimizer,
            epoch,
            args.prefix,
            args.run_name,
            f"ckpt-{epoch + 1}.pt",
        )

    def pre_inference(self, model: nn.Module, **kwargs):
        self.pre_train(model=model, **kwargs)

        self.diffusion.eval()
        self.vae.eval()

    @staticmethod
    def create_default_args():
        args = super(LDMTrainer, LDMTrainer).create_default_args()

        args.run_name = "LDM"
        args.model_type = "sde"

        args.use_vae = False
        args.latent_size = args.img_size // 4

        return args

    @staticmethod
    def get_arg_parser():
        parser = super(LDMTrainer, LDMTrainer).get_arg_parser()

        d_args = LDMTrainer.create_default_args()

        parser.add_argument("--num_classes", type=int, required=True)

        parser.add_argument("--use_vae", type=bool, default=d_args.use_vae)
        parser.add_argument("--latent_size", type=int, default=d_args.latent_size)

        return parser
