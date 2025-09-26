from typing import Any, Literal, Optional
import torch
from torch import nn
import logging
from trainer.grad import GradientTrainerState
import utils
from models.unet.conditional import ConditionalUNet
from models.vit.dit import DiT
from models.diffusion.base import Diffusion
from models.diffusion.ldm import LDM_Params, LDM
from models.vae.base import VAE
from .ddpm import DDPMTrainer
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
                    args.vae_latent_channels,
                    args.latent_size,
                    args.latent_size,
                )
            else:
                params.input_size = (args.in_channels, args.img_size, args.img_size)

            return LDM(params)

    def create_vae_model(self) -> Optional[VAE]:
        args = self.args

        if not args.use_vae:
            return None

        vae_args = {}

        for key, value in args.__dict__.items():
            if key.startswith("vae_"):
                vae_args[key.split("vae_")[1]] = value

        vae_trainer = vae.VAETrainer(**vae_args)

        vae_state = vae_trainer.load_last_checkpoint()
        vae_model = vae_state.model
        vae_model.eval()

        return vae_model

    def create_model(self):
        args = self.args

        channels = args.in_channels
        img_size = args.img_size

        if args.use_vae:
            channels = args.vae_latent_channels
            img_size = args.latent_size

        if args.eps_theta_type == "unet":
            eps_theta = ConditionalUNet.from_params_with_kwargs(
                args,
                in_channels=channels,
                out_channels=channels,
            )
            tau_theta = nn.Embedding(args.num_classes, eps_theta.emb_dim)

            return nn.ModuleDict({"eps_theta": eps_theta, "tau_theta": tau_theta})

        elif args.eps_theta_type == "dit":
            eps_theta = DiT.from_params_with_kwargs(
                args,
                in_channels=channels,
                input_size=img_size,
                learn_sigma=False,
            )
            tau_theta = nn.Embedding(args.num_classes, eps_theta.hidden_size)

            return nn.ModuleDict({"eps_theta": eps_theta, "tau_theta": tau_theta})

        else:
            raise ValueError(f"{args.eps_theta_type} is not valid for `eps_theta_type`")

    def pre_train(self, model: nn.ModuleDict, **kwargs):
        self.diffusion = self.create_diffusion_model(**model)
        self.diffusion.train()

        self.vae = self.create_vae_model()

    def train_step(self, batch: Any, **kwargs) -> torch.Tensor:
        device = self.args.device

        images = batch[0].to(device)
        labels = batch[1].to(device)

        if isinstance(self.vae, VAE):
            images = self.vae.encode(images)

        t = self.diffusion.t(images.shape[0])

        loss = self.diffusion.calc_loss(images, t, labels)

        return loss

    def save_step(self, state: GradientTrainerState, **kwargs):
        args = self.args

        logging.info(f"Sampling for epoch {state.epoch + 1}")
        self.diffusion.eval()
        labels = torch.arange(args.num_classes).long().to(args.device)
        sampled_images = self.diffusion.sample(
            n=args.num_classes,
            y=labels,
        )
        self.diffusion.train()

        decoded_images = self.vae.decode(sampled_images)

        logging.info(f"Saving results for epoch {state.epoch + 1}")
        utils.save_images(
            decoded_images,
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
        self.vae.eval()

    @staticmethod
    def create_default_args():
        args = super(LDMTrainer, LDMTrainer).create_default_args()
        vae_args = vae.VAETrainer.create_default_args()

        args.run_name = "LDM"
        args.model_type = "sde"
        args.eps_theta_type = "unet"

        args.use_vae = False
        args.latent_size = args.img_size

        utils.add_prefixed_namespace(vae_args, args, "vae_")

        return args

    @staticmethod
    def get_arg_parser():
        parser = super(LDMTrainer, LDMTrainer).get_arg_parser()
        vae_parser = vae.VAETrainer.get_arg_parser()

        d_args = LDMTrainer.create_default_args()

        parser.add_argument("--num_classes", type=int, required=True)
        parser.add_argument(
            "--eps_theta_type",
            type=Literal["unet", "dit"],
            default=d_args.eps_theta_type,
        )

        parser.add_argument("--use_vae", type=bool, default=d_args.use_vae)
        parser.add_argument("--latent_size", type=int, default=d_args.latent_size)

        utils.add_prefixed_arguments(vae_parser, parser, "vae_")

        return parser
