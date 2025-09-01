from typing import Any
import numpy as np
import torch
from torch import optim, nn
import logging
import utils
from models.unet.label_conditioned import LabelConditionedUNet
from models.diffusion.base import Diffusion
from models.diffusion.cfg import CFG_Params, CFG
from trainer.models.ddpm import DDPMTrainer


class CFGTrainer(DDPMTrainer):
    def create_diffusion_model(self, eps_theta: nn.Module) -> Diffusion:
        args = self.args

        if args.model_type == "sde":
            params = CFG_Params(args.device)
            params.eps_theta = eps_theta
            params.beta_min = args.beta_min
            params.beta_max = args.beta_max
            params.input_size = (args.in_channels, args.img_size, args.img_size)
            params.fast_cfg = args.fast_cfg

            return CFG(params)

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

        if np.random.random() < args.alpha and self.diffusion.training:
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

        logging.info(f"Sampling for epoch {epoch + 1}")
        self.diffusion.eval()
        labels = torch.arange(args.num_classes).long().to(args.device)
        sampled_images = self.diffusion.sample(
            n=args.num_classes,
            labels=labels,
            cfg_scale=args.cfg_scale,
        )
        self.diffusion.train()
        logging.info(f"Saving results for epoch {epoch + 1}")
        utils.save_images(
            sampled_images,
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

    def post_train(self):
        pass

    @staticmethod
    def create_default_args():
        args = super(CFGTrainer, CFGTrainer).create_default_args()

        args.run_name = "CFG"
        args.model_type = "default"
        args.alpha = 0.1
        args.fast_cfg = True
        args.cfg_scale = 0.1

        return args

    @staticmethod
    def get_arg_parser():
        parser = super(CFGTrainer, CFGTrainer).get_arg_parser()

        d_args = CFGTrainer.create_default_args()

        parser.add_argument("--alpha", type=float, default=d_args.alpha)
        parser.add_argument("--fast_cfg", type=bool, default=d_args.fast_cfg)
        parser.add_argument("--cfg_scale", type=float, default=d_args.cfg_scale)
        parser.add_argument("--num_classes", type=int, required=True)

        return parser
