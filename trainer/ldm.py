import os
import torch
import numpy as np
from torch import optim, nn
from torch.utils.tensorboard.writer import SummaryWriter
from argparse import Namespace
import logging
from tqdm import tqdm
from models.unet.conditional import ConditionalUNet
from models.diffusion.base import Diffusion
from models.sde.ldm import SDE_LDM_Params, SDE_LDM
import utils


logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


def create_diffusion_model(eps_theta: nn.Module, args: Namespace) -> Diffusion:
    if args.model_type == "default":
        raise NotImplementedError("default type is not implemented yet.")

    if args.model_type == "sde":
        params = SDE_LDM_Params(args.device)
        params.eps_theta = eps_theta
        params.beta_start = args.beta_start
        params.beta_end = args.beta_end
        params.input_size = (args.in_channels, args.img_size, args.img_size)

        return SDE_LDM(params)


def load_last_checkpoint(args: Namespace):
    eps_theta = ConditionalUNet(
        in_channels=args.in_channels,
        out_channels=args.in_channels,
        time_dim=args.time_dim,
    ).to(args.device)

    optimizer = optim.AdamW(eps_theta.parameters(), lr=args.lr)

    last_epoch = 0

    if hasattr(args, "checkpoint") and args.checkpoint is not None:
        logging.info(f"Loading checkpoint {args.checkpoint}")
        last_epoch = utils.load_state_dict(
            eps_theta,
            optimizer,
            args.run_name,
            args.checkpoint,
            args.device,
        )

        eps_theta.to(args.device)

    return eps_theta, optimizer, last_epoch


def train(args: Namespace):
    utils.setup_logging(args.run_name)
    device = args.device

    dataset = utils.create_dataset(args)
    dataloader = utils.create_dataloader(dataset, args)

    eps_theta, optimizer, last_epoch = load_last_checkpoint(args)

    diffusion = create_diffusion_model(eps_theta, args)
    diffusion.train()

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    len_data = len(dataloader)

    for epoch in range(last_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch+1}")

        for i, batch in enumerate(
            tqdm(
                dataloader,
                desc=f"Training [{epoch + 1}/{args.epochs}]",
            )
        ):
            images = batch[0].to(device)
            labels = batch[1].to(device)
            t = diffusion.t(images.shape[0])

            if np.random.random() < args.alpha:
                labels = None

            loss = diffusion.calc_loss(images, t, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar("Loss", loss.item(), global_step=epoch * len_data + i)

        if (epoch + 1) % args.save_freq == 0:
            logging.info(f"Sampling for epoch {epoch+1}")
            diffusion.eval()
            sampled_images = None
            raise NotImplementedError("sampling not implemented yet.")
            diffusion.train()
            logging.info(f"Saving results for epoch {epoch+1}")
            utils.save_images(sampled_images, args.run_name, f"{epoch+1}.jpg")
            utils.save_state_dict(
                eps_theta,
                optimizer,
                epoch,
                args.run_name,
                f"ckpt-{epoch+1}.pt",
            )


def create_default_args():
    args = Namespace()
    args.run_name = "LDM"
    args.model_type = "sde"
    args.epochs = 500
    args.batch_size = 12
    args.shuffle = True
    args.img_size = 64
    args.in_channels = 3
    args.T = 1000
    args.beta_start = 1e-4
    args.beta_end = 2e-2
    args.time_dim = 256
    args.device = "cuda"
    args.lr = 3e-4
    args.alpha = 0.1
    args.checkpoint = None
    args.save_freq = 5

    return args


def lunch():
    import argparse

    parser = argparse.ArgumentParser()

    d_args = create_default_args()

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
    parser.add_argument("--time_dim", type=int, default=d_args.time_dim)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--device", type=str, default=d_args.device)
    parser.add_argument("--lr", type=float, default=d_args.lr)
    parser.add_argument("--alpha", type=float, default=d_args.alpha)
    parser.add_argument("--checkpoint", type=str, default=d_args.checkpoint)
    parser.add_argument("--save_freq", type=int, default=d_args.save_freq)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    lunch()
