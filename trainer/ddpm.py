import os
import torch
from torch import optim, nn
from torch.utils.tensorboard.writer import SummaryWriter
from argparse import Namespace
import utils
from models.unet.base import UNet
from models.diffusion.ddpm import Diffusion_DDPM
import logging


logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


def train(args: Namespace):
    utils.setup_logging(args.run_name)
    device = args.device

    dataset = utils.create_dataset(args)
    dataloader = utils.create_dataloader(dataset, args)

    model = UNet(
        in_channels=args.in_channels,
        out_channels=args.in_channels,
        time_dim=args.time_dim,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    diffusion = Diffusion_DDPM(
        T=args.T,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        img_size=args.img_size,
        in_channels=args.in_channels,
        device=device,
    )

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    len_data = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}")

        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            t = diffusion.t(images.shape[0])

            x_t, noise = diffusion.forward(images, t)
            noise_pred = model(x_t, t)

            loss = mse(noise, noise_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar("MSE", loss.item(), global_step=epoch * len_data + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        utils.save_images(sampled_images, args.run_name, f"{epoch}.jpg")
        utils.save_model(model, args.run_name, f"ckpt-{epoch}.pt")


def create_default_args():
    args = Namespace()
    args.run_name = "DDPM_unconditional"
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

    return args


def lunch():
    import argparse

    parser = argparse.ArgumentParser()

    d_args = create_default_args()

    parser.add_argument("--run_name", type=str, default=d_args.run_name)
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

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    lunch()
