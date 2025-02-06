import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from argparse import Namespace
import logging
from tqdm import tqdm
from models.vae.base import VAE
from models.vae.vq import VAE_VQ_Params, VAE_VQ
import utils


logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


def create_vae_model(args: Namespace) -> VAE:
    if args.model_type == "vq":
        params = VAE_VQ_Params(args.device)
        params.in_channels = args.in_channels
        params.img_size = args.img_size
        params.hidden_dim = args.hidden_dim
        params.embedding_dim = args.embedding_dim
        params.n_embeddings = args.n_embeddings
        params.res_h_dim = args.res_h_dim
        params.n_res_layers = args.n_res_layers
        params.beta = args.beta

        return VAE_VQ(params)


def load_last_checkpoint(args: Namespace):
    vae = create_vae_model(args).to(args.device)

    optimizer = optim.Adam(vae.parameters(), lr=args.lr)

    last_epoch = 0

    if hasattr(args, "checkpoint") and args.checkpoint is not None:
        logging.info(f"Loading checkpoint {args.checkpoint}")
        last_epoch = utils.load_state_dict(
            vae,
            optimizer,
            args.run_name,
            args.checkpoint,
            args.device,
        )

        vae.to(args.device)

    return vae, optimizer, last_epoch


def calc_var(dataloader: DataLoader) -> torch.Tensor:
    mean = 0.0
    mean_sq = 0.0
    for batch in tqdm(dataloader, desc="Calculating variance of input dataset"):
        data = batch[0]

        mean = data.mean()
        mean_sq = (data**2).mean()

    return torch.sqrt(mean_sq - mean**2)


def train(args: Namespace):
    utils.setup_logging(args.run_name)
    device = args.device

    dataset = utils.create_dataset(args)
    dataloader = utils.create_dataloader(dataset, args)

    vae, optimizer, last_epoch = load_last_checkpoint(args)

    vae.train()

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    len_data = len(dataloader)

    var = calc_var(dataloader)

    for epoch in range(last_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch+1}")

        for i, batch in enumerate(
            tqdm(
                dataloader,
                desc=f"Training [{epoch + 1}/{args.epochs}]",
            )
        ):
            images = batch[0].to(device)

            loss = vae.calc_loss(images, var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar("Loss", loss.item(), global_step=epoch * len_data + i)

        if (epoch + 1) % args.save_freq == 0:
            logging.info(f"Sampling for epoch {epoch+1}")
            vae.eval()
            sampled_images = vae.sample(n=images.shape[0])
            vae.train()
            logging.info(f"Saving results for epoch {epoch+1}")
            utils.save_images(sampled_images, args.run_name, f"{epoch+1}.jpg")
            utils.save_state_dict(
                vae,
                optimizer,
                epoch,
                args.run_name,
                f"ckpt-{epoch+1}.pt",
            )


def create_default_args():
    args = Namespace()
    args.run_name = "VAE-VQ"
    args.model_type = "vq"
    args.epochs = 500
    args.batch_size = 12
    args.shuffle = True
    args.in_channels = 3
    args.img_size = 54
    args.hidden_dim = 64
    args.embedding_dim = 32
    args.n_embeddings = 256
    args.res_h_dim = 16
    args.n_res_layers = 1
    args.beta = 0.25
    args.device = "cuda"
    args.lr = 3e-4
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
    parser.add_argument("--in_channels", type=int, default=d_args.in_channels)
    parser.add_argument("--img_size", type=int, default=d_args.img_size)
    parser.add_argument("--hidden_dim", type=int, default=d_args.hidden_dim)
    parser.add_argument("--embedding_dim", type=int, default=d_args.embedding_dim)
    parser.add_argument("--n_embeddings", type=int, default=d_args.n_embeddings)
    parser.add_argument("--res_h_dim", type=int, default=d_args.res_h_dim)
    parser.add_argument("--n_res_layers", type=int, default=d_args.n_res_layers)
    parser.add_argument("--beta", type=float, default=d_args.beta)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--device", type=str, default=d_args.device)
    parser.add_argument("--lr", type=float, default=d_args.lr)
    parser.add_argument("--checkpoint", type=str, default=d_args.checkpoint)
    parser.add_argument("--save_freq", type=int, default=d_args.save_freq)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    lunch()
