import os
import torch
from torch import optim, nn
from torch.utils.tensorboard.writer import SummaryWriter
from argparse import Namespace
import logging
from tqdm import tqdm
from models.unet.conditional import ConditionalUNet
from models.diffusion.base import Diffusion
from models.vae.base import VAE
from models.sde.ldm import SDE_LDM_Params, SDE_LDM
import utils
from . import vae


logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


def create_diffusion_model(
    eps_theta: nn.Module,
    tau_theta: nn.Module,
    args: Namespace,
) -> Diffusion:
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


def load_last_checkpoint(args: Namespace):
    eps_theta = ConditionalUNet(
        in_channels=args.z_channels,
        out_channels=args.z_channels,
    ).to(args.device)

    tau_theta = nn.Embedding(args.num_classes, eps_theta.emb_dim).to(args.device)

    params = nn.ParameterList()
    params.extend(list(eps_theta.parameters()))
    params.extend(list(tau_theta.parameters()))

    optimizer = optim.AdamW(params, lr=args.lr)

    last_epoch = 0

    if hasattr(args, "checkpoint") and args.checkpoint is not None:
        logging.info(f"Loading checkpoint {args.checkpoint}")
        last_epoch = utils.load_state_dict(
            {
                "eps_theta": eps_theta,
                "tau_theta": tau_theta,
            },
            optimizer,
            args.prefix,
            args.run_name,
            args.checkpoint,
            args.device,
        )

        eps_theta.to(args.device)
        tau_theta.to(args.device)

    return eps_theta, tau_theta, optimizer, last_epoch


def create_vae_model(args: Namespace) -> VAE:
    vae_args = vae.create_default_args()
    vae_args.checkpoint = args.vae_checkpoint
    vae_args.device = args.device
    vae_args.in_channels = args.in_channels
    vae_args.img_size = args.img_size

    vae_model = vae.load_last_checkpoint(vae_args)[0]

    return vae_model


def train(args: Namespace):
    utils.setup_logging(args.run_name, args.prefix)
    device = args.device

    dataset = utils.create_dataset(args)
    dataloader = utils.create_dataloader(dataset, args)

    eps_theta, tau_theta, optimizer, last_epoch = load_last_checkpoint(args)

    diffusion = create_diffusion_model(eps_theta, tau_theta, args)
    diffusion.train()

    vae_model = create_vae_model(args)
    vae_model.eval()

    logger = SummaryWriter(os.path.join(args.prefix, "runs", args.run_name))

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

            encoded_images = vae_model.encode(images)

            t = diffusion.t(encoded_images.shape[0])

            loss = diffusion.calc_loss(encoded_images, t, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar("Loss", loss.item(), global_step=epoch * len_data + i)

        if (epoch + 1) % args.save_freq == 0:
            logging.info(f"Sampling for epoch {epoch+1}")
            diffusion.eval()
            labels = torch.arange(args.num_classes).long().to(device)
            sampled_images = diffusion.sample(
                n=args.num_classes,
                y=labels,
            )
            diffusion.train()

            decoded_images = vae_model.decode(sampled_images)

            logging.info(f"Saving results for epoch {epoch+1}")
            utils.save_images(
                decoded_images,
                args.prefix,
                args.run_name,
                f"{epoch+1}.jpg",
            )
            utils.save_state_dict(
                {
                    "eps_theta": eps_theta,
                    "tau_theta": tau_theta,
                },
                optimizer,
                epoch,
                args.prefix,
                args.run_name,
                f"ckpt-{epoch+1}.pt",
            )


def create_default_args():
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


def lunch():
    import argparse

    parser = argparse.ArgumentParser()

    d_args = create_default_args()

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

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    lunch()
