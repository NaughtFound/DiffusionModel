import os
from utils import *
from models.unet import UNet
from models.simple_diffusion import SimpleDiffusion
from torch import optim, nn
from torch.utils.tensorboard.writer import SummaryWriter
import logging


logging.basicConfig(
    format='%(asctime)s - %(levelname)s: %(message)s',
    level=logging.INFO,
    datefmt='%I:%M:%S'
)


def train(args: dict):
    setup_logging(args.run_name)
    device = args.device

    dataloader = create_dataset(args)

    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    diffusion = SimpleDiffusion(img_size=args.img_size, device=device)

    logger = SummaryWriter(os.path.join('runs', args.run_name))

    len_data = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f'Starting epoch {epoch}')

        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            t = diffusion.t(images.shape[0]).to(device)

            x_t, noise = diffusion.forward(images, t)
            noise_pred = model(x_t, t)

            loss = mse(noise, noise_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar(
                'MSE',
                loss.item(),
                global_step=epoch*len_data + i
            )

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(
            sampled_images,
            os.path.join('results', args.run_name, f'{epoch}.jpg')
        )
        torch.save(
            model.state_dict(),
            os.path.join('models', args.run_name, f'ckpt-{epoch}.pt')
        )


def lunch():
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.run_name = 'DDPM_unconditional'
    args.epochs = 500
    args.batch_size = 12
    args.shuffle = True
    args.img_size = 64
    args.dataset_path = ''
    args.device = 'cuda'
    args.lr = 3e-4

    train(args)
