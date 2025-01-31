import os
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, datasets
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from argparse import Namespace


def plot_images(images: torch.Tensor, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    grid_numpy = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(32, 32))
    plt.imshow(grid_numpy)
    plt.show()


def save_images(images: torch.Tensor, run_name: str, file_name: str, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    grid_numpy = grid.permute(1, 2, 0).cpu().numpy()

    image = Image.fromarray(grid_numpy)
    image.save(os.path.join("results", run_name, file_name))


def save_state_dict(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    run_name: str,
    file_name: str,
):
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    torch.save(
        state_dict,
        os.path.join("weights", run_name, file_name),
    )


def load_state_dict(
    model: nn.Module,
    optimizer: optim.Optimizer,
    run_name: str,
    file_name: str,
    device: torch.device,
) -> int:
    path = os.path.join("weights", run_name, file_name)
    if os.path.exists(path):
        state_dict = torch.load(path, weights_only=True, map_location=device)
        model.load_state_dict(state_dict.get("model"))
        optimizer.load_state_dict(state_dict.get("optimizer"))
        return state_dict.get("epoch")

    return 0


def create_dataset(args: Namespace) -> Dataset:
    if hasattr(args, "dataset"):
        return args.dataset

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = datasets.ImageFolder(
        root=args.dataset_path,
        transform=transform,
    )

    return dataset


def create_dataloader(dataset: Dataset, args: Namespace) -> DataLoader:
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    return dataloader


def setup_logging(run_name: str):
    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("weights", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
