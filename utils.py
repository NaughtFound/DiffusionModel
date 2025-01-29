import os
import torch
import torch.nn as nn
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


def save_model(model: nn.Module, run_name: str, file_name: str):
    torch.save(
        model.state_dict(),
        os.path.join("weights", run_name, file_name),
    )


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
