import os
import torch
import torchvision
from torchvision import transforms, datasets
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images: torch.Tensor, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    grid_numpy = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(32, 32))
    plt.imshow(grid_numpy)
    plt.show()


def save_images(images: torch.Tensor, path: str, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    grid_numpy = grid.permute(1, 2, 0).cpu().numpy()

    image = Image.fromarray(grid_numpy)
    image.save(path)


def create_dataset(args: dict) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(80),
            transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = datasets.ImageFolder(args.dataset_path, transform=transform)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    return dataloader


def setup_logging(run_name: str):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
