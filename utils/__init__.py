import os
import torch
from torch import nn, optim
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from typing import Union


def plot_images(images: torch.Tensor, heatmap: bool = False, **kwargs):
    grid = torchvision.utils.make_grid(images, normalize=True)
    grid_numpy = grid.permute(1, 2, 0).cpu().numpy()

    if heatmap:
        grid_numpy = grid_numpy.mean(axis=2)

    plt.figure(figsize=(32, 32))
    plt.imshow(grid_numpy, **kwargs)
    plt.axis("off")
    plt.show()


def to_image(x: torch.Tensor) -> torch.Tensor:
    x_0 = (x.clamp(-1, 1) + 1) / 2
    x_0 = (x_0 * 255).to(torch.uint8)

    return x_0


def save_images(
    x: torch.Tensor,
    prefix: str,
    run_name: str,
    file_name: str,
    save_as_grid: bool = True,
    **kwargs,
):
    images = to_image(x)

    file_path = os.path.join(prefix, "results", run_name, file_name)

    if save_as_grid:
        grid = torchvision.utils.make_grid(images, **kwargs)
        grid_numpy = grid.permute(1, 2, 0).cpu().numpy()

        image = Image.fromarray(grid_numpy)
        image.save(file_path)
    else:
        file_path, file_ext = os.path.splitext(file_path)

        if not file_ext:
            file_ext = ".png"

        os.makedirs(file_path, exist_ok=True)
        for i, image_ts in enumerate(images):
            image_numpy = image_ts.permute(1, 2, 0).squeeze().cpu().numpy()
            image = Image.fromarray(image_numpy)
            image.save(os.path.join(file_path, f"{i}{file_ext}"))


def save_state_dict(
    model: Union[nn.Module, dict[str, nn.Module]],
    optimizer: Union[optim.Optimizer, dict[str, optim.Optimizer]],
    epoch: int,
    prefix: str,
    run_name: str,
    file_name: str,
):
    state_dict = {"epoch": epoch}

    if isinstance(optimizer, optim.Optimizer):
        state_dict["optimizer"] = optimizer.state_dict()
    else:
        for optim_name in optimizer:
            state_dict[optim_name] = optimizer[optim_name].state_dict()

    if isinstance(model, nn.Module):
        state_dict["model"] = model.state_dict()
    else:
        for model_name in model:
            state_dict[model_name] = model[model_name].state_dict()

    torch.save(
        state_dict,
        os.path.join(prefix, "weights", run_name, file_name),
    )


def load_state_dict(
    model: Union[nn.Module, dict[str, nn.Module]],
    optimizer: Union[optim.Optimizer, dict[str, optim.Optimizer]],
    prefix: str,
    run_name: str,
    file_name: str,
    device: torch.device,
) -> int:
    path = os.path.join(prefix, "weights", run_name, file_name)
    if os.path.exists(path):
        state_dict = torch.load(path, weights_only=True, map_location=device)

        if isinstance(optimizer, optim.Optimizer):
            optimizer.load_state_dict(state_dict.get("optimizer"))
        else:
            for optim_name in optimizer:
                optimizer[optim_name].load_state_dict(state_dict.get(optim_name))

        if isinstance(model, nn.Module):
            model.load_state_dict(state_dict.get("model"))
        else:
            for model_name in model:
                model[model_name].load_state_dict(state_dict.get(model_name))

        return state_dict.get("epoch")

    return -1


def setup_logging(run_name: str, prefix: str = "."):
    os.makedirs(os.path.join(prefix, "weights"), exist_ok=True)
    os.makedirs(os.path.join(prefix, "results"), exist_ok=True)
    os.makedirs(os.path.join(prefix, "weights", run_name), exist_ok=True)
    os.makedirs(os.path.join(prefix, "results", run_name), exist_ok=True)


def fill_tail_dims(x: torch.Tensor, x_like: torch.Tensor):
    return x[(...,) + (None,) * (x_like.dim() - x.dim())]
