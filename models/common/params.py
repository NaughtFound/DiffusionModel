import torch


class ModelParams:
    device: torch.device

    def __init__(self, device: torch.device) -> None:
        self.device = device
