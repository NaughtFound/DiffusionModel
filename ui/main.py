import os
import torch
from typing import Literal
import streamlit as st
from trainer import ddpm, cfg, ldm
import utils


class DiffusionConfigs:
    model_name: Literal["ddpm", "cfg", "ldm"]
    in_channels: int
    num_classes: int
    checkpoint: str
    device: Literal["cpu", "cuda"]
    time: tuple[float, float]

    def load_checkpoints(self):
        model_path = os.path.join("ui", "weights", self.model_name)

        if not os.path.exists(model_path):
            return []

        checkpoints = [f for f in os.listdir(model_path) if f.endswith(".pt")]

        if len(checkpoints) > 0:
            self.checkpoint = checkpoints[-1]
        else:
            self.checkpoint = None

        return checkpoints

    def __init__(self):
        self.model_name = "ddpm"
        self.in_channels = 1
        self.num_classes = 10
        self.checkpoint = None
        self.device = "cpu"
        self.time = (0, 1)


def inference_tab(config: DiffusionConfigs):
    if config.model_name == "ddpm":
        args = ddpm.create_default_args()
        args.prefix = "ui"
        args.in_channels = config.in_channels
        args.model_type = "sde"
        args.run_name = "ddpm"
        args.checkpoint = config.checkpoint
        args.device = config.device

        if config.checkpoint is None:
            st.toast("Please Select Model Weights")
            return

        eps_theta, _, _ = ddpm.load_last_checkpoint(args)
        diffusion = ddpm.create_diffusion_model(eps_theta, args)

    if config.model_name == "cfg":
        args = cfg.create_default_args()
        args.prefix = "ui"
        args.in_channels = config.in_channels
        args.num_classes = config.num_classes
        args.model_type = "sde"
        args.run_name = "cfg"
        args.checkpoint = config.checkpoint
        args.device = config.device

        if config.checkpoint is None:
            st.toast("Please Select Model Weights")
            return

        eps_theta, _, _ = cfg.load_last_checkpoint(args)
        diffusion = cfg.create_diffusion_model(eps_theta, args)


def main():
    config = DiffusionConfigs()

    st.set_page_config(layout="wide", page_title="Diffusion Models Playground")
    st.title("Diffusion Models Playground")
    tabs = st.tabs(
        [
            "Inference With Diffusion Models",
            "Generate Counterfactual Images",
        ]
    )

    st.sidebar.title("Model Configuration")
    config.model_name = st.sidebar.selectbox(
        label="Model Type",
        options=["DDPM", "CFG", "LDM"],
    ).lower()
    config.device = st.sidebar.radio(
        label="Inference On",
        options=["CUDA", "CPU"],
    ).lower()

    checkpoints = config.load_checkpoints()

    config.checkpoint = st.sidebar.selectbox(
        label="Model Weights",
        options=checkpoints,
    )

    with tabs[0]:
        inference_tab(config)


if __name__ == "__main__":
    torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

    utils.setup_logging("ddpm", "ui")
    utils.setup_logging("cfg", "ui")
    utils.setup_logging("ldm", "ui")

    main()
