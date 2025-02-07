import os
import torch
from typing import Literal
import streamlit as st
from PIL import Image
from trainer import ddpm, cfg, ldm
import utils


class DiffusionConfigs:
    model_name: Literal["ddpm", "cfg", "ldm"]
    in_channels: int
    img_size: int
    num_classes: int
    checkpoint: str
    device: Literal["cpu", "cuda"]
    time: tuple[float, float]
    beta_start: float
    beta_end: float

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

    def load_model(self):
        if self.model_name == "ddpm":
            args = ddpm.create_default_args()
            args.prefix = "ui"
            args.in_channels = self.in_channels
            args.img_size = self.img_size
            args.model_type = "sde"
            args.run_name = "ddpm"
            args.checkpoint = self.checkpoint
            args.device = self.device
            args.beta_start = self.beta_start
            args.beta_end = self.beta_end

            if self.checkpoint is None:
                return

            eps_theta, _, _ = ddpm.load_last_checkpoint(args)
            diffusion = ddpm.create_diffusion_model(eps_theta, args)

        elif self.model_name == "cfg":
            args = cfg.create_default_args()
            args.prefix = "ui"
            args.in_channels = self.in_channels
            args.img_size = self.img_size
            args.num_classes = self.num_classes
            args.model_type = "sde"
            args.run_name = "cfg"
            args.checkpoint = self.checkpoint
            args.device = self.device
            args.beta_start = self.beta_start
            args.beta_end = self.beta_end

            if self.checkpoint is None:
                return

            eps_theta, _, _ = cfg.load_last_checkpoint(args)
            diffusion = cfg.create_diffusion_model(eps_theta, args)

        elif self.model_name == "ldm":
            args = ldm.create_default_args()
            args.prefix = "ui"
            args.in_channels = self.in_channels
            args.img_size = self.img_size
            args.num_classes = self.num_classes
            args.model_type = "sde"
            args.run_name = "ldm"
            args.checkpoint = self.checkpoint
            args.device = self.device
            args.beta_start = self.beta_start
            args.beta_end = self.beta_end

            if self.checkpoint is None:
                return

            eps_theta, _, _ = ldm.load_last_checkpoint(args)
            diffusion = ldm.create_diffusion_model(eps_theta, args)

        return diffusion

    def __init__(self):
        self.model_name = "ddpm"
        self.in_channels = 1
        self.img_size = 64
        self.num_classes = 10
        self.checkpoint = None
        self.device = "cpu"
        self.time = (0, 1)


def inference_tab(config: DiffusionConfigs):
    diffusion = config.load_model()
    if diffusion is None:
        st.toast("Please Select Model Weights")
        return

    if config.model_name == "ddpm":
        st.markdown(
            """
        >:primary[DDPM (Denoising Diffusion Probabilistic Model)] generates images by gradually removing noise from random data.
        The model learns to reverse a diffusion process that adds noise to images, allowing it to create new images from pure noise.
        """
        )

        n = st.number_input("Number of Images", step=1, value=1)

        if st.button("Generate Image"):
            x = diffusion.sample(n=n)
            images = utils.to_image(x).squeeze().cpu().numpy()

            cols = st.columns(4)

            for i in range(n):
                image = images[i]
                pil_image = Image.fromarray(image)

                cols[i % 4].image(pil_image)

    if config.model_name == "cfg":
        st.markdown(
            """
        >:primary[CFG (Classifier-Free Guidance)] enhances diffusion models by removing the need for a classifier.
        It works by combining the predictions from both conditioned and unconditioned models.
        """
        )

        n = st.number_input("Number of Images", step=1, value=1)
        label = st.selectbox("Label", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        cfg_scale = st.slider("CFG Scale", min_value=0.0, max_value=1.0, value=0.5)

        labels = torch.Tensor([label]).long().to(config.device)

        if st.button("Generate Image"):
            x = diffusion.sample(n=n, labels=labels, cfg_scale=cfg_scale)
            images = utils.to_image(x).squeeze().cpu().numpy()

            cols = st.columns(4)

            for i in range(n):
                image = images[i]
                pil_image = Image.fromarray(image)

                cols[i % 4].image(pil_image)


def main():
    config = DiffusionConfigs()

    st.set_page_config(layout="centered", page_title="Diffusion Models Playground")
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

    config.img_size = st.sidebar.slider(
        "Image Size",
        min_value=32,
        max_value=512,
        step=32,
    )

    config.beta_start, config.beta_end = st.sidebar.slider(
        "Beta",
        min_value=0.0,
        max_value=100.0,
        step=0.01,
        value=(1.0, 20.0),
    )

    with tabs[0]:
        inference_tab(config)


if __name__ == "__main__":
    torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

    utils.setup_logging("ddpm", "ui")
    utils.setup_logging("cfg", "ui")
    utils.setup_logging("ldm", "ui")

    main()
