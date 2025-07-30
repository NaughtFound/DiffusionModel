from typing import Any, Sequence
import torch
from torch import optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from models.vae.base import VAE
from models.vae.vq import VAE_VQ_Params, VAE_VQ
from trainer.grad import GradientTrainer
import utils


class VAETrainer(GradientTrainer):
    def create_model(self) -> VAE:
        args = self.args

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

    def load_last_checkpoint(self):
        args = self.args

        vae = self.create_model().to(args.device)

        optimizer = optim.Adam(vae.parameters(), lr=args.lr)

        last_epoch = -1

        if hasattr(args, "checkpoint") and args.checkpoint is not None:
            logging.info(f"Loading checkpoint {args.checkpoint}")
            last_epoch = utils.load_state_dict(
                vae,
                optimizer,
                args.prefix,
                args.run_name,
                args.checkpoint,
                args.device,
            )

            vae.to(args.device)

        return vae, optimizer, last_epoch

    def calc_var(self, dataloader: DataLoader) -> torch.Tensor:
        sm = 0.0
        sm_sq = 0.0
        total = 0
        for batch in tqdm(dataloader, desc="Calculating variance of input dataset"):
            if isinstance(batch, Sequence):
                batch = batch[0]

            total += len(batch)
            sm += batch.sum(0)
            sm_sq += (batch**2).sum(0)

        mean = sm / total
        mean_sq = sm_sq / total

        return mean_sq - mean**2

    def pre_train(self, dataloader: DataLoader, **kwargs):
        self.var = self.calc_var(dataloader)

    def train_step(self, model: VAE, batch: Any) -> torch.Tensor:
        device = self.args.device

        if isinstance(batch, Sequence):
            batch = batch[0]

        images = batch.to(device)

        loss = model.calc_loss(images, self.var)

        return loss

    def save_step(
        self,
        model: VAE,
        optimizer: optim.Optimizer,
        epoch: int,
        batch: Any,
    ):
        args = self.args

        if isinstance(batch, Sequence):
            batch = batch[0]

        batch = batch.to(args.device)

        logging.info(f"Sampling for epoch {epoch + 1}")
        model.eval()
        sampled_images = model(batch)
        model.train()
        utils.save_images(
            sampled_images,
            args.prefix,
            args.run_name,
            f"{epoch + 1}.jpg",
        )
        logging.info(f"Saving results for epoch {epoch + 1}")
        utils.save_state_dict(
            model,
            optimizer,
            epoch,
            args.prefix,
            args.run_name,
            f"ckpt-{epoch + 1}.pt",
        )

    def post_train(self):
        pass

    def pre_inference(self, model: VAE, **kwargs):
        self.vae = model
        self.vae.eval()

    def create_default_args(self):
        args = super().create_default_args()
        args.run_name = "VAE-VQ"
        args.model_type = "vq"
        args.in_channels = 3
        args.img_size = 54
        args.hidden_dim = 64
        args.embedding_dim = 32
        args.n_embeddings = 256
        args.res_h_dim = 16
        args.n_res_layers = 1
        args.beta = 0.25

        return args

    def get_arg_parser(self):
        parser = super().get_arg_parser()

        d_args = self.create_default_args()

        parser.add_argument("--in_channels", type=int, default=d_args.in_channels)
        parser.add_argument("--img_size", type=int, default=d_args.img_size)
        parser.add_argument("--hidden_dim", type=int, default=d_args.hidden_dim)
        parser.add_argument("--embedding_dim", type=int, default=d_args.embedding_dim)
        parser.add_argument("--n_embeddings", type=int, default=d_args.n_embeddings)
        parser.add_argument("--res_h_dim", type=int, default=d_args.res_h_dim)
        parser.add_argument("--n_res_layers", type=int, default=d_args.n_res_layers)
        parser.add_argument("--beta", type=float, default=d_args.beta)

        return parser
