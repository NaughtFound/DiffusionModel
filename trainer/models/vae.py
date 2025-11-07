from typing import Any, Literal, Sequence
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from models.vae.base import VAE
from models.vae.vq import VAE_VQ_Params, VAE_VQ
from models.vae.kl import VAE_KL_Params, VAE_KL
from trainer.grad import GradientTrainer, GradientTrainerState
import utils


class VAETrainer(GradientTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.global_step = 0
        self.optimizer_idx = {
            "AdamW_vae": 0,
            "AdamW_disc": 1,
        }

    def create_model(self) -> VAE:
        args = self.args

        if args.model_type == "vq":
            params = VAE_VQ_Params(args.device)
            params.in_channels = args.in_channels
            params.hidden_dim = args.hidden_dim
            params.latent_channels = args.latent_channels
            params.n_embeddings = args.n_embeddings
            params.res_h_dim = args.res_h_dim
            params.n_res_layers = args.n_res_layers
            params.beta = args.beta

            return VAE_VQ(params)

        if args.model_type == "kl":
            params = VAE_KL_Params(args.device)
            params.in_channels = args.in_channels
            params.out_channels = args.in_channels
            params.latent_channels = args.latent_channels
            params.pretrained_model_name_or_path = args.pretrained_model_name_or_path
            params.lpips_model_path = args.lpips_model_path
            params.disc_start = args.disc_start
            params.log_var_init = args.log_var_init
            params.kl_weight = args.kl_weight
            params.pixel_loss_weight = args.pixel_loss_weight
            params.disc_n_layers = args.disc_n_layers
            params.disc_in_channels = args.disc_in_channels
            params.disc_factor = args.disc_factor
            params.disc_weight = args.disc_weight
            params.perceptual_weight = args.perceptual_weight
            params.disc_conditional = args.disc_conditional
            params.disc_loss = args.disc_loss

            return VAE_KL(params)

    def load_last_checkpoint(self):
        args = self.args

        vae = self.create_model().to(args.device)

        if args.model_type == "vq":
            optimizer = optim.Adam(vae.parameters(), lr=args.lr)

        if args.model_type == "kl" and isinstance(vae, VAE_KL):
            vae_params = nn.ParameterList()
            vae_params.extend(list(vae.vae.encoder.parameters()))
            vae_params.extend(list(vae.vae.decoder.parameters()))
            vae_params.extend(list(vae.vae.quant_conv.parameters()))
            vae_params.extend(list(vae.vae.post_quant_conv.parameters()))

            optimizer = {
                "AdamW_vae": optim.Adam(vae_params, lr=args.lr),
                "AdamW_disc": optim.Adam(
                    vae.loss.discriminator.parameters(),
                    lr=args.lr,
                ),
            }

        last_epoch = -1
        run_id = None

        if hasattr(args, "checkpoint") and args.checkpoint is not None:
            logging.info(f"Loading checkpoint {args.checkpoint}")
            last_epoch, run_id = utils.load_state_dict(
                vae,
                optimizer,
                args.prefix,
                args.run_name,
                args.checkpoint,
                args.device,
            )

            vae.to(args.device)

        self.global_step += last_epoch

        return GradientTrainerState(
            model=vae,
            optimizer=optimizer,
            epoch=last_epoch,
            run_id=run_id,
        )

    def calc_var(self, dataloader: DataLoader) -> torch.Tensor:
        sm = 0.0
        sm_sq = 0.0
        total = 0
        for batch in tqdm(dataloader, desc="Calculating variance of input dataset"):
            if isinstance(batch, Sequence):
                batch = batch[0]

            batch = batch.to(self.args.device)

            total += len(batch)
            sm += batch.sum(0)
            sm_sq += (batch**2).sum(0)

        mean = sm / total
        mean_sq = sm_sq / total

        return mean_sq - mean**2

    def pre_train(self, state: GradientTrainerState):
        if self.args.model_type == "vq":
            dataloader = state.get_param("train_dataloader")

            self.var = self.calc_var(dataloader)

    def train_step(
        self,
        model: VAE,
        batch: Any,
        optim_name: str,
        **kwargs,
    ) -> torch.Tensor:
        args = self.args

        self.global_step += 1

        if isinstance(batch, Sequence):
            batch = batch[0]

        images = batch.to(args.device)

        if args.model_type == "vq":
            loss = model.calc_loss(x=images, var=self.var)

        if args.model_type == "kl":
            loss = model.calc_loss(
                x=images,
                optimizer_idx=self.optimizer_idx.get(optim_name, 0),
                global_step=self.global_step,
            )

        return loss

    def save_step(self, state: GradientTrainerState):
        args = self.args

        batch = state.get_param("batch")

        if isinstance(batch, Sequence):
            batch = batch[0]

        batch = batch.to(args.device)

        logging.info(f"Sampling for epoch {state.epoch + 1}")
        state.model.eval()
        sampled_images = state.model(batch)
        state.model.train()
        utils.save_images(
            sampled_images,
            args.prefix,
            args.run_name,
            f"{state.epoch + 1}.jpg",
        )
        logging.info(f"Saving results for epoch {state.epoch + 1}")
        utils.save_state_dict(
            model=state.model,
            optimizer=state.optimizer,
            epoch=state.epoch,
            prefix=args.prefix,
            run_name=args.run_name,
            file_name=f"ckpt-{state.epoch + 1}.pt",
            run_id=state.run_id,
        )

    def pre_inference(self, state: GradientTrainerState[VAE]):
        self.vae = state.model
        self.vae.eval()

    @staticmethod
    def create_default_args():
        args = super(VAETrainer, VAETrainer).create_default_args()

        args.run_name = "VAE-VQ"
        args.model_type = "vq"
        args.in_channels = 3
        args.hidden_dim = 64
        args.latent_channels = 32
        args.n_embeddings = 256
        args.res_h_dim = 16
        args.n_res_layers = 1
        args.beta = 0.25
        args.pretrained_model_name_or_path = None
        args.lpips_model_path = None
        args.disc_start = 50001
        args.log_var_init = 0.0
        args.kl_weight = 1.0
        args.pixel_loss_weight = 1.0
        args.disc_n_layers = 3
        args.disc_in_channels = 3
        args.disc_factor = 1.0
        args.disc_weight = 1.0
        args.perceptual_weight = 1.0
        args.disc_conditional = False
        args.disc_loss = "hinge"

        return args

    @staticmethod
    def get_arg_parser():
        parser = super(VAETrainer, VAETrainer).get_arg_parser()

        d_args = VAETrainer.create_default_args()

        parser.add_argument("--in_channels", type=int, default=d_args.in_channels)
        parser.add_argument("--hidden_dim", type=int, default=d_args.hidden_dim)
        parser.add_argument(
            "--latent_channels",
            type=int,
            default=d_args.latent_channels,
        )
        parser.add_argument("--n_embeddings", type=int, default=d_args.n_embeddings)
        parser.add_argument("--res_h_dim", type=int, default=d_args.res_h_dim)
        parser.add_argument("--n_res_layers", type=int, default=d_args.n_res_layers)
        parser.add_argument("--beta", type=float, default=d_args.beta)

        parser.add_argument(
            "--pretrained_model_name_or_path",
            type=str,
            default=d_args.pretrained_model_name_or_path,
        )
        parser.add_argument(
            "--lpips_model_path",
            type=str,
            default=d_args.lpips_model_path,
        )
        parser.add_argument("--disc_start", type=int, default=d_args.disc_start)
        parser.add_argument("--log_var_init", type=float, default=d_args.log_var_init)
        parser.add_argument("--kl_weight", type=float, default=d_args.kl_weight)
        parser.add_argument(
            "--pixel_loss_weight",
            type=float,
            default=d_args.pixel_loss_weight,
        )
        parser.add_argument("--disc_n_layers", type=int, default=d_args.disc_n_layers)
        parser.add_argument(
            "--disc_in_channels",
            type=int,
            default=d_args.disc_in_channels,
        )
        parser.add_argument("--disc_factor", type=float, default=d_args.disc_factor)
        parser.add_argument("--disc_weight", type=float, default=d_args.disc_weight)
        parser.add_argument(
            "--perceptual_weight", type=float, default=d_args.perceptual_weight
        )
        parser.add_argument(
            "--disc_conditional",
            type=bool,
            default=d_args.disc_conditional,
        )
        parser.add_argument(
            "--disc_loss",
            type=Literal["hinge", "vanilla"],
            default=d_args.disc_loss,
        )

        return parser
