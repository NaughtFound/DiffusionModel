from torch import nn
from models.diffusion.base import Diffusion
from models.diffusion.ddim import DDIM_Params, DDIM
from trainer.models.ddpm import DDPMTrainer


class DDIMTrainer(DDPMTrainer):
    def create_diffusion_model(self, eps_theta: nn.Module) -> Diffusion:
        args = self.args

        if args.model_type == "sde":
            params = DDIM_Params(args.device)
            params.eps_theta = eps_theta
            params.sigma_min = args.sigma_min
            params.sigma_max = args.sigma_max
            params.beta_min = args.beta_min
            params.beta_max = args.beta_max
            params.input_size = (args.in_channels, args.img_size, args.img_size)

            return DDIM(params)

    @staticmethod
    def create_default_args():
        args = super(DDIMTrainer, DDIMTrainer).create_default_args()

        args.run_name = "DDIM"
        args.model_type = "sde"
        args.sigma_min = 0.02
        args.sigma_max = 50

        return args

    @staticmethod
    def get_arg_parser():
        parser = super(DDIMTrainer, DDIMTrainer).get_arg_parser()

        d_args = DDIMTrainer.create_default_args()

        parser.add_argument("--sigma_min", type=float, default=d_args.sigma_min)
        parser.add_argument("--sigma_max", type=float, default=d_args.sigma_max)

        return parser
