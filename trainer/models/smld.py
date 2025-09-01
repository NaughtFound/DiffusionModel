from torch import nn
from models.diffusion.base import Diffusion
from models.diffusion.smld import SMLD_Params, SMLD
from .ddpm import DDPMTrainer


class SMLDTrainer(DDPMTrainer):
    def create_diffusion_model(self, eps_theta: nn.Module) -> Diffusion:
        args = self.args

        if args.model_type == "sde":
            params = SMLD_Params(args.device)
            params.eps_theta = eps_theta
            params.sigma_min = args.sigma_min
            params.sigma_max = args.sigma_max
            params.input_size = (args.in_channels, args.img_size, args.img_size)

            return SMLD(params)

    @staticmethod
    def create_default_args():
        args = super(SMLDTrainer, SMLDTrainer).create_default_args()

        args.run_name = "SMLD"
        args.model_type = "sde"
        args.sigma_min = 0.02
        args.sigma_max = 50

        return args

    @staticmethod
    def get_arg_parser():
        parser = super(SMLDTrainer, SMLDTrainer).get_arg_parser()

        d_args = SMLDTrainer.create_default_args()

        parser.add_argument("--sigma_min", type=float, default=d_args.sigma_min)
        parser.add_argument("--sigma_max", type=float, default=d_args.sigma_max)

        return parser
