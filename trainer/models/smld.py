from torch import nn
from models.diffusion.base import Diffusion
from models.sde.smld import SDE_SMLD_Params, SDE_SMLD
from trainer.models.ddpm import DDPMTrainer


class SMLDTrainer(DDPMTrainer):
    def create_diffusion_model(self, eps_theta: nn.Module) -> Diffusion:
        args = self.args

        if args.model_type == "default":
            raise NotImplementedError("default type is not implemented yet.")

        if args.model_type == "sde":
            params = SDE_SMLD_Params(args.device)
            params.eps_theta = eps_theta
            params.sigma_min = args.sigma_min
            params.sigma_max = args.sigma_max
            params.input_size = (args.z_channels, args.img_size, args.img_size)

            return SDE_SMLD(params)

    def create_default_args(self):
        args = super().create_default_args()
        args.run_name = "SMLD"
        args.model_type = "sde"
        args.sigma_min = 0.02
        args.sigma_max = 50

        return args

    def get_arg_parser(self):
        parser = super().get_arg_parser()

        d_args = self.create_default_args()

        parser.add_argument("--sigma_min", type=float, default=d_args.sigma_min)
        parser.add_argument("--sigma_max", type=float, default=d_args.sigma_max)

        return parser
