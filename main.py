import argparse
import logging
from trainer.base import Trainer
from utils.parser import ConfigParser


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%I:%M:%S",
    )

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--config_module", type=str)

    args = parser.parse_args()

    config = ConfigParser(args.config_path).parse()

    if args.config_module is not None:
        if args.config_module not in config:
            raise KeyError(f"{args.config_module} module not found")

        module = config[args.config_module]

        if isinstance(module, Trainer):
            module.train()

    else:
        for k in config:
            logging.info(f"Running {k}")
            module = config[k]
            if isinstance(module, Trainer):
                module.train()
