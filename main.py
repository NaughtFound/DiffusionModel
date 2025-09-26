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

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-m", "--module", type=str, nargs="*")

    args = parser.parse_args()

    config = ConfigParser(args.config).parse()

    if args.module is not None:
        for module_name in args.module:
            if module_name not in config:
                raise KeyError(f"{module_name} module not found")

            logging.info(f"Running {module_name}")

            module = config[module_name]

            if isinstance(module, Trainer):
                module.train()

    else:
        for k in config:
            logging.info(f"Running {k}")
            module = config[k]
            if isinstance(module, Trainer):
                module.train()
