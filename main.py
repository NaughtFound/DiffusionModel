import argparse
import logging
from trainer.base import Trainer
from kaizo import ConfigParser
from kaizo.utils import FnWithKwargs


def parse_key_value_args(args: list[str]) -> dict[str]:
    result = {}
    key = None
    for arg in args:
        if arg.startswith("--"):
            key = arg.lstrip("-")
            result[key] = True
        else:
            if key:
                result[key] = arg
                key = None
    return result


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%I:%M:%S",
    )

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-m", "--module", type=str, nargs="*")

    args, extra_args = parser.parse_known_args()

    kwargs = parse_key_value_args(extra_args)

    config = ConfigParser(args.config, kwargs).parse()

    if args.module is not None:
        for module_name in args.module:
            if module_name not in config:
                raise KeyError(f"{module_name} module not found")

            logging.info(f"Running {module_name}")

            module = config[module_name]

            if isinstance(module, Trainer):
                module.train()

            elif isinstance(module, FnWithKwargs):
                module.__call__()

    else:
        for k in config:
            logging.info(f"Running {k}")
            module = config[k]

            if isinstance(module, Trainer):
                module.train()

            elif isinstance(module, FnWithKwargs):
                module.__call__()
