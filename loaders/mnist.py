from torchvision import transforms, datasets
from utils.loader import ConfigKey, DataloaderConfig, DatasetLoader


class MNISTLoader(DatasetLoader):
    def get_transform(self):
        args = self.args

        def dequantize(x, nvals=256):
            """[0, 1] -> [0, nvals] -> add uniform noise -> [0, 1]"""
            noise = x.new().resize_as_(x).uniform_()
            x = x * (nvals - 1) + noise
            x = x / nvals
            return x

        transform = transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                dequantize,
            ]
        )

        return transform

    def get_dataloader_configs(self):
        args = self.args

        kwargs = {
            "root": args.dataset_path,
            "transform": self.get_transform(),
            "target_transform": args.target_transform,
        }

        train_dataset = datasets.MNIST(
            **kwargs,
            train=True,
            download=args.download,
        )
        valid_dataset = datasets.MNIST(
            **kwargs,
            train=False,
            download=False,
        )

        train_config = DataloaderConfig(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
        )

        valid_config = DataloaderConfig(
            dataset=valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

        return {
            ConfigKey.train: train_config,
            ConfigKey.valid: valid_config,
        }
