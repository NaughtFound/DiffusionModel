# DiffusionModel

A modular PyTorch package for developing, training, and evaluating diffusion-based generative models. This project provides a configuration-driven workflow that makes it easy to experiment with new datasets, models, training procedures, and metrics without editing source code. It is designed to be flexible and extensible, making it suitable for both research and production use.

---

## Features

* **Configuration-first design** – experiments are controlled via YAML configuration files
* **Advanced parser** – dynamically imports and instantiates Python classes/functions from configs
* **Dataset abstraction** – flexible `DatasetLoader` base class with predefined loaders like MNIST
* **Model zoo** – modular implementations of UNet, VAE, and ViT, plus mixins for extensibility
* **Training framework** – base trainer logic, with gradient-based trainers for diffusion methods
* **Evaluation metrics** – base `Metrics` class and extendable evaluation modules
* **Interactive UI** – Streamlit app for inference and checkpoint visualization
* **Logging support** – built-in logging for key values like loss; logging of `Metrics` is possible through custom trainers

---

## Project Structure

The repository is organized into modular components:

```
DiffusionModel/
├── configs/        # YAML configuration files for experiments
├── loaders/        # Dataset loaders built on DatasetLoader base class
├── metrics/        # Evaluation metrics (FID, FVA, etc.)
├── models/         # Model implementations (UNet, VAE, ViT, base, common)
├── notebooks/      # Example notebooks for prototyping
├── trainer/        # Training logic: base Trainer and extensions
├── ui/             # Streamlit-based user interface for inference
├── utils/          # Utility functions: parser, loader, helpers
└── main.py         # Entrypoint script for running experiments via configs
```

### `configs/`

Contains experiment definitions in YAML format. A config describes:

* Which dataset loader to use
* Which model architecture to instantiate
* Which trainer class to run and with what parameters
* Metrics and evaluation setup

### `loaders/`

Defines dataset loaders extending from `utils/loader.DatasetLoader`. Example: `MNISTLoader`. Loaders are configurable from YAML, allowing easy swapping or parameterization of datasets.

### `metrics/`

Implements evaluation logic. The `Metrics` base class in `metrics/base.py` provides a common interface for new metrics. Subclasses extend this to implement domain-specific measures like FID or Inception Score.

### `models/`

Houses model architectures:

* `unet/` – UNet-based diffusion backbones
* `vae/` – Variational Autoencoders
* `vit/` – Vision Transformers
* `base/` – abstract model definitions
* `common/` – shared logic and mixins (e.g., `Model.from_params` to instantiate models from configs)

### `notebooks/`

Contains Jupyter notebooks for quick testing, prototyping, or demonstrating functionality. These are supplementary; the main execution is via `main.py`.

### `trainer/`

Defines the training process:

* `base.py` – generic `Trainer` base class
* Extensions – specialized trainers (e.g., gradient-based trainers for diffusion models)
* Includes logging of key metrics like **loss** by default
* If you want to log evaluation `Metrics` (from the `metrics` module), you can create a **custom trainer** to incorporate them into the training loop

### `ui/`

Streamlit interface for inference. Lets you:

* Load checkpoints
* Select a diffusion model
* Generate images interactively

### `utils/`

Helper functionality for the package:

* `parser.py` – advanced YAML parser, dynamically resolving modules, functions, and references
* `loader.py` – defines the `DatasetLoader` base class for flexible dataset integration

### `main.py`

The entrypoint script. It:

* Reads and parses a YAML config
* Resolves and instantiates all modules (trainer, model, dataset, metrics)
* Executes training or inference workflows

---

## Configuration System

The parser in `utils/parser.py` makes experiments highly flexible. Each YAML file may contain:

* **Direct values**: simple scalars (e.g., `epochs: 10`)
* **Resolvable values**: dictionaries specifying `module`, `source`, `call`, `args`, and optional `lazy`
* **References**: re-use values defined earlier in the config (`args.variable_name`)
* **Local modules**: custom Python files specified by `local` to extend functionality

This design lets you declaratively define entire experiments.

### Example Config

Below is the example config file:

```yaml
ddpm:
  module: trainer.models.ddpm
  source: DDPMTrainer
  args:
    prefix: notebooks
    model_type: sde
    img_size: 32
    in_channels: 1
    batch_size: 64
    shuffle: True
    save_freq: 50
    dataset_path: ./notebooks/data
    beta_min: 0.1
    beta_max: 1
    target_transform:
    download: True
    loader:
      module: loaders
      source: MNISTLoader
      call: False
```

This config demonstrates:

* Loading **MNIST** with `MNISTLoader`
* Using an **SDE-based diffusion trainer**
* Training with `batch_size=64` and saving checkpoints every 50 steps
* Building a UNet-like model with `in_channels=1` and `img_size=32`
* Since `ddpm` is a `Trainer` object, after parsing completed, the `train` function is called

> ⚠️ The config specifies `DDPMTrainer` as an example trainer. The structure allows swapping this with other trainers by editing only the config.

Run the experiment (which only executes the `ddpm` module defined in the config):

```bash
python main.py --config_path configs/mnist.yml --config_module=ddpm
```

---

## Core Components in Depth

### Parser

* Dynamically imports modules and their attributes (classes, functions)
* Resolves callables and instantiates them with parsed arguments
* Supports lazy instantiation for deferred evaluation (`FnWithKwargs` class)
* Allows referencing earlier config values to avoid duplication
* Enables custom extensions with `local` modules

### Dataset Loader

* Abstract base `DatasetLoader` defines dataset initialization and preprocessing
* Example: `MNISTLoader` fetches and prepares the MNIST dataset
* Loaders accept arguments directly from configs (e.g., `batch_size`, `shuffle`)

### Models

* **Base classes**: define common methods and interfaces
* **UNet**: backbone for most diffusion architectures
* **VAE**: for latent generative modeling
* **ViT**: transformer-based vision backbones
* **Common mixins**: provide methods like `from_params` for instantiating models from parsed configs

### Trainer

* **Trainer**: base class providing the training loop structure
* **Specialized trainers**: extend `Trainer` for specific algorithms (e.g., diffusion models)
* Handles checkpoints, logging, and optimizer/criterion setup
* Logs **loss and other core training statistics** by default
* Logging of evaluation `Metrics` is optional and requires creating a **custom trainer** to integrate them

### Metrics

* `Metrics` base class: defines evaluation API
* Subclasses implement specific metrics such as FID and Inception Score
* Metrics can be injected from configs, just like trainers or models

### UI

* Streamlit interface that:

  * Loads a trained checkpoint
  * Lets you choose a diffusion model
  * Generates and visualizes new images
* Designed for interactive exploration of results

---

## Extending the Package

You can extend functionality without touching core code:

* **New dataset**: implement a loader subclass and register in configs
* **New model**: subclass given base class and it will expose via `from_params`
* **New trainer**: subclass `Trainer` with your algorithm
* **New metrics**: implement and place in `metrics/`

All new modules can be integrated simply by referencing them in YAML configs.

---

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
