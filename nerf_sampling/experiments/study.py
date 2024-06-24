"""Script for running hyperparameters sweep."""

import optuna
import torch
import wandb
import yaml

from nerf_sampling.nerf_pytorch.utils import (
    load_obj_from_config,
    override_config,
    set_global_device,
)
from nerf_sampling.nerf_pytorch.loss_functions import SamplerLossInput

optuna.logging.set_verbosity(optuna.logging.DEBUG)


def objective(trial):
    """Run NeRF and sampling network training with provided configuration."""
    with open("./configs/lego.yaml", "r") as fin:
        model = "lego_sampler_module"
        config = yaml.safe_load(fin)[model]

    torch.manual_seed(42)  # 0

    # get names of environment variables

    set_global_device(config["kwargs"]["device"])
    EPOCHS = 150_000

    # N_samples = trial.suggest_int("N_samples", 2, 32)
    # sampler_train_frequency = trial.suggest_int("sampler_train_frequency", 2, 50)
    sampler_lr = trial.suggest_float("sampler_lr ", 1e-8, 1)
    # sampler_loss_weight = trial.suggest_float("sampler_loss_weight", 1e-8, 1)
    # n_layers = trial.suggest_int("n_layers", 3, 8)
    # layer_width = trial.suggest_categorical("layer_width", [128, 256, 512])
    # sampler_loss_input = trial.suggest_categorical(
    #     "sampler_loss_input",
    #     [
    #         SamplerLossInput.DENSITY.value,
    #         SamplerLossInput.ALPHAS.value,
    #         SamplerLossInput.WEIGHTS.value,
    #     ],
    # )

    override = {
        "N_importance": 128,
        "N_samples": 32,
        "sampler_loss_input": None,
        "sampler_lr": sampler_lr,
        "sampler_loss_weight": None,
        "sampler_train_frequency": None,
        "n_layers": 5,
        "layer_width": 128,
        "train_sampler_only": True,
    }
    override_config(config=config["kwargs"], update=override)

    group = "train_sampler_only_experiment"
    run = wandb.init(
        project="nerf-sampling",
        config=config["kwargs"],
        mode="online",
        group=group,
        reinit=True,
    )
    basedir = wandb.run.dir
    print(f"{basedir=}")
    datadir = "../dataset/lego"
    config["kwargs"]["datadir"] = datadir
    config["kwargs"]["basedir"] = basedir
    config["kwargs"]["trial"] = trial
    config["kwargs"][
        "ft_path"
    ] = "/home/mubuntu/Desktop/uni/implicit_representations/nerf-sampling/nerf_sampling/dataset/lego/pretrained_model/200000.tar"

    trainer = load_obj_from_config(cfg=config)
    psnr = trainer.train(N_iters=EPOCHS + 1)
    if wandb.run is not None:
        wandb.finish(quiet=True)

    return psnr


study_name = "train_sampler_only"
storage_name = study_name
study = optuna.create_study(
    direction="maximize",
    study_name=study_name,
    pruner=optuna.pruners.MedianPruner(),
    storage=f"sqlite:///{storage_name}.db",
    load_if_exists=True,
)
study.optimize(objective, n_trials=500)
