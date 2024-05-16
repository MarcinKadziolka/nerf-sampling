"""Script for running hyperparameters sweep."""

import optuna
import torch
import wandb
import yaml

from nerf_sampling.nerf_pytorch.utils import load_obj_from_config, override_config
from nerf_sampling.nerf_pytorch.loss_functions import SamplerLossInput

optuna.logging.set_verbosity(optuna.logging.DEBUG)


def objective(trial):
    """Run NeRF and sampling network training with provided configuration."""
    with open("./configs/lego.yaml", "r") as fin:
        model = "lego_sampler_module"
        config = yaml.safe_load(fin)[model]

    torch.manual_seed(42)  # 0

    # get names of environment variables

    if torch.cuda.is_available():
        torch.set_default_device(device="cuda")

    EPOCHS = 200_000

    N_samples = trial.suggest_int("N_samples", 2, 64)
    sampler_train_frequency = trial.suggest_int("sampler_train_frequency", 2, 50)
    sampler_lr = trial.suggest_float("sampler_lr ", 1e-6, 1e-1)
    sampler_loss_weight = trial.suggest_float("sampler_loss_weight", 1e-5, 1)
    n_layers = trial.suggest_int("n_layers", 3, 8)
    layer_width = trial.suggest_categorical("layer_width", [128, 256, 512])
    sampler_loss_input = trial.suggest_categorical(
        "sampler_loss_input",
        [
            SamplerLossInput.DENSITY,
            SamplerLossInput.ALPHAS,
            SamplerLossInput.WEIGHTS,
        ],
    )

    override = {
        "density_in_loss": True,
        "max_density": False,
        "N_samples": N_samples,
        "sampler_loss_input": sampler_loss_input,
        "sampler_lr": sampler_lr,
        "sampler_loss_weight": sampler_loss_weight,
        "sampler_train_frequency": sampler_train_frequency,
        "n_layers": n_layers,
        "layer_width": layer_width,
    }
    override_config(config=config["kwargs"], update=override)

    run = wandb.init(
        project="nerf-sampling",
        config=config["kwargs"],
        mode="online",
        group="optuna_testing",
        reinit=True,
    )
    basedir = wandb.run.dir
    print(f"{basedir=}")
    datadir = "../dataset/lego"
    config["kwargs"]["datadir"] = datadir
    config["kwargs"]["basedir"] = basedir
    config["kwargs"]["trial"] = trial

    trainer = load_obj_from_config(cfg=config)
    psnr = trainer.train(N_iters=EPOCHS + 1)
    if wandb.run is not None:
        wandb.finish(quiet=True)

    return psnr


study = optuna.create_study(
    direction="maximize",
    study_name="nerf_sampling",
    pruner=optuna.pruners.MedianPruner(),
)
study.optimize(objective, n_trials=100)
