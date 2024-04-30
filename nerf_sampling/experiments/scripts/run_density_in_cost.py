"""Script for running SphereNeRF."""

import os
from typing import Literal

import click
import torch
import wandb
import yaml

from nerf_sampling.nerf_pytorch.utils import load_obj_from_config


@click.command()
@click.option(
    "--hparams_path",
    help="Path to configuration file.",
    type=str,
    default="experiments/configs/lego.yaml",
)
@click.option("--model", help="Model type.", type=str, default="lego_sampler_module")
@click.option(
    "--wandb_mode",
    type=click.Choice(["online", "offline", "disabled"], case_sensitive=False),
    default="online",
    help="Set the mode for wandb logging.",
)
def main(
    hparams_path: str,
    wandb_mode: Literal["online", "offline", "disabled"],
    model: str,
):
    """Run NeRF and sampling network training with provided configuration."""
    with open(hparams_path, "r") as fin:
        hparams = yaml.safe_load(fin)[model]

    torch.manual_seed(42)  # 0

    # get names of environment variables
    datadir = os.environ.get("DATADIR", "./dataset/lego")
    basedir = os.environ.get("BASEDIR", "./current_logs")

    hparams["kwargs"]["datadir"] = datadir
    hparams["kwargs"]["basedir"] = basedir

    if hparams["kwargs"]["device"] == "cuda":
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

    EPOCHS = 7000000
    hparams["kwargs"]["density_in_loss"] = True

    samples = 64
    n_iter = 10
    density_loss_weight = 1e-3
    max_density = False
    hparams["kwargs"]["N_samples"] = samples
    hparams["kwargs"]["max_density"] = max_density
    hparams["kwargs"]["sampling_train_frequency"] = n_iter
    hparams["kwargs"]["density_loss_weight"] = density_loss_weight
    expname = f"density_in_loss_samples_{samples}_weight_{density_loss_weight}_every_{n_iter}_iter_max_{max_density}"
    hparams["kwargs"]["expname"] = expname

    print(f"{wandb_mode=}")
    wandb.init(project="nerf-sampling", config=hparams["kwargs"], mode=wandb_mode)
    trainer = load_obj_from_config(cfg=hparams)
    trainer.train(N_iters=EPOCHS + 1)

    return


if __name__ == "__main__":
    main()
