"""Script for running SphereNeRF."""

import click
import torch
import wandb
import yaml

from nerf_sampling.nerf_pytorch.loss_functions import SamplerLossInput
from nerf_sampling.nerf_pytorch.utils import (
    load_obj_from_config,
    override_config,
    set_global_device,
)


@click.command()
@click.option(
    "-c",
    "--config",
    help="Path to configuration file.",
    type=str,
    default="experiments/configs/lego.yaml",
    show_default=True,
)
@click.option(
    "-d",
    "--dataset",
    help="Path to dataset folder.",
    type=str,
    default="./dataset/lego",
    show_default=True,
)
@click.option(
    "-m",
    "--model",
    help="Model type.",
    type=str,
    default="lego_sampler_module",
    show_default=True,
)
@click.option(
    "-w",
    "--wandb",
    type=click.Choice(["online", "offline", "disabled"], case_sensitive=False),
    default="disabled",
    help="Set the mode for wandb logging.",
    show_default=True,
)
def main(**click_kwargs):
    """Run NeRF and sampling network training with provided configuration."""
    with open(click_kwargs["config"], "r") as fin:
        model = click_kwargs["model"]
        config = yaml.safe_load(fin)[model]
    override = {}
    override_config(config=config["kwargs"], update=override)

    torch.manual_seed(42)  # 0

    set_global_device(config["kwargs"]["device"])
    EPOCHS = 100_000_000

    print(f"wandb: {click_kwargs['wandb']}")
    wandb.init(
        project="nerf-sampling",
        config=config["kwargs"],
        mode=click_kwargs["wandb"],
    )
    basedir = wandb.run.dir
    print(f"{basedir=}")
    datadir = click_kwargs["dataset"]
    config["kwargs"]["datadir"] = datadir
    config["kwargs"]["basedir"] = basedir

    trainer = load_obj_from_config(cfg=config)
    trainer.train(N_iters=EPOCHS + 1)

    return


if __name__ == "__main__":
    main()
