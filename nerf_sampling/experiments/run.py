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
@click.option(
    "-si",
    "--single_image",
    is_flag=True,
    default=False,
    help="Train sampling network on single image.",
    show_default=True,
)
@click.option(
    "-sr",
    "--single_ray",
    is_flag=True,
    default=False,
    help="Train sampling network on single ray.",
    show_default=True,
)
def main(**click_kwargs):
    """Run NeRF and sampling network training with provided configuration."""
    with open(click_kwargs["config"], "r") as fin:
        model = click_kwargs["model"]
        config = yaml.safe_load(fin)[model]
    config["kwargs"]["single_image"] = click_kwargs["single_image"]
    config["kwargs"]["single_ray"] = click_kwargs["single_ray"]
    override = {
        "N_sampler_samples": 2,
        "distance": 0.1,
        "N_samples": 64,
        "N_importance": 128,
        "sampler_lr": 1e-3,
        "n_layers": 5,
        "layer_width": 128,
        "train_sampler_only": True,
    }

    override_config(config=config["kwargs"], update=override)

    torch.manual_seed(42)  # 0

    set_global_device(config["kwargs"]["device"])
    EPOCHS = 100_000_000

    print(f"wandb: {click_kwargs['wandb']}")
    wandb.init(
        project="nerf-sampling",
        config=config["kwargs"],
        mode=click_kwargs["wandb"],
        tags=[
            "sampler_only",
            "pretrained_model",
            "z_vals",
            "uniform_grid",
            "LL_gaussian_log_likelihood",
        ],
    )
    basedir = wandb.run.dir
    print(f"{basedir=}")
    datadir = click_kwargs["dataset"]
    config["kwargs"]["datadir"] = datadir
    config["kwargs"]["basedir"] = basedir
    ft_path = "/home/mubuntu/Desktop/uni/implicit_representations/nerf-sampling/nerf_sampling/dataset/lego/pretrained_model/200000.tar"
    sampler_path = None
    config["kwargs"]["ft_path"] = ft_path
    config["kwargs"]["sampler_path"] = sampler_path

    trainer = load_obj_from_config(cfg=config)
    trainer.train(N_iters=EPOCHS + 1)

    return


if __name__ == "__main__":
    main()
