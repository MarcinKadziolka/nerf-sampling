"""Script for running SphereNeRF."""

import click
import torch
import wandb
import yaml

from nerf_sampling.nerf_pytorch.utils import (
    load_obj_from_config,
    override_config,
    set_global_device,
)
from nerf_sampling.definitions import ROOT_DIR


@click.command()
@click.option(
    "-c",
    "--config",
    help="Path to configuration file.",
    type=str,
    default=f"{ROOT_DIR}/experiments/configs/lego.yaml",
    show_default=True,
)
@click.option(
    "-d",
    "--dataset",
    help="Path to dataset folder.",
    type=str,
    default=f"{ROOT_DIR}/dataset/drums",
    show_default=True,
)
@click.option(
    "-m",
    "--model",
    help="Model type.",
    type=str,
    default="lego_depth_net_module",
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
@click.option(
    "-ip",
    "--i_print",
    default=1000,
    help="Frequency of log printing.",
    show_default=True,
)
def main(**click_kwargs):
    """Run NeRF and sampling network training with provided configuration."""
    with open(click_kwargs["config"], "r") as fin:
        model = click_kwargs["model"]
        config = yaml.safe_load(fin)[model]
    config["kwargs"]["single_image"] = click_kwargs["single_image"]
    config["kwargs"]["single_ray"] = click_kwargs["single_ray"]
    config["kwargs"]["i_print"] = click_kwargs["i_print"]
    override = {
        "depth_net_lr": 1e-4,
        "n_layers": 10,
        "layer_width": 256,
        "train_depth_net_only": True,
        "sphere_radius": 2,
        "i_testset": 100,
    }

    override_config(config=config["kwargs"], update=override)

    torch.manual_seed(42)  # 0

    set_global_device(config["kwargs"]["device"])
    EPOCHS = 100_000

    print(f"wandb: {click_kwargs['wandb']}")
    wandb.init(
        project="nerf-sampling",
        config=config["kwargs"],
        mode=click_kwargs["wandb"],
        tags=[
            "train_depth_net_only",
            "bigger_network",
            "pretrained_model",
            "depth_z_vals_prediction",
            "single_point",
            "sphere_intersection",
            "drums",
            "DELETE",
        ],
    )
    basedir = wandb.run.dir
    print(f"{basedir=}")
    datadir = click_kwargs["dataset"]
    config["kwargs"]["datadir"] = datadir
    config["kwargs"]["basedir"] = basedir
    ft_path = f"{ROOT_DIR}/dataset/drums/pretrained_model/200000.tar"
    depth_net_path = None

    # ft_path = depth_net_path
    config["kwargs"]["ft_path"] = ft_path
    config["kwargs"]["depth_net_path"] = depth_net_path

    # for wandb purposes
    config["kwargs"]["sampling_mode"] = "depth_only"

    trainer = load_obj_from_config(cfg=config)
    psnr = trainer.train(N_iters=EPOCHS + 1)

    print(f"Final psnr: {psnr}")

    return


if __name__ == "__main__":
    main()
