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
    "-dp",
    "--dataset_path",
    help="Path to dataset folder.",
    type=str,
    default=f"{ROOT_DIR}/dataset/drums",
    show_default=True,
)
@click.option(
    "-d",
    "--dataset",
    help="Name of the dataset to render.",
    type=str,
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
    "-rt",
    "--render_test",
    is_flag=True,
    default=False,
    help="Perform render test",
    show_default=True,
)
@click.option(
    "-po",
    "--plot_object",
    is_flag=True,
    default=False,
    help="Save plot of object during rendering test. This option only applies when --render_test is enabled.",
    show_default=True,
)
@click.option(
    "-cn",
    "--compare_nerf",
    is_flag=True,
    default=False,
    help="Compare depth network predictions to the original NeRF most important samples.",
    show_default=True,
)
@click.option(
    "-nm",
    "--nerf_max",
    is_flag=True,
    default=False,
    help="Use nerf max points to render",
    show_default=True,
)
@click.option(
    "-nf",
    "--nerf_full",
    is_flag=True,
    default=False,
    help="Use full nerf to render",
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
    config["kwargs"]["plot_object"] = click_kwargs["plot_object"]
    config["kwargs"]["i_print"] = click_kwargs["i_print"]
    config["kwargs"]["compare_nerf"] = click_kwargs["compare_nerf"]
    config["kwargs"]["use_nerf_max_pts"] = click_kwargs["nerf_max"]
    config["kwargs"]["use_full_nerf"] = click_kwargs["nerf_full"]
    config["kwargs"]["render_only"] = True
    config["kwargs"]["render_test"] = True

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
        ],
    )

    basedir = wandb.run.dir
    print(f"{basedir=}")
    datadir = click_kwargs["dataset_path"]
    ft_path = None
    depth_net_path = None
    if (dataset_name := click_kwargs["dataset"]) is not None:
        datadir = f"{ROOT_DIR}/dataset/{dataset_name}"
        ft_path = f"{ROOT_DIR}/dataset/{dataset_name}/pretrained_model/200000.tar"
        depth_net_path = f"{ROOT_DIR}/pretrained_depth_nets/{dataset_name}/files/sampler_experiment/100000.tar"

    config["kwargs"]["datadir"] = datadir
    config["kwargs"]["basedir"] = basedir

    config["kwargs"]["ft_path"] = ft_path
    config["kwargs"]["depth_net_path"] = depth_net_path

    config["kwargs"]["n_depth_samples"] = 128
    config["kwargs"]["distance"] = 1
    config["kwargs"]["sampling_mode"] = "depth_only"

    override = {
        "depth_net_lr": 1e-4,
        "n_layers": 10,
        "layer_width": 256,
        "train_depth_net_only": True,
        "sphere_radius": 2,
    }

    override_config(config=config["kwargs"], update=override)

    torch.manual_seed(42)  # 0

    set_global_device(config["kwargs"]["device"])
    EPOCHS = 100_000_000

    trainer = load_obj_from_config(cfg=config)
    psnr = trainer.train(N_iters=EPOCHS + 1)

    print(f"Final psnr: {psnr}")

    return


if __name__ == "__main__":
    main()
