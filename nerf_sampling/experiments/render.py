"""Script for rendering depth net results."""

import os

import click
import torch
import wandb
import yaml

from nerf_sampling.definitions import ROOT_DIR
from nerf_sampling.nerf_pytorch.utils import (
    RenderingMode,
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
    default=f"{ROOT_DIR}/experiments/configs/lego.yaml",
    show_default=True,
)
@click.option(
    "-dp",
    "--dataset_path",
    help="Path to dataset folder.",
    type=str,
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
    "-ssd",
    "--save_scene_data",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "-rd",
    "--rendering_mode",
    type=click.Choice(
        ["depth", "ndepth", "compare", "max", "smax", "full"], case_sensitive=False
    ),
    help="""
    \b
    depth: Use novel depth network for point prediciton.
    ndepth: Use NeRF depth map for single point.
    compare: Get MSE between depth and max nerf.
    max: Use most important NeRF point for rendering.
    smax: Sample around most important NeRF point for rendering.
    full: Use all NeRF points for rendering.
    """,
    default="depth",
    show_default=True,
)
@click.option(
    "-e",
    "--experiments",
    is_flag=True,
    default=False,
    help="Use automatic experiments.",
    show_default=True,
)
@click.option(
    "-tmp",
    "--temporary",
    is_flag=True,
    default=False,
    help="Use temporary folder for experiment.",
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

    config["kwargs"]["save_scene_data"] = click_kwargs["save_scene_data"]
    config["kwargs"]["i_print"] = click_kwargs["i_print"]

    rendering_mode = click_kwargs["rendering_mode"]
    config["kwargs"]["rendering_mode"] = RenderingMode[rendering_mode.upper()]

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

    datadir = click_kwargs["dataset_path"]
    ft_path = None
    depth_net_path = None
    if (dataset_name := click_kwargs["dataset"]) is not None:
        datadir = f"{ROOT_DIR}/dataset/{dataset_name}"
        ft_path = f"{ROOT_DIR}/pretrained/nerf/{dataset_name}/200000.tar"
        depth_net_path = f"{ROOT_DIR}/pretrained/depth_net/{dataset_name}/files/sampler_experiment/200000.tar"
        print(f"{dataset_name=}")
    if datadir is None:
        print(
            "Please specify the name of the dataset or provide the path to the folder"
        )
        return

    wandb.run.dir = f"./logs/{dataset_name}"
    basedir = wandb.run.dir
    print(f"{basedir=}")

    set_global_device(config["kwargs"]["device"])
    EPOCHS = 100_000_000

    override = {
        "depth_net_lr": 1e-4,
        "n_layers": 10,
        "layer_width": 256,
        "train_depth_net_only": True,
        "sphere_radius": 2,
    }
    override_config(config=config["kwargs"], update=override)

    torch.manual_seed(42)  # 0

    config["kwargs"]["datadir"] = datadir
    config["kwargs"]["basedir"] = basedir

    config["kwargs"]["ft_path"] = ft_path
    config["kwargs"]["depth_net_path"] = depth_net_path

    # manual values
    n_samples = 2
    distance = 0.01
    # n_samples = None
    # distance = None
    sampling_mode = "uniform"  # uniform, gaussian, depth_only

    if rendering_mode == "depth":
        config["kwargs"][
            "expname"
        ] = f"{dataset_name}_depth_net_render_n_samples_{n_samples}_distance_{distance}_sampling_mode_{sampling_mode}"
    else:
        config["kwargs"][
            "expname"
        ] = f"{dataset_name}_{click_kwargs['rendering_mode']}_render"
    if click_kwargs["temporary"]:
        config["kwargs"]["expname"] = "tmp"

    config["kwargs"]["n_depth_samples"] = n_samples
    config["kwargs"]["distance"] = distance
    config["kwargs"]["sampling_mode"] = sampling_mode

    if click_kwargs["experiments"]:
        wandb.run.dir = f"./logs/{dataset_name}/experiments"
        basedir = wandb.run.dir
        print(f"{basedir=}")
        os.makedirs(basedir, exist_ok=True)
        n_samples_list = [2, 32, 64, 128]
        distances = [0.1, 0.3, 0.5, 1]
        sampling_modes = ["uniform", "gaussian"]
        f = os.path.join(basedir, "experiments_results.txt")
        with open(f, "w") as file:
            file.write(f"Experiments")
        for sampling_mode in sampling_modes:
            config["kwargs"]["basedir"] = os.path.join(basedir, sampling_mode)
            with open(f, "a") as file:
                file.write(f"\n\nSampling mode: {sampling_mode}\n\n")
            for n_samples in n_samples_list:
                with open(f, "a") as file:
                    file.write(f"N_samples: {n_samples}:\n")
                for distance in distances:
                    config["kwargs"][
                        "expname"
                    ] = f"{dataset_name}_depth_net_render_n_samples_{n_samples}_distance_{distance}_sampling_mode_{sampling_mode}"
                    config["kwargs"]["n_depth_samples"] = n_samples
                    config["kwargs"]["distance"] = distance
                    config["kwargs"]["sampling_mode"] = sampling_mode
                    trainer = load_obj_from_config(cfg=config)
                    psnr = trainer.train(N_iters=EPOCHS + 1)
                    with open(f, "a") as file:
                        file.write(f"    Distance: {distance}, PSNR: {psnr:.2f}\n")
        return

    trainer = load_obj_from_config(cfg=config)
    psnr = trainer.train(N_iters=EPOCHS + 1)

    print(f"Final psnr: {psnr}")

    return


if __name__ == "__main__":
    main()
