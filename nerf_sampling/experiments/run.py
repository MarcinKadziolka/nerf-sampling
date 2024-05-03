"""Script for running SphereNeRF."""

import click
import torch
import wandb
import yaml

from nerf_sampling.nerf_pytorch.utils import load_obj_from_config, override_config


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

    torch.manual_seed(42)  # 0

    # get names of environment variables

    if config["kwargs"]["device"] == "cuda":
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

    EPOCHS = 7000000

    override = {
        "density_in_loss": True,
        "max_density": False,
        "N_samples": 64,
        "sampler_lr": 1e-4,
        "sampler_loss_weight": 1e-3,
        "sampler_train_frequency": 10,
    }
    override_config(config=config["kwargs"], update=override)

    print(f"wandb: {click_kwargs['wandb']}")
    wandb.init(
        project="nerf-sampling", config=config["kwargs"], mode=click_kwargs["wandb"]
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
