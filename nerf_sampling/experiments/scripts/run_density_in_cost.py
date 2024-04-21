"""Script for running SphereNeRF."""

import click
import torch
import yaml
from nerf_sampling.nerf_pytorch.utils import load_obj_from_config
import os


@click.command()
@click.option(
    "--hparams_path",
    help="Type of selected dataset",
    type=str,
    default="experiments/configs/lego.yaml",
)
@click.option("--model", help="Selected model", type=str, default="lego_sampler_module")
def main(
    hparams_path: str,
    model: str,
):
    """Main."""
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

    EPOCHS = 150000
    hparams["kwargs"]["density_in_loss"] = True

    samples = 64
    n_iter = 10
    density_loss_weight = 1e-3
    hparams["kwargs"]["N_samples"] = samples
    hparams["kwargs"]["sampling_train_frequency"] = n_iter
    hparams["kwargs"]["density_loss_weight"] = density_loss_weight
    expname = f"density_in_loss_samples_{samples}_weight_{density_loss_weight}_every_{n_iter}_iter"
    hparams["kwargs"]["expname"] = expname
    trainer = load_obj_from_config(cfg=hparams)
    trainer.train(N_iters=EPOCHS + 1)

    return


if __name__ == "__main__":
    main()
