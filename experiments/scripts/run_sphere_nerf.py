"""Script for run sphere nerf."""
import click
import torch
import yaml

from nerf_pytorch.run_nerf import train

torch.set_default_tensor_type('torch.cuda.FloatTensor')


@click.command()
@click.option(
    "--hparams_path",
    help="Type of selected dataset",
    type=str,
    default="experiments/configs/normalized.yaml"
)
@click.option(
    "--model",
    help="Selected model",
    type=str,
    default="airplane"
)
def main(
    hparams_path: str,
    model: str,
):
    """Main."""
    with open(hparams_path, "r") as fin:
        hparams = yaml.safe_load(fin)[model]

    train(
        half_res=hparams["white_bkgd"],
        no_batching=hparams["no_batching"],
        N_samples=hparams["N_samples"],
        N_importance=hparams["N_importance"],
        use_viewdirs=hparams["use_viewdirs"],
        white_bkgd=hparams["white_bkgd"],
        device=hparams["device"],
        N_rand=hparams["N_rand"],
        expname=hparams["data"]["expname"],
        basedir=hparams["data"]["basedir"],
        datadir=hparams["data"]["datadir"],
        dataset_type=hparams["data"]["dataset_type"],
        config_path=hparams_path
    )


if __name__ == "__main__":
    main()
