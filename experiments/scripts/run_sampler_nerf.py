"""Script for running SphereNeRF."""
import click
import torch
import yaml
from nerf_pytorch.utils import load_obj_from_config
import os

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


@click.command()
@click.option(
    "--hparams_path",
    help="Type of selected dataset",
    type=str,
    default="experiments/configs/lego.yaml"
)
@click.option(
    "--model",
    help="Selected model",
    type=str,
    default="lego_sampler_module"
)
def main(
        hparams_path: str,
        model: str,
):
    """Main."""
    with open(hparams_path, "r") as fin:
        hparams = yaml.safe_load(fin)[model]

    torch.manual_seed(42)  # 0

    # get names of environment variables
    datadir = os.environ.get('DATADIR')
    basedir = os.environ.get('BASEDIR')

    hparams['kwargs']['datadir'] = datadir
    hparams['kwargs']['basedir'] = basedir

    trainer = load_obj_from_config(cfg=hparams)
    trainer.train(N_iters=700001)


if __name__ == "__main__":
    main()
