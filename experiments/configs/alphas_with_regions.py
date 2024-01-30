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
    default="experiments/configs/lego_base.yaml"
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

    # Whether to use regions when alphas are in loss
    hparams['kwargs']['use_regions'] = True # Split ray into regions
    hparams['kwargs']['use_noise'] = False

    EPOCHS = 150000

    # With alphas
    hparams['kwargs']['use_alphas_in_loss'] = True
    for samples in [8, 16, 32]:
        hparams['kwargs']['N_samples'] = samples
        for alphas_weight in [0.01, 0.1, 1]:

            expname = f'alphas_samples={samples}_weight={alphas_weight}'
            hparams['kwargs']['expname'] = expname
            hparams['kwargs']['alphas_loss_weight'] = alphas_weight
            trainer = load_obj_from_config(cfg=hparams)
            trainer.train(N_iters=EPOCHS+1)
            

if __name__ == "__main__":
    main()
