"""Script for running SphereNeRF."""
import click
import torch
import yaml
from nerf_pytorch.utils import load_obj_from_config
import os

EPOCHS = 200000

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
    hparams['kwargs']['use_regions'] = False
    hparams['kwargs']['use_noise'] = False


    # Use regions when alphas are in loss
    hparams['kwargs']['use_regions'] = False
    hparams['kwargs']['use_alphas_in_loss'] = False
    hparams['kwargs']['alphas_loss_weight'] = 0
    hparams['kwargs']['use_noise'] = False

    hparams['kwargs']['use_summing'] = True

    for increase_after in [500, 2500, 5000]:
        for n_rand in [1024, 2048]:
            for n_samples, max_group in [(128, 16), (64, 8)]:
                hparams['kwargs']['expname'] = f'[SUMMING] n_samples={n_samples} max_group={max_group} n_rand={n_rand} increase_after={increase_after}'
                hparams['kwargs']['max_group_size'] = max_group
                hparams['kwargs']['N_samples'] = n_samples
                hparams['kwargs']['increase_group_size_after'] = increase_after
                hparams['kwargs']['N_rand'] = n_rand

                trainer = load_obj_from_config(cfg=hparams)
                trainer.train(N_iters=EPOCHS+1)
            

if __name__ == "__main__":
    main()
