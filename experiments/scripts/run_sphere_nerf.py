"""Script for run sphere nerf."""
import click
import torch
import yaml
from nerf_pytorch.utils import load_obj_from_config

from sphere_nerf_mod.spheres import Spheres

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
    default="airplane_sphere_module"
)
def main(
        hparams_path: str,
        model: str,
):
    """Main."""
    with open(hparams_path, "r") as fin:
        hparams = yaml.safe_load(fin)[model]

    spheres = Spheres(
        center=torch.zeros((64, 3)),
        radius=torch.vstack((((torch.range(1, 63) / 100).reshape(-1, 1) + 2.2),
                            torch.Tensor([[10]])))
    )

    '''spheres = Spheres(
        center=torch.zeros((64, 3)),
        radius=torch.Tensor((torch.range(1, 64) / 100).reshape(-1, 1)) + 2.2
    )'''

    hparams["kwargs"]["spheres"] = spheres

    trainer = load_obj_from_config(cfg=hparams)
    trainer.train(N_iters=1000000)


if __name__ == "__main__":
    main()
