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
        center=torch.Tensor(
            [
                [2, 0, 0],
                [0, 2, 0],
                [0, 2, 0],
            ]
        ),
        radius=torch.Tensor(
            [
                [1], [2], [1]
            ]
        )
    )

    hparams["kwargs"]["spheres"] = spheres

    trainer = load_obj_from_config(cfg=hparams)
    trainer.train()


if __name__ == "__main__":
    main()
