"""Script for running SphereNeRF."""
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

    number_of_spheres = 3
    smallest_sphere_radius = 2
    distance_between_spheres = 1 / 40

    spheres = Spheres(
        center=torch.zeros((number_of_spheres, 3)),
        radius=(
            (torch.range(1, number_of_spheres) * distance_between_spheres)
            .reshape(-1, 1)
            + (smallest_sphere_radius - 1)
        )
    )
    hparams["kwargs"]["spheres"] = spheres

    trainer = load_obj_from_config(cfg=hparams)
    trainer.train(N_iters=10001)


if __name__ == "__main__":
    main()
