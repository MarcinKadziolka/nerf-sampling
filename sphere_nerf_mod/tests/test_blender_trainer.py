"""Tests for blender trainer."""
import torch

from sphere_nerf_mod.lines import Lines
from sphere_nerf_mod.spheres import Spheres
from sphere_nerf_mod.trainers.blender_trainer import BlenderTrainer


rays = Lines(
    origin=torch.Tensor(
        [
            [0., 0, 0],
            [0., 1, 0],
            [2., 32, 0]
        ]
    ),
    direction=torch.Tensor(
        [
            [1., 0, 0],
            [1., 0, 0],
            [2., 0, 0]
        ]
    ),
)

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

blender_trainer = BlenderTrainer(spheres)


def test_sample_points():
    """Test sampling points."""
    # points = blender_trainer.sample_points(rays)
    # TODO assertions
