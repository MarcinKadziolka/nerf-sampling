"""Tests for blender trainer."""
import torch

from sphere_nerf_mod.lines import Lines
from sphere_nerf_mod.spheres import Spheres
from sphere_nerf_mod.trainers.blender_trainer import SphereBlenderTrainer


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

blender_trainer = SphereBlenderTrainer(spheres)


def test_sample_points_on_spheres():
    """Test sampling points."""
    point_coordinate_if_nan = 100
    points = blender_trainer.sample_points_on_spheres(
        rays=rays,
        point_coordinate_if_nan=point_coordinate_if_nan
    )
    correct_result = torch.Tensor(
        [
            [
                [1, 0, 0],
                [2, 1, 0],
                [100, 100, 100]
            ],
            [
                [0, 0, 0],
                [-1.7321, 1, 0],
                [100, 100, 100]
            ],
            [
                [100, 100, 100],
                [0, 1, 0],
                [100, 100, 100]
            ]
        ]
    )
    assert torch.isclose(points, correct_result, rtol=1e-04).all()


if __name__ == "__main__":
    test_sample_points_on_spheres()
