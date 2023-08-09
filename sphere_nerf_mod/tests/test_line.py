"""Tests for all line module functions."""

import torch

from sphere_nerf_mod.lines import Lines
from sphere_nerf_mod.spheres import Spheres


lines = Lines(
    origin=torch.Tensor(
        [
            [0., 0, 0],
            [0., 0, 0],
            [0., 1, 0],
            [2., 32, 0]
        ]
    ),
    direction=torch.Tensor(
        [
            [1., 0, 0],
            [1., 0, 0],
            [1., 0, 0],
            [2., 0, 0]
        ]
    ),
)


def test_find_intersection_points_with_sphere():
    """Test finding intersection points with sphere."""
    sphere = Spheres(
        center=torch.Tensor(
            [[2, 0, 0]]
        ),
        radius=torch.Tensor(
            [[1]]
        )
    )
    solutions = lines.find_intersection_points_with_sphere(sphere)
    true = torch.Tensor(
            [[1., 0, 0], [3., 0, 0]]
        ) # to nie jest true na razie TODO
    assert solutions == true


def test_select_closest_point_to_origin():
    """Test selecting the closest point to line origin."""
    points = [torch.Tensor([[1, 0, 0]]), torch.Tensor([[3, 0, 0]])]
    assert lines.select_closest_point_to_origin(points) == points[0]


if __name__ == "__main__":
    test_find_intersection_points_with_sphere()
    test_select_closest_point_to_origin()
