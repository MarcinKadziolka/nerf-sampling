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
            [
                [2, 0, 0],
                [0, 2, 0],
                [0, 2, 0],
                [2, 0, 0],
                [2, 0, 0]
            ]
        ),
        radius=torch.Tensor(
            [
                [1], [2], [1], [1], [1]
            ]
        )
    )
    solutions = lines.find_intersection_points_with_sphere(sphere)
    print(solutions)
    # write assert TODO
    # assert solutions == true


def test_select_closest_point_to_origin():
    """Test selecting the closest point to line origin."""
    points = torch.rand(5, 4, 2, 3)  # [5 sphere, 4 lines, 2 solutions, 3D]
    points[0, 0] = torch.nan  # check what if point is nan
    points[3, 0, 0] = torch.nan
    solution = lines.select_closest_point_to_origin(points)
    print(solution)
    # TODO write assert


if __name__ == "__main__":
    test_find_intersection_points_with_sphere()
    test_select_closest_point_to_origin()
