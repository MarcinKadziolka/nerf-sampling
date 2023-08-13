"""Tests for all line module functions."""

import torch
import skspatial.objects

from sphere_nerf_mod.lines import Lines
from sphere_nerf_mod.spheres import Spheres

lines = Lines(
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


def test_find_intersection_points_with_sphere():
    """Test finding intersection points with sphere."""
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
    solutions = lines.find_intersection_points_with_sphere(spheres)
    number_of_spheres = spheres.center.shape[0]
    number_of_lines = lines.origin.shape[0]
    for sphere_index in range(number_of_spheres):
        single_sphere = skspatial.objects.Sphere(
            list(spheres.center[sphere_index]),
            spheres.radius[sphere_index][0]
        )
        for line_index in range(number_of_lines):
            single_line = skspatial.objects.Line(
                list(lines.origin[line_index]),
                list(lines.direction[line_index])
            )
            true_intersection_points = torch.Tensor(
                single_sphere.intersect_line(single_line))
            assert solutions[sphere_index][line_index].equal(
                true_intersection_points)


def test_select_closest_point_to_origin():
    """Test selecting the closest point to line origin."""
    points = torch.rand(3, 3, 2, 3)  # [3 sphere, 3 lines, 2 solutions, 3D]
    points[0, 0] = torch.nan  # check what if point is nan
    #points[3, 0, 0] = torch.nan
    solution = lines.select_closest_point_to_origin(points)
    print(solution)
    # TODO write assert


if __name__ == "__main__":
    test_find_intersection_points_with_sphere()
    test_select_closest_point_to_origin()
