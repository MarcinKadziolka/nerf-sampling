"""Tests for all line module functions."""

import torch

from sphere_nerf_mod.line import Line
from sphere_nerf_mod.sphere import Sphere


line = Line(torch.zeros(1, 3), torch.Tensor([[1, 0, 0]]))


def test_find_intersection_points_with_sphere():
    """Test finding intersection points with sphere."""
    sphere = Sphere(torch.Tensor([[2, 0, 0]]), 1)
    assert line.find_intersection_points_with_sphere(sphere) ==\
        [torch.Tensor([[1, 0, 0]]), torch.Tensor([[3, 0, 0]])]


def test_select_closest_point_to_origin():
    """Test selecting closest point to line origin."""
    points = [torch.Tensor([[1, 0, 0]]), torch.Tensor([[3, 0, 0]])]
    assert line.select_closest_point_to_origin(points) == points[0]
