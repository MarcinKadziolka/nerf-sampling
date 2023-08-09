"""Line module - contains a Line class, representing a line in 3D space."""

import torch
from torch.nn.functional import normalize

from sphere_nerf_mod.spheres import Spheres
from sphere_nerf_mod.utils import solve_quadratic_equation


class Lines:
    """A line in 3D space, has an origin point and a direction vector."""

    def __init__(
            self,
            origin: torch.Tensor,
            direction: torch.Tensor
    ):
        """Initialize a lines with an origin point and a direction vector.

        Args:
            origin - torch.Tensor comprising of a shape [N, 3],
            where N represents the count of lines, and 3 corresponds to
            a set of three coordinates defining a point in 3D space.
            direction - torch.Tensor comprising a shape [N, 3] it is
            direction vectors. Direction vectors is always normalized.
        """
        self.origin = origin
        self.direction = normalize(direction)

    def find_intersection_points_with_sphere(
            self, sphere: Spheres
    ) -> torch.Tensor:
        """Find common points with a sphere.

        Solves the quadratic equation d^2 + bd + c = 0
        where solutions d are the distances from the line origin
        to the intersection points. Then find intersection_points.
        """
        origin_to_sphere_center_vector = self.origin - sphere.center
        b = 2 * (self.direction * origin_to_sphere_center_vector).sum(dim=1)
        c = torch.norm(origin_to_sphere_center_vector, dim=1) ** 2 - sphere.radius ** 2
        equation_solutions = solve_quadratic_equation(
            torch.ones_like(b), b, torch.squeeze(c)
        )
        intersection_points = torch.unsqueeze(
            self.origin, 1
        ) + torch.unsqueeze(
            equation_solutions.T, 2
        ) * torch.unsqueeze(
            self.direction, 1
        )
        return intersection_points  # [n_lines, 2 points, 3D]

    def select_closest_point_to_origin(
            self, points: list[torch.Tensor(1, 3)]) -> torch.Tensor(1, 3):
        """Select the point closest to the line origin."""
        distances_to_origin = [torch.norm(p - self.origin) for p in points]
        return points[distances_to_origin.index(min(distances_to_origin))]
