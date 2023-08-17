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

        :arg:
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
        Since we consider quadratic equation there is always two
        points (hence 2 points dim in return). It could be nan point.
        Args:
            sphere: Spheres objs. represented m_spheres.
        Return:
            Torch Tensor represented 3D points with dimension:
            [m_spheres, n_lines, 2 points, 3D]

        """
        # [n_lines, m_spheres, 3D]
        origin_to_sphere_center_vector = torch.unsqueeze(
            self.origin, 1
        ) - sphere.center

        # [n_lines, m_spheres]
        b = 2 * (torch.unsqueeze(
            self.direction, 1
        ) * origin_to_sphere_center_vector).sum(dim=2)

        # [n_lines, m_spheres]
        c = torch.norm(
            origin_to_sphere_center_vector, dim=2
        ) ** 2 - sphere.radius.T ** 2

        equation_solutions = solve_quadratic_equation(
            torch.ones_like(b), b, c
        )  # [2_points, n_lines, m_spheres]

        intersection_points = (torch.unsqueeze(
            self.origin, 2
        ) + torch.unsqueeze(
            equation_solutions.T, 2
        ) * torch.unsqueeze(
            self.direction, 2
        )).transpose(3, 2)

        return intersection_points  # [m_spheres, n_lines, 2 points, 3D]

    def select_closest_point_to_origin(
            self, points: torch.Tensor
    ) -> torch.Tensor:
        """Select the point closest to the line origin."""
        distances_to_origin = torch.norm(
            points - torch.unsqueeze(self.origin, 1),
            dim=3
        )

        n_lines = self.origin.shape[0]
        m_points = points.shape[0]
        _ones = torch.ones(m_points, n_lines)

        line_indexes = (
            torch.arange(0, self.origin.shape[0]) * _ones
        ).long()
        points_indexes = (
            torch.arange(0, m_points) * _ones.T
        ).T.long()

        indexes = torch.min(distances_to_origin, dim=2).indices
        selected_points = points[points_indexes, line_indexes, indexes]

        return selected_points  # [m_spheres, n_lines, 3D]

    def get_number(self):
        return self.origin.shape[0]