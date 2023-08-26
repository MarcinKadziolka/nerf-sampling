"""Lines module - contains a Lines class, representing lines in 3D space."""

import torch
from torch.nn.functional import normalize

from sphere_nerf_mod.spheres import Spheres
from sphere_nerf_mod.utils import solve_quadratic_equation


class Lines:
    """Lines in 3D space, each has an origin point and a direction vector."""

    def __init__(
            self,
            origin: torch.Tensor,
            direction: torch.Tensor
    ):
        """Initialize lines with origin points and direction vectors.

        Each line has a corresponding origin point and a direction vector.
        Args:
            origin: torch.Tensor with shape (N, 3),
             where N is the number of lines, and 3 corresponds to
             a set of three coordinates defining a point in 3D space.
            direction: torch.Tensor with shape (N, 3).

        """
        self.origin = origin
        self.direction = direction

    def find_intersection_points_with_sphere(
            self, sphere: Spheres
    ) -> torch.Tensor:
        """Find intersection points with spheres.

        Finds intersection points of the lines with the given spheres.
        Returns nan values if a given line and sphere do not intersect.
        Args:
            sphere: Spheres - the spheres checked for intersection points.
        Return:
            torch.Tensor containing the intersection points and with shape
            [m_spheres, n_lines, 2 points, 3D].

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
        """Select the point closest to the line origin.

        For each set of points chooses the closest point to the origin
        of the corresponding line.
        Args:
            points: torch.Tensor of points to select from.
             The tensor has the shape (n_sets, n_lines, n_points, 3), where
             n_sets - the number of sets of points, for each of which the
             operation is performed.
             n_lines - the number of lines.
             n_points - the number of points for each line, out of which only
             the closest is returned.
        Return:
            Tensor with shape (n_sets, n_lines, 3).

        """
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
        """Return the number of lines."""
        return self.origin.shape[0]

    def transform_points_to_single_number_representation(
            self,
            points
    ) -> torch.Tensor:
        """Transform points to a representation based on the line parameters.

        Transforms points on lines, represented by
        absolute coordinates in 3D space to a single number form.
        For a 3D point `p` on a line `l` the returned representation
        of the point is the number `r`, such that:
        `p = l.origin + r * l.direction`.
        Args:
            points: 3D points to transform. shape should be (rays, points, 3).
        Return:
            Tensor with transformed points, with shape (rays, points).

        """
        result = (points - torch.unsqueeze(self.origin, 1)) \
            / torch.unsqueeze(self.direction, 1)
        return torch.nanmean(result, dim=2)
