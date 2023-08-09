"""Line module - contains a Line class, representing a line in 3D space."""

import torch
from torch.nn.functional import normalize

from sphere_nerf_mod.sphere import Sphere
from sphere_nerf_mod.utils import solve_quadratic_equation


class Line:
    """A line in 3D space, has an origin point and a direction vector.

    Public methods:
    - find_intersection_points_with_sphere
    - select_closest_point_to_origin
    """

    def __init__(self, origin: torch.Tensor(1, 3),
                 direction: torch.Tensor(1, 3)):
        """Initialize a line with an origin point and a direction vector."""
        self.origin = origin
        self.direction = normalize(direction)

    def find_intersection_points_with_sphere(
            self, sphere: Sphere) -> list[torch.Tensor(1, 3)]:
        """Find common points with a sphere.

        Solves the quadratic equation d^2 + bd + c = 0
        where solutions d are the distances from the line origin
        to the intersection points.
        """
        origin_to_sphere_center_vector = self.origin - sphere.center
        b = 2 * (torch.dot(self.direction, origin_to_sphere_center_vector))
        c = torch.norm(origin_to_sphere_center_vector) ** 2 - sphere.radius ** 2
        equation_solutions = solve_quadratic_equation(1, b, c)
        intersection_points = [torch.Tensor(self.origin + d * self.direction)
                               for d in equation_solutions]
        return intersection_points

    def select_closest_point_to_origin(
            self, points: list[torch.Tensor(1, 3)]) -> torch.Tensor(1, 3):
        """Select the point closest to the line origin."""
        distances_to_origin = [torch.norm(p - self.origin) for p in points]
        return points[distances_to_origin.index(min(distances_to_origin))]
