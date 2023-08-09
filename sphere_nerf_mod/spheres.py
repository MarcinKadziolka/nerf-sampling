"""Sphere module.

Contains a Sphere class, representing a sphere in 3D space.
"""

import torch


class Spheres:
    """A sphere in 3D space, has a center point and radius."""

    def __init__(
            self,
            center: torch.Tensor,
            radius: torch.Tensor
    ):
        """Initialize a spheres with a center points and radius.

        Args:
            center - torch.Tensor comprising of a shape [N, 3],
            where N represents the count of spheres, and 3 corresponds to
            a set of three coordinates defining a point in 3D space.
            radius - torch.Tensor comprising of a shape [N, 1].

        """
        self.center = center
        self.radius = radius
