"""Sphere module.

Contains a Sphere class, representing a sphere in 3D space.
"""

import torch


class Spheres:
    """Spheres in 3D space, each have a center point and radius."""

    def __init__(
            self,
            center: torch.Tensor,
            radius: torch.Tensor
    ):
        """Initialize a spheres with a center points and radius.

        :arg:
            center - torch.Tensor comprising of a shape [N, 3],
            where N represents the count of spheres, and 3 corresponds to
            a set of three coordinates defining a point in 3D space.
            radius - torch.Tensor comprising of a shape [N, 1].

        """
        self.center = center
        self.radius = radius

        if center.shape[1] != 3:
            raise f"Center has to be 3D point. Find {center.shape[1]} dim."

        if radius.shape != torch.Size([center.shape[0], 1]):
            raise ValueError(
                f"""Please check radius dimensions.
                Find {radius.shape} but it should be
                {torch.Size([center.shape[0], 1])}."""
            )
