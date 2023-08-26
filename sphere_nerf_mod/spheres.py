"""Spheres module.

Contains a Spheres class, representing spheres in 3D space.
"""

import torch


class Spheres:
    """Spheres in 3D space, each has a center point and radius."""

    def __init__(
            self,
            center: torch.Tensor,
            radius: torch.Tensor
    ):
        """Initialize spheres with center points and radii.

        Args:
            center: torch.Tensor with shape (N, 3),
             where N is the number of spheres, and 3 corresponds to
             a set of three coordinates defining a point in 3D space.
            radius: torch.Tensor with shape (N, 1).

        """
        self.center = center
        self.radius = radius

        if center.shape[1] != 3:
            raise f"Center has to be a 3D point." \
                  f" Found {center.shape[1]} shape instead."

        if radius.shape != torch.Size([center.shape[0], 1]):
            raise ValueError(
                f"""Please check radius dimensions.
                Found {radius.shape},
                expected: {torch.Size([center.shape[0], 1])}."""
            )

    def get_number(self):
        """Return the number of spheres."""
        return self.center.shape[0]
