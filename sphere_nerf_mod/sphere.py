"""Sphere module.

Contains a Sphere class, representing a sphere in 3D space.
"""

import torch


class Sphere:
    """A sphere in 3D space, has a center point and radius."""

    def __init__(self, center: torch.Tensor(1, 3), radius: float):
        """Initialize a sphere with a center point and radius."""
        self.center = center
        self.radius = radius
