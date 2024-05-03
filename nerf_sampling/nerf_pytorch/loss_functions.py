"""Loss functions for sampler network."""

import torch


def mean_density_loss(density: torch.Tensor) -> torch.Tensor:
    """Calculate mean density."""
    return -torch.mean(density)


def max_density_loss(density: torch.Tensor) -> torch.Tensor:
    """Calculate mean of each ray max density."""
    return -torch.mean(torch.max(density, dim=1, keepdim=True)[0])
