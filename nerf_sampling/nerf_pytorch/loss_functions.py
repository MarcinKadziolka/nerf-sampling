"""Loss functions for sampler network."""

import torch
from enum import Enum


def alphas_or_weights_loss(alphas_or_weights: torch.Tensor) -> torch.Tensor:
    """Calculate loss of alphas or weights.

    Because alphas and weights are in the range [0, 1],
    we can construct the loss function in such way
    that minimizing it will equal cost to 0.
    """
    return 1 - torch.mean(alphas_or_weights)


def mean_density_loss(density: torch.Tensor) -> torch.Tensor:
    """Calculate mean density.

    Because we want to maximize density,
    returned value has minus sign,
    so that loss can be minimized.
    """
    return -torch.mean(density)


def max_density_loss(density: torch.Tensor) -> torch.Tensor:
    """Calculate mean of each ray max density."""
    return -torch.mean(torch.max(density, dim=1, keepdim=True)[0])
