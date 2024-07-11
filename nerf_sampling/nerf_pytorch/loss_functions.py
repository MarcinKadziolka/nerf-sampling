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


def gaussian_log_likelihood(x, u, s):
    term1 = torch.log(torch.tensor(2, device=torch.device("cpu")) * torch.pi * s**2)
    term2 = ((x - u) ** 2) / (s**2)
    return torch.sum(0.5 * (term1 + term2))


class SamplerLossInput(Enum):
    """Store options for sampler loss function input."""

    DENSITY = 0
    ALPHAS = 1
    WEIGHTS = 2
