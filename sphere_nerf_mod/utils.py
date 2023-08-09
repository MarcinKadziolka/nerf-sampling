"""Utils module - contains useful low-level utility functions."""

import math

import torch


def solve_quadratic_equation(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor
):
    """Solve quadratic equation ax^2 + bx + c = 0."""

    solutions = torch.ones(2, *b.shape)  # prepare solution result [2, n]

    delta = b ** 2 - 4 * a * c
    pm = torch.Tensor([1, -1]).repeat(delta.shape[0], 1)  # To compute minus/plus [n, 2]

    idxs = torch.where(delta < 0)[0]
    if len(idxs) != 0:
        solutions[:, idxs] = torch.ones_like(solutions[:, idxs]) * torch.nan

    idxs = torch.where(delta == 0)[0]
    if len(idxs) != 0:
        solutions[1, idxs] = torch.ones_like(solutions[1, idxs]) * torch.nan
        solutions[0, idxs] = -b[idxs] / (2 * a[idxs])

    idxs = torch.where(delta > 0)[0]
    sqrt_delta = torch.sqrt(delta[idxs])
    solutions_delta_positive = (-b[idxs] - (pm[idxs].T * sqrt_delta)) / (2 * a[idxs])
    solutions[:, idxs] = solutions_delta_positive

    return solutions
