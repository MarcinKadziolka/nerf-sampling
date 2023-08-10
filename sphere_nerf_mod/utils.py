"""Utils module - contains useful low-level utility functions."""

import math

import torch


def solve_quadratic_equation(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor
):
    """Solve quadratic equation ax^2 + bx + c = 0.

    Return nan if solution does not exist.
    """
    delta = b ** 2 - 4 * a * c  # [n_lines, m_sphere]

    # To compute minus/plus [2 solutions, n_lines, m_sphere]
    pm = torch.stack([torch.ones_like(delta), -torch.ones_like(delta)])

    _ones = torch.ones_like(delta)

    sqrt_delta = torch.sqrt(delta)
    solutions = (-b - (pm * sqrt_delta)) / (2 * a)

    only_one_solution = torch.where(delta == 0, _ones * torch.nan, _ones)
    only_one_solution = only_one_solution
    solutions[1] = only_one_solution * solutions[1]

    return solutions
