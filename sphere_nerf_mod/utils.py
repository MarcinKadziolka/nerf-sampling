"""Utils module - contains useful low-level utility functions."""
import torch
import numpy as np


def solve_quadratic_equation(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor
):
    """Solve quadratic equation ax^2 + bx + c = 0.

    Solves the quadratic equation with tensor coefficients.
    Returns nan if solution does not exist.
    The arguments can be of any shape, as long as their shapes are the same.

    For the argument shape `(x1, ..., xn)` the returned tensor
    has the shape (2, x1, ..., xn).
    For the equation specified by arguments' value at [x1, ..., xn],
    its' solutions are at returned tensor's [0, x1, ..., xn], [1, x1, ..., xn].
    """
    delta = b ** 2 - 4 * a * c  # [n_lines, m_sphere]

    # To compute minus/plus [2 solutions, n_lines, m_sphere]
    pm = torch.stack([torch.ones_like(delta), -torch.ones_like(delta)])

    sqrt_delta = torch.sqrt(delta)
    solutions = (-b - (pm * sqrt_delta)) / (2 * a)

    return solutions


def change_cartesian_to_spherical(x, y, z, r=None):
    """Cartesian coordinates to spherical polar coordinates.
    Cartesian -> spherical
    (x,y,z) -> (phi, theta, r)
    r = sqrt(x**2, y**2, z**2)
    tg(phi) = z/x
    sin(theta) = y/r
    """
    if r is None:
        r = torch.sqrt(x**2 + y**2 + z**2)
    phi = torch.arctan(z/x)
    theta = torch.arcsin(y/r)
    return torch.stack((phi.T, theta.T, r.T)).swapaxes(0, 2)
