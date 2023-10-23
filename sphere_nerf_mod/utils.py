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


def reflected(
    spheres,
    img_height,
    img_width,
    base_camera,
    focal=555 * 3
):
    transform_matrix_focal = torch.tensor(
        [[focal, 0, 0.5 * img_width],
         [0, focal, 0.5 * img_height],
         [0, 0, 1]]
    )

    rays_o, rays_d = get_rays_focal(
        img_height,
        img_width,
        transform_matrix_focal,
        base_camera,
        origin=spheres.center[0]
    )

    return rays_o, rays_d


def get_rays_focal(
    img_height,
    img_width,
    transform_matrix_focal,
    base_camera,
    origin
):
    i, j = torch.meshgrid(
torch.linspace(0, img_width - 1, img_width),
        torch.linspace(0, img_height - 1, img_height)
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()

    _w = transform_matrix_focal[0][2]
    focal = transform_matrix_focal[0][0]

    dirs = torch.stack(
        [(i - _w) / focal, -(j - _w) / focal, -torch.ones_like(i)], -1)

    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * base_camera[:3, :3], -1
    )
    rays_o = origin.expand(rays_d.shape)
    return rays_o, -rays_d
