"""Provides functions for visualization of outputs of NeRF model and sampler."""

import torch
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes


def plot_histogram(
    densities: torch.Tensor, title: str = "Histogram"
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot density histogram.

    Args:
        densities: [N_rays, N_samples]. Output of NeRF model or its transformation.
            Either raw densities, alphas or weights
        title: Title of the plot.
    """
    # densities [N_rays, N_samples]
    densities = torch.flatten(densities)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(densities)
    ax.set_title(title)
    ax.set_xlabel("Density")
    ax.set_ylabel("N of samples")
    return fig, ax


def visualize_rays_pts(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    pts: Optional[torch.Tensor] = None,
    n_rays: int = 3,
    near: float = 2.0,
    far: float = 6.0,
    title: str = "Points sampled on rays",
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot randomly selected rays and sampled points (if they are provided).

    Args:
        rays_o: [N_rays, 3]. Origin points of rays.
        rays_d: [N_rays, 3]. Direction vectors of rays.
        pts: [N_rays, N_samples, 3]. If provided display sampled points along given ray.
        n_rays: Num of rays to randomly select.
        near: Nearest distance for a ray.
        far: Farthest distance for a ray.
        title: Title for the plot.
    """
    fig, ax = _initialize_3d_plot()
    _plot_rays(ax, rays_o, rays_d, near, far)
    if pts is not None:
        _plot_points(ax, pts, s=s, c=c)
    plt.title(title)
    return fig, ax


def _initialize_3d_plot() -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Initialize 3d plot and set axis labels."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.xaxis._axinfo["label"]["space_factor"] = 2.0
    ax.yaxis._axinfo["label"]["space_factor"] = 2.0
    ax.zaxis._axinfo["label"]["space_factor"] = 2.0
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=45)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    return fig, ax


def normalize_directions(rays_d: torch.Tensor) -> torch.Tensor:
    """Normalizes direction vectors (rays_d) in 3D space.

    Args:
        rays_d: [N_rays, 3]. Direction vectors of rays

    Returns:
        Normalized direction vectors.
    """
    return rays_d / torch.linalg.norm(rays_d, dim=1, keepdims=True)


def test_normalize_directions():  # noqa: D103
    rays_d = torch.Tensor([[1.5, 0, 3.14], [-1, 0.25, 0.33]])
    # vector = [x, y, z]
    # magnitude = sqrt(x^2 + y^2 + z^2)
    # expected_vector =  [x/magnitude, y/magnitude, z/magnitude]
    expected = torch.Tensor(
        [
            [0.43104810784, 0, 0.90232737241],
            [-0.92394970017, 0.23098742504, 0.30490340105],
        ]
    )
    direction_norm = normalize_directions(rays_d)
    assert rays_d.shape == direction_norm.shape
    assert torch.allclose(expected, direction_norm)


def plot_rays(
    rays_o: torch.Tensor, rays_d: torch.Tensor, near: float = 2, far: float = 6
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plots rays in 3D space.

    Args:
        rays_o: [N_rays, 3]. Origin points of rays.
        rays_d: [N_rays, 3]. Direction vectors of rays.
        near: Nearest distance for a ray.
        far: Farthest distance for a ray.
    """
    fig, ax = _initialize_3d_plot()
    _plot_rays(ax, rays_o, rays_d, near, far)
    return fig, ax


def _plot_rays(
    ax, rays_o: torch.Tensor, rays_d: torch.Tensor, near: float = 2, far: float = 6
) -> None:
    """Plots rays in 3D space on given axes.

    Args:
        ax: The 3D axes to plot on.
        rays_o: [N_rays, 3]. Origin points of rays.
        rays_d: [N_rays, 3]. Direction vectors of rays.
        near: Nearest distance for a ray.
        far: Farthest distance for a ray.
    """
    direction_norm = normalize_directions(rays_d)
    near_segment = rays_o + direction_norm * near
    far_segment = rays_o + direction_norm * far

    for origin, near_pt, far_pt in zip(rays_o, near_segment, far_segment):
        ax.plot(
            [origin[0], near_pt[0]],
            [origin[1], near_pt[1]],
            [origin[2], near_pt[2]],
            color="red",
        )
        ax.plot(
            [near_pt[0], far_pt[0]],
            [near_pt[1], far_pt[1]],
            [near_pt[2], far_pt[2]],
            color="gray",
        )


def plot_points(ray_pts: torch.Tensor, s: int = 20):
    """Plot points per rays.

    Args:
      ray_pts: [N_rays, N_samples, 3]. 3D points to plot
      s: Marker size.
    """
    fig, ax = _initialize_3d_plot()
    _plot_points(ax, ray_pts, s=s)
    return fig, ax


def _plot_points(ax, ray_pts: torch.Tensor, s: int = 20) -> matplotlib.axes.Axes:
    """Plot points per rays on axes.

    Args:
      ax: matplotlib.axes
      ray_pts: [N_rays, N_samples, 3]. 3D points to plot on given axes
      s: Marker size.
    """
    for pts in ray_pts:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s)
    return ax
    )
