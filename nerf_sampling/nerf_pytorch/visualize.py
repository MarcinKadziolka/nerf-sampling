"""Provides functions for visualization of outputs of NeRF model and sampler."""

import random
import torch
from typing import Optional
import matplotlib.pyplot as plt


def plot_density_histogram(densities: torch.Tensor, title: str = "Histogram"):
    """Plot density histogram.

    Args:
        densities: [N_rays, N_samples]. Output of NeRF model or its transformation.
            Either raw densities, alphas or weights
        title: Title of the plot.
    """
    # densities [N_rays, N_samples]
    densities = torch.flatten(densities).detach().cpu()
    plt.hist(densities)
    plt.title(title)
    plt.xlabel("Density")
    plt.ylabel("N of samples")
    plt.show()


def visualize_random_rays_pts(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    pts: Optional[torch.Tensor] = None,
    n_rays: int = 10,
    near: float = 2.0,
    far: float = 6.0,
    title: str = "3D plot",
) -> None:
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
    n_rays = min(n_rays, len(rays_o))
    indices = random.sample(range(len(rays_o)), n_rays)

    selected_rays_o = rays_o[indices].cpu()
    selected_rays_d = rays_d[indices].cpu()
    selected_pts = pts[indices].detach().cpu() if pts is not None else None

    _, ax = initialize_3d_plot()
    plot_rays(ax, selected_rays_o, selected_rays_d, near, far)
    if selected_pts is not None:
        plot_points(ax, selected_pts)
    plt.title(title)
    plt.show()


def initialize_3d_plot() -> tuple:
    """Initialize 3d plot and set axis labels."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.xaxis._axinfo["label"]["space_factor"] = 2.0
    ax.yaxis._axinfo["label"]["space_factor"] = 2.0
    ax.zaxis._axinfo["label"]["space_factor"] = 2.0
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return fig, ax


def normalize_directions(rays_d: torch.Tensor) -> torch.Tensor:
    """Normalizes direction vectors (rays_d) in 3D space.

    Args:
        rays_d: [N_rays, 3]. Direction vectors of rays

    Returns:
        Normalized direction vectors.
    """
    return rays_d / torch.linalg.norm(rays_d, dim=1, keepdims=True)


def test_normalize_directions():
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
    ax, rays_o: torch.Tensor, rays_d: torch.Tensor, near: float = 2, far: float = 6
) -> None:
    """Plots rays in 3D space.

    Args:
        ax: The 3D axes to plot on.
        rays_o: [N_rays, 3]. Origin points of rays
        rays_d: [N_rays, 3]. Direction vectors of rays
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


def plot_points(ax, ray_pts: torch.Tensor) -> None:
    """Plot points per rays.

    Args:
      ax: matplotlib.axes
      ray_pts: [N_rays, N_samples, 3]. 3D points to plot on given axes
    """
    for pts in ray_pts:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])


def main():
    """Run example visualization of rays and points."""
    rays_o = torch.zeros((6, 3))
    rays_d = torch.Tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    )
    pts = torch.Tensor(
        [
            [[1, 0, 0]],
            [[0, 2, 0]],
            [[0, 0, 3]],
            [[-4, 0, 0]],
            [[0, -5, 0]],
            [[0, 0, -6]],
        ]
    )
    visualize_random_rays_pts(rays_o, rays_d, pts)


if __name__ == "__main__":
    main()
