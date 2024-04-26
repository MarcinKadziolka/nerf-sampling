"""Utility functions."""

import importlib
import random

import matplotlib.pyplot as plt
import torch
import wandb

from . import visualize


def load_obj_from_config(cfg: dict):
    """Create an object based on the specified module path and kwargs."""
    module_name, class_name = cfg["module"].rsplit(".", maxsplit=1)

    cls = getattr(
        importlib.import_module(module_name),
        class_name,
    )

    return cls(**cfg["kwargs"])


def freeze_model(model):
    """Set requires_grad of all parameters model to False."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    """Set requires_grad of all parameters model to True."""
    for param in model.parameters():
        param.requires_grad = True


def wandb_log_rays(rays_o, rays_d, pts, info, step, title):
    """Log ray and points plot to wandb."""
    rays_fig, _ = visualize.visualize_random_rays_pts(rays_o, rays_d, pts, title=title)
    wandb.log(
        {
            f"Ray plot {info} {step}": wandb.Image(rays_fig),
        }
    )
    plt.close(rays_fig)


def get_dense_indices(densities: torch.Tensor, min_density: float):
    """Obtains indices of points that have density > min_density.

    Args:
      densities: [H, W, N_samples].
      min_density: Minimum density required to get index.
    """
    return densities > min_density


def get_random_points(pts: torch.Tensor, k: int):
    """Extract k random points from provided.

    Args:
      pts: [N_samples, 3].
      k: How many random points to select.

    Returns:
      pts: [k, 3].
    """
    indices = random.sample(range(len(pts)), k=k)
    return pts[indices]
