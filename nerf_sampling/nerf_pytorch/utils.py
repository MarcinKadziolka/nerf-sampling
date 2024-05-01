"""Utility functions."""

import importlib
import random
from typing import Optional

import matplotlib.pyplot as plt
import torch
import wandb

from nerf_sampling.nerf_pytorch.run_nerf_helpers import NeRF

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


def get_dense_indices(densities: torch.Tensor, min_density: torch.Tensor):
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


def save_state(
    global_step: int,
    network_fn: NeRF,
    network_fine: Optional[NeRF],
    optimizer,
    sampling_network,
    sampling_optimizer,
    path: str,
) -> None:
    """Saves states of provided data to path.

    Args:
      global_step: Current iteration of the training.
      network_fn: Usually coarse network, unless network_fine is None, then it's the main NeRF model.
      network_fine: Network evaluating N_c + N_f samples (section 5.3: Implementation details). Optional.
      optimizer: Optimizer for NeRF model.
      sampling_network: Model responsibile for sampling.
      sampling_optimizer: Optimizer of sampling network.
      path: Path to save directory including filename. Example: /your/dir/data.tar
    """
    data = {
        "global_step": global_step,
        "network_fn_state_dict": network_fn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "sampling_optimizer_state_dict": sampling_optimizer.state_dict(),
        "sampling_network": sampling_network.state_dict(),
    }
    if network_fine is not None:
        data["network_fine_state_dict"] = network_fine.state_dict()
    torch.save(data, path)
    print("Saved checkpoints at", path)


def load_nerf(network_fn: NeRF, network_fine: Optional[NeRF], optimizer, ckpt):
    """Loades states of nerf models and optim from checkpoint.

    Args:
      network_fn: Usually coarse network, unless network_fine is None, then it's the main NeRF model.
      network_fine: Network evaluating N_c + N_f samples (section 5.3: Implementation details). Optional.
      optimizer: Optimizer for NeRF model.
      ckpt: Path to save directory including filename. Example: /your/dir/data.tar
    """
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # Load model
    network_fn.load_state_dict(ckpt["network_fn_state_dict"])
    if network_fine is not None:
        network_fine.load_state_dict(ckpt["network_fine_state_dict"])


def load_sampling_network(sampling_network, sampling_optimizer, ckpt):
    sampling_optimizer.load_state_dict(ckpt["sampling_optimizer_state_dict"])
    sampling_network.load_state_dict(ckpt["sampling_network"])
