"""Utility functions."""

import importlib
import random
from typing import Optional, Union, Literal
import torch

from nerf_sampling.nerf_pytorch.run_nerf_helpers import NeRF


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
      ckpt: Loaded data dict. ckpt = torch.load(path)
    """
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print("Successfully loaded optimizer")
    # Load model
    network_fn.load_state_dict(ckpt["network_fn_state_dict"])
    print("Successfully loaded network_fn")
    if network_fine is not None:
        network_fine.load_state_dict(ckpt["network_fine_state_dict"])
    print("Successfully loaded network_fine")


def load_sampling_network(sampling_network, sampling_optimizer, ckpt):
    """Loades states of sampling model and optim from checkpoint.

    Args:
      sampling_network: Model responsibile for sampling.
      sampling_optimizer: Optimizer for sampling model.
      ckpt: Loaded data dict. ckpt = torch.load(path)
    """
    sampling_optimizer.load_state_dict(ckpt["sampling_optimizer_state_dict"])
    print("Successfully loaded sampling_optimizer")
    sampling_network.load_state_dict(ckpt["sampling_network"])
    print("Successfully loaded sampling_network")


def override_config(config, update):
    """Overrides config with update dict only when the keys match.

    Args:
        config: Dictionary to override.
        update: Dictionary with changes to apply.

    Raises:
        KeyError: If key that is supposed to be overwritten does not exist.
    """
    config_keys = config.keys()
    for key, value in update.items():
        if key in config_keys:
            config[key] = value
        else:
            raise KeyError(f"Key {key} does not exist in config")


def set_global_device(device: Union[Literal["cuda"], Literal["cpu"]]):
    """Set all tensors to be created on device."""
    if device == "cuda":
        if torch.cuda.is_available():
            torch.set_default_device(device="cuda")
    elif device == "cpu":
        torch.set_default_device(device="cpu")
