"""Utility functions."""

import importlib
import random
from typing import Optional, Union, Literal
import torch
from torch.nn import functional as F

from nerf_sampling.nerf_pytorch.run_nerf_helpers import NeRF
from enum import Enum, auto


class RenderingMode(Enum):
    COMPARE = auto()
    MAX = auto()
    FULL = auto()
    DEPTH = auto()
    NDEPTH = auto()


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


def get_min_indices(densities: torch.Tensor, min_density: torch.Tensor):
    """Obtains indices of points that have density >= min_density.

    Args:
      densities: [H, W, N_samples].
      min_density: Minimum density required to get index.
    """
    return densities >= min_density


def get_random_indices(N_samples: torch.Tensor, k: int):
    """Selects k random indices from a range of 0 to N_samples - 1.

    Args:
      N_samples: Total number of population.
      k: How many random points to select.

    Returns:
      List[int]: A list containing k randomly selected indices.
    """
    return random.sample(range(N_samples), k=k)


def save_state(
    global_step: int,
    network_fn: NeRF,
    network_fine: Optional[NeRF],
    optimizer,
    depth_network,
    sampling_optimizer,
    path: str,
) -> None:
    """Saves states of provided data to path.

    Args:
      global_step: Current iteration of the training.
      network_fn: Usually coarse network, unless network_fine is None, then it's the main NeRF model.
      network_fine: Network evaluating N_c + N_f samples (section 5.3: Implementation details). Optional.
      optimizer: Optimizer for NeRF model.
      depth_network: Model responsibile for sampling.
      sampling_optimizer: Optimizer of sampling network.
      path: Path to save directory including filename. Example: /your/dir/data.tar
    """
    data = {
        "global_step": global_step,
        "network_fn_state_dict": network_fn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "sampling_optimizer_state_dict": sampling_optimizer.state_dict(),
        "depth_network": depth_network.state_dict(),
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


def load_depth_network(depth_network, sampling_optimizer, ckpt):
    """Loades states of sampling model and optim from checkpoint.

    Args:
      depth_network: Model responsibile for sampling.
      sampling_optimizer: Optimizer for sampling model.
      ckpt: Loaded data dict. ckpt = torch.load(path)
    """
    sampling_optimizer.load_state_dict(ckpt["sampling_optimizer_state_dict"])
    print("Successfully loaded sampling_optimizer")
    depth_network.load_state_dict(ckpt["depth_network"])
    print("Successfully loaded depth_network")


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


def check_grad(model):
    for param in model.parameters():
        if any(torch.flatten(param.grad)):
            return True
    return False


def solve_quadratic_equation(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """Solve quadratic equation ax^2 + bx + c = 0.

    Solves the quadratic equation with tensor coefficients.
    Returns nan if solution does not exist.
    The arguments can be of any shape, as long as their shapes are the same.

    For the argument shape `(x1, ..., xn)` the returned tensor
    has the shape (2, x1, ..., xn).
    For the equation specified by arguments' value at [x1, ..., xn],
    its' solutions are at returned tensor's [0, x1, ..., xn], [1, x1, ..., xn].
    """
    delta = b**2 - 4 * a * c  # [n_lines]

    # To compute minus/plus [2 solutions, n_lines]
    pm = torch.stack([torch.ones_like(delta), -torch.ones_like(delta)])

    sqrt_delta = torch.sqrt(delta)
    solutions = (-b - (pm * sqrt_delta)) / (2 * a)

    return solutions


def find_intersection_points_with_sphere(
    origin,
    direction,
    sphere_radius,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find intersection points with spheres.

    Finds intersection points of the lines with the given spheres.
    Returns nan values if a given line and sphere do not intersect.
    Args:
        calc_intersection_points_cartesian: bool if return pts in cartesian.
    Return:
        torch.Tensor containing the intersection points and with shape
        [m_spheres, n_lines, 2 points, 3D].
        t: torch.Tensor, z_vals
    """

    intersection_points = None
    sphere_center = torch.tensor([0, 0, 0])
    # [n_lines, 3D]
    origin_to_sphere_center_vector = origin - sphere_center

    # [n_lines]
    b = 2 * (direction * origin_to_sphere_center_vector).sum(dim=1)

    # [n_lines]
    c = torch.norm(origin_to_sphere_center_vector, dim=1) ** 2 - sphere_radius.T**2

    a = (direction * direction).sum(dim=1)

    equation_solutions = solve_quadratic_equation(a, b, c)  # [2_points, n_lines]

    t = equation_solutions.T  # [n_lines, 2]

    intersection_points = origin.unsqueeze(1) + t.unsqueeze(2) * direction.unsqueeze(1)
    return t, intersection_points  # [n_lines, 2 points, 3D]


def sample_points_around_mean(
    rays_o, rays_d, mean, n_samples=32, mode="gaussian", std=0.1
):
    # rays_d = F.normalize(rays_d)
    if mode == "depth_only":
        z_vals = mean
    elif mode == "gaussian":
        z_vals, _ = torch.cat(
            [mean + std * torch.randn(mean.shape[0], n_samples - 1), mean], dim=-1
        ).sort(dim=-1)
    elif mode == "uniform":
        grid = torch.linspace(-std, std, steps=n_samples - 1)

        # Expand the grid to match the shape of outputs
        expanded_grid = grid.view(1, -1).expand(mean.size(0), -1)

        # Add the grid to the outputs to center the samples around outputs
        z_vals, _ = torch.cat([mean + expanded_grid, mean], dim=-1).sort(dim=-1)

        # Clip the values between 0 and 1
        z_vals = torch.clip(z_vals, 2, 6)
    return (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None],
        z_vals,
    )
