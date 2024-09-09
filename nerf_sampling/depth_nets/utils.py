from torch.nn import functional as F
import torch


def scale_points_with_weights(
    z_vals: torch.Tensor, rays_o: torch.Tensor, rays_d: torch.Tensor
):
    """Scales rays that starts at origin by values returned from depth_net."""
    # normalized_rays_d = F.normalize(rays_d)
    return rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]


def scale_to_near_far(outputs, rays_o, rays_d, near, far):
    """Directly scales z_vals from range [0, 1] to the range [NEAR, FAR]."""
    # [N_rays, N_samples]
    z_vals = near * (1 - outputs) + far * outputs
    z_vals, _ = z_vals.sort(dim=-1)

    return scale_points_with_weights(z_vals, rays_o, rays_d), z_vals
