from torch.nn import functional as F
import torch


def scale_points_with_weights(
    z_vals: torch.Tensor, rays_o: torch.Tensor, rays_d: torch.Tensor
):
    """Scales rays that starts at origin by values returned from sampler."""
    normalized_rays_d = F.normalize(rays_d)
    return rays_o[..., None, :] + normalized_rays_d[..., None, :] * z_vals[..., :, None]
