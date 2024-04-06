"""
Implements baseline sampling network torch module
"""

from nerf_sampling.nerf_pytorch.run_nerf_helpers import get_embedder
from .utils import scale_points_with_weights
from torch import nn
import torch.nn.functional as F
import torch


class BaselineSampler(nn.Module):
    """
    Baseline sampling network
    """

    def __init__(
        self,
        origin_channels=3,
        direction_channels=3,
        output_channels=40,
        noise_size=None,
        use_regions=False,
        use_summing=False,
        far=6,
        near=2,
    ):
        super(BaselineSampler, self).__init__()
        self.noise_size = noise_size
        self.use_regions = use_regions
        self.w1 = 256
        self.w2 = 128
        self.origin_channels = origin_channels
        self.direction_channels = direction_channels
        self.output_channels = output_channels
        self.far = far
        self.near = near
        self.use_summing = use_summing
        self.group_size = output_channels

        self.origin_embedder, self.origin_dims = get_embedder(
            multires=10, input_dims=origin_channels
        )
        self.direction_embedder, self.direction_dims = get_embedder(
            multires=10, input_dims=direction_channels
        )

        self.origin_layers = nn.ModuleList(
            [
                nn.Linear(self.origin_dims, self.w2),
                nn.Linear(self.w2, self.w2),
                nn.Linear(self.w2, self.w2),
                nn.Linear(self.w2, self.w2),
                nn.Linear(self.w2, self.w2),
                nn.Linear(self.w2, self.w2),
            ]
        )

        self.direction_layers = nn.ModuleList(
            [
                nn.Linear(self.direction_dims, self.w2),
                nn.Linear(self.w2, self.w2),
                nn.Linear(self.w2, self.w2),
                nn.Linear(self.w2, self.w2),
                nn.Linear(self.w2, self.w2),
                nn.Linear(self.w2, self.w2),
            ]
        )

        self.layers = nn.ModuleList(
            [
                nn.Linear(self.w2 * 2, self.w2),
                nn.Linear(self.w2, self.w2),
                nn.Linear(self.w2, self.w2),
                nn.Linear(self.w2, self.w2),
                nn.Linear(self.w2, self.w1),
            ]
        )

        self.last = nn.Linear(self.w1, self.output_channels)

    def set_group_size(self, k):
        """Changes size of group"""
        print(f"Increased group size to {k}")
        self.group_size = k

    def scale_without_regions(self, outputs, rays_o, rays_d):
        """Directly scales points from NN output to the range [NEAR, FAR]"""
        # [N_rays, N_samples]
        z_vals = self.near * (1 - outputs) + self.far * outputs

        # To add noise self.noise_size times we need to broadcast vector
        n_rays = z_vals.shape[0]
        if self.noise_size:
            z_vals = z_vals.unsqueeze(-1).repeat(1, 1, self.noise_size)
            z_vals = z_vals + torch.normal(mean=0, std=0.001, size=z_vals.size())
            z_vals = z_vals.reshape([n_rays, self.output_channels * self.noise_size])
            z_vals = torch.clamp(z_vals, min=self.near, max=self.far)

        z_vals, _ = z_vals.sort(dim=-1)
        # [N_rays, N_samples, 3] and [N_rays, N_samples]
        # Scaled points in visualizer have to be associated with ray origin
        # From origin to points x such that d(origin, x) = 2 line is blue
        # From x to point y such that d(origin, y) = 6 line is red

        if self.use_summing and self.group_size < n_rays and self.group_size > 1:
            print(f"z_vals shape before summing {z_vals.shape}")
            z_vals = torch.mean(z_vals.reshape(n_rays, -1, self.group_size), -1)
            print(f"z_vals shape after summing {z_vals.shape}")

        return scale_points_with_weights(z_vals, rays_o, rays_d), z_vals

    def scale_with_regions(self, outputs, rays_o, rays_d):
        """
        Splits rays into regions to prevent squeezing all points
        in one place when alphas are used in loss.
        This method does not support adding noise for now.
        """

        # We need to divide [NEAR, FAR] into N_samples equal parts
        intervals = torch.linspace(self.near, self.far, self.output_channels + 1).view(
            -1, 1
        )
        intervals = torch.cat((intervals[:-1], intervals[1:]), dim=1)

        # Now we need to scale i-th output to accordingly to the i-th interval range
        intervals_exp = intervals.unsqueeze(0).expand(outputs.shape[0], -1, -1)
        min_vals, max_vals = intervals_exp[:, :, 0], intervals_exp[:, :, 1]
        z_vals = min_vals + outputs * (max_vals - min_vals)

        return scale_points_with_weights(z_vals, rays_o, rays_d), z_vals

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        """
        For given ray origins and directions returns points sampled along ray.
        `self.output_channels` points per ray.
        Points are within [near, far] distance from the origin.
        """
        embedded_origin = self.origin_embedder(rays_o)
        embedded_direction = self.direction_embedder(rays_d)

        origin_outputs = embedded_origin
        direction_outputs = embedded_direction

        for layer in self.origin_layers:
            origin_outputs = layer(origin_outputs)
            origin_outputs = F.relu(origin_outputs)

        for layer in self.direction_layers:
            direction_outputs = layer(direction_outputs)
            direction_outputs = F.relu(direction_outputs)

        outputs = torch.cat([origin_outputs, direction_outputs], -1)

        for layer in self.layers:
            outputs = layer(outputs)
            outputs = F.relu(outputs)

        outputs = self.last(outputs)
        outputs = F.sigmoid(outputs)

        if not self.use_regions:
            return self.scale_without_regions(outputs, rays_o, rays_d)

        return self.scale_with_regions(outputs, rays_o, rays_d)
