"""
Implements baseline sampling network torch module
"""
import torch
from torch import nn
import torch.nn.functional as F
from nerf_pytorch.run_nerf_helpers import get_embedder
from .utils import scale_points_with_weights


class BaselineSampler(nn.Module):
    """
    Baseline sampling network
    """
    def __init__(
            self,
            origin_channels = 3,
            direction_channels = 3,
            output_channels = 40,
            noise_size = None,
            far = 6,
            near = 2
    ):
        super(BaselineSampler, self).__init__()
        self.noise_size = noise_size
        self.w1 = 256
        self.w2 = 128
        self.origin_channels = origin_channels
        self.direction_channels = direction_channels
        self.output_channels = output_channels
        self.far = far
        self.near = near

        self.origin_embedder, self.origin_dims = get_embedder(multires=10, input_dims=origin_channels)
        self.direction_embedder, self.direction_dims = get_embedder(multires=10, input_dims=direction_channels)

        self.origin_layers = nn.ModuleList(
            [
                nn.Linear(self.origin_dims, self.w2),
                nn.Linear(self.w2,self.w2),
                nn.Linear(self.w2,self.w2),
                nn.Linear(self.w2,self.w2),
                nn.Linear(self.w2,self.w2),
                nn.Linear(self.w2, self.w2),
            ]
        )

        self.direction_layers = nn.ModuleList(
            [
                nn.Linear(self.direction_dims, self.w2),
                nn.Linear(self.w2,self.w2),
                nn.Linear(self.w2,self.w2),
                nn.Linear(self.w2,self.w2),
                nn.Linear(self.w2,self.w2),
                nn.Linear(self.w2, self.w2),
            ]
        )

        self.layers = nn.ModuleList(
            [
                nn.Linear(self.w2 * 2, self.w2),
                nn.Linear(self.w2,self.w2),
                nn.Linear(self.w2,self.w2),
                nn.Linear(self.w2, self.w2),
                nn.Linear(self.w2, self.w1)
            ]
        )

        self.last = nn.Linear(self.w1, self.output_channels)

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

        # [N_rays, N_samples]
        z_vals = self.near * (1 - outputs) + self.far * outputs

        # To add noise self.noise_size times we need to broadcast vector
        n_rays = z_vals.shape[0]
        if self.noise_size:
            z_vals = z_vals.unsqueeze(-1).repeat(1,1,self.noise_size)
            z_vals = z_vals + torch.normal(mean=0, std=0.001, size=z_vals.size())
            z_vals = z_vals.reshape([n_rays, self.output_channels * self.noise_size])

        z_vals, _ = z_vals.sort(dim=-1)
        print(f"z_vals output shape {z_vals.shape}")
        # [N_rays, N_samples, 3] and [N_rays, N_samples]
        # Scaled points in visualizer have to be associated with ray origin
        # From origin to points x such that d(origin, x) = 2 line is blue
        # From x to point y such that d(origin, y) = 6 line is red

        # Save batch from last epoch
        return scale_points_with_weights(z_vals, rays_o, rays_d), z_vals
    