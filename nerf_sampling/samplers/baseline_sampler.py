"""Implements baseline sampling network torch module."""

from nerf_sampling.nerf_pytorch.run_nerf_helpers import get_embedder
from .utils import scale_points_with_weights
from torch import nn
import torch.nn.functional as F
import torch


class BaselineSampler(nn.Module):
    """Baseline sampling network."""

    def __init__(
        self,
        hidden_sizes: list[int] = [128 for _ in range(6)],
        cat_hidden_sizes: list[int] = [128, 128, 128, 128, 256],
        origin_channels: int = 3,
        direction_channels: int = 3,
        near: int = 2,
        far: int = 6,
        n_samples: int = 32,
        multires: int = 10,
    ):
        """Initializes sampling network.

        Args:
            hidden_sizes: Hidden size of each consecutive linear layer for origin and direction.
            cat_hidden_sizes: Hidden size of each consecutive linear layer
                for concatenated output of origin and direction layers.
            origin_channels: Expected number of channels of ray origin vector.
            direction_channels: Expected number of channels of ray direction vector.
            near: Nearest distance for a ray.
            far: Farthest distance for a ray.
            n_samples: Number of samples to output along a ray.
        """
        super(BaselineSampler, self).__init__()
        self.n_samples = n_samples
        self.far = far
        self.near = near

        self.origin_embedder, self.origin_dims = get_embedder(
            multires=multires, input_dims=origin_channels
        )
        self.direction_embedder, self.direction_dims = get_embedder(
            multires=multires, input_dims=direction_channels
        )

        origin_layers: list[nn.Linear | nn.ReLU] = [
            nn.Linear(self.origin_dims, hidden_sizes[0]),
            nn.ReLU(),
        ]
        direction_layers: list[nn.Linear | nn.ReLU] = [
            nn.Linear(self.direction_dims, hidden_sizes[0]),
            nn.ReLU(),
        ]

        for i, size in enumerate(hidden_sizes[:-1]):
            for layers in [origin_layers, direction_layers]:
                layers.append(
                    nn.Linear(in_features=size, out_features=hidden_sizes[i + 1])
                )
                layers.append(nn.ReLU())

        cat_layers: list[nn.Linear | nn.ReLU | nn.Sigmoid] = [
            nn.Linear(
                hidden_sizes[-1] * 2 + self.origin_dims + self.direction_dims,
                cat_hidden_sizes[0],
            ),
            nn.ReLU(),
        ]

        for i, size in enumerate(cat_hidden_sizes[:-1]):
            cat_layers.append(
                nn.Linear(in_features=size, out_features=cat_hidden_sizes[i + 1])
            )
            cat_layers.append(nn.ReLU())

        self.origin_layers = nn.Sequential(*origin_layers)
        self.direction_layers = nn.Sequential(*direction_layers)
        self.cat_layers = nn.Sequential(*cat_layers)

        self.to_n_samples = nn.Linear(cat_hidden_sizes[-1], self.n_samples)
        self.sigmoid = nn.Sigmoid()

    def scale_to_near_far(self, outputs, rays_o, rays_d):
        """Directly scales points from NN output to the range [NEAR, FAR]."""
        # [N_rays, N_samples]
        z_vals = self.near * (1 - outputs) + self.far * outputs
        z_vals, _ = z_vals.sort(dim=-1)
        # [N_rays, N_samples, 3] and [N_rays, N_samples]
        # Scaled points in visualizer have to be associated with ray origin
        # From origin to points x such that d(origin, x) = 2 line is blue
        # From x to point y such that d(origin, y) = 6 line is red

        return scale_points_with_weights(z_vals, rays_o, rays_d), z_vals

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        """For given ray origins and directions returns points sampled along ray.

        self.n_samples points per ray.
        Points are within [near, far] distance from the origin.
        """
        embedded_origin = self.origin_embedder(rays_o)
        embedded_direction = self.direction_embedder(rays_d)

        origin_outputs = self.origin_layers(embedded_origin)
        direction_outputs = self.direction_layers(embedded_direction)

        outputs = torch.cat([origin_outputs, direction_outputs], -1)
        skip_connection = torch.cat([outputs, embedded_origin, embedded_direction], -1)

        concat_outputs = self.cat_layers(skip_connection)
        n_samples_output = self.to_n_samples(concat_outputs)
        final_outputs = self.sigmoid(n_samples_output)
        return self.scale_to_near_far(final_outputs, rays_o, rays_d)
