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
        n_samples: int = 0,
        distance: float = 0.05,
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
        self.far = far
        self.near = near
        self.distance = distance
        self.n_samples = n_samples

        self.origin_embedder, self.origin_dims = get_embedder(
            multires=multires, input_dims=origin_channels
        )
        self.direction_embedder, self.direction_dims = get_embedder(
            multires=multires, input_dims=direction_channels
        )

        origin_layers: list[nn.Linear | nn.ReLU] = [
            nn.Linear(self.origin_dims + self.origin_dims, hidden_sizes[0]),
        ]
        direction_layers: list[nn.Linear | nn.ReLU] = [
            nn.Linear(self.direction_dims + self.direction_dims, hidden_sizes[0]),
        ]

        # no ReLU here, it's added in the forward() method
        for i, size in enumerate(hidden_sizes[:-1]):
            for layers in [origin_layers, direction_layers]:
                layers.append(
                    nn.Linear(
                        # account for skip connection (concatenating output of the layer with embedded origins/directions)
                        in_features=size + self.origin_dims,
                        out_features=hidden_sizes[i + 1],
                    )
                )

        cat_layers: list[nn.Linear | nn.ReLU] = [
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

        self.to_mean = nn.Linear(cat_hidden_sizes[-1], 1)
        self.sigmoid = nn.Sigmoid()

        print(self)
        print(f"{self.n_samples=}")
        print(f"{self.distance=}")

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

    def get_z_vals(self, mean):
        grid = torch.linspace(-self.distance, self.distance, steps=self.n_samples)

        # Expand the grid to match the shape of outputs
        expanded_grid = grid.view(1, -1).expand(mean.size(0), -1)

        # Add the grid to the outputs to center the samples around outputs
        noise_z_vals = mean + expanded_grid

        # Clip the values between 0 and 1
        noise_z_vals = torch.clip(noise_z_vals, 0, 1)
        return noise_z_vals

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        """For given ray origins and directions returns points sampled along ray.

        self.n_samples points per ray.
        Points are within [near, far] distance from the origin.
        """
        embedded_origin = self.origin_embedder(rays_o)
        embedded_direction = self.direction_embedder(rays_d)

        origin_outputs = embedded_origin
        for layer in self.origin_layers:
            # skip connection in every layer
            origin_outputs = layer(torch.cat([origin_outputs, embedded_origin], -1))
            nn.LeakyReLU(origin_outputs)

        direction_outputs = embedded_direction
        for layer in self.direction_layers:
            # skip connection in every layer
            direction_outputs = layer(
                torch.cat([direction_outputs, embedded_direction], -1)
            )
            nn.LeakyReLU(direction_outputs)

        outputs = torch.cat([origin_outputs, direction_outputs], -1)
        skip_connection = torch.cat([outputs, embedded_origin, embedded_direction], -1)

        concat_outputs = self.cat_layers(skip_connection)
        predicted_mean = self.to_mean(concat_outputs)
        sigmoid_predicted_mean = self.sigmoid(predicted_mean)

        z_vals = self.get_z_vals(sigmoid_predicted_mean)
        mean = (
            self.near * (1 - sigmoid_predicted_mean) + self.far * sigmoid_predicted_mean
        )
        return self.scale_to_near_far(z_vals, rays_o, rays_d), mean
