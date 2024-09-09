"""Implements baseline sampling network torch module."""

from nerf_sampling.nerf_pytorch.run_nerf_helpers import get_embedder
from nerf_sampling.nerf_pytorch.utils import find_intersection_points_with_sphere
from torch import nn
import torch.nn.functional as F
import torch


class DepthNet(nn.Module):
    """Baseline sampling network."""

    def __init__(
        self,
        hidden_sizes: list[int] = [128 for _ in range(6)],
        cat_hidden_sizes: list[int] = [128, 128, 128, 128, 256],
        origin_channels: int = 3,
        direction_channels: int = 3,
        multires: int = 10,
        sphere_radius: float = 2.0,
        near: int = 2,
        far: int = 6,
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
        super(DepthNet, self).__init__()
        self.sphere_radius = torch.tensor([sphere_radius])
        self.near = near
        self.far = far

        self.origin_embedder, self.origin_dims = get_embedder(
            multires=multires, input_dims=origin_channels
        )
        self.direction_embedder, self.direction_dims = get_embedder(
            multires=multires, input_dims=direction_channels
        )

        self.intersection_points_embedder, self.intersection_points_dim = get_embedder(
            multires=multires, input_dims=6
        )

        origin_layers: list[nn.Linear | nn.ReLU] = [
            nn.Linear(self.origin_dims + self.origin_dims, hidden_sizes[0]),
        ]
        direction_layers: list[nn.Linear | nn.ReLU] = [
            nn.Linear(self.direction_dims + self.direction_dims, hidden_sizes[0]),
        ]

        intersection_layers: list[nn.Linear | nn.ReLU] = [
            nn.Linear(
                self.intersection_points_dim + self.intersection_points_dim,
                hidden_sizes[0],
            ),
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

        for i, size in enumerate(hidden_sizes[:-1]):
            intersection_layers.append(
                nn.Linear(
                    # account for skip connection (concatenating output of the layer with embedded intersection_points)
                    in_features=size + self.intersection_points_dim,
                    out_features=hidden_sizes[i + 1],
                )
            )

        cat_layers: list[nn.Linear | nn.LeakyReLU] = [
            nn.Linear(
                hidden_sizes[-1] * 3
                + self.origin_dims
                + self.direction_dims
                + self.intersection_points_dim,
                cat_hidden_sizes[0],
            ),
            nn.LeakyReLU(),
        ]

        for i, size in enumerate(cat_hidden_sizes[:-1]):
            cat_layers.append(
                nn.Linear(in_features=size, out_features=cat_hidden_sizes[i + 1])
            )
            cat_layers.append(nn.LeakyReLU())

        self.origin_layers = nn.Sequential(*origin_layers)
        self.direction_layers = nn.Sequential(*direction_layers)
        self.intersection_layers = nn.Sequential(*intersection_layers)
        self.cat_layers = nn.Sequential(*cat_layers)
        self.to_depth = nn.Sequential(nn.Linear(cat_hidden_sizes[-1], 1), nn.Sigmoid())

        print(self)

    def calculate_intersection_points(self, rays_o, rays_d):
        t, intersection_points = find_intersection_points_with_sphere(
            rays_o, rays_d, self.sphere_radius
        )
        return intersection_points

    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
    ):
        """For given ray origins and directions returns points sampled along ray.

        self.n_samples points per ray.
        Points are within [near, far] distance from the origin.
        """
        embedded_origin = self.origin_embedder(rays_o)
        embedded_direction = self.direction_embedder(rays_d)
        intersection_points = self.calculate_intersection_points(
            rays_o=rays_o, rays_d=rays_d
        )
        embedded_intersection = self.intersection_points_embedder(
            torch.flatten(intersection_points, start_dim=1)
        )

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

        intersection_outputs = embedded_intersection
        for layer in self.intersection_layers:
            # skip connection in every layer
            intersection_outputs = layer(
                torch.cat([intersection_outputs, embedded_intersection], -1)
            )
            nn.LeakyReLU(intersection_outputs)

        outputs = torch.cat(
            [origin_outputs, direction_outputs, intersection_outputs], -1
        )
        skip_connection = torch.cat(
            [outputs, embedded_origin, embedded_direction, embedded_intersection], -1
        )

        concat_outputs = self.cat_layers(skip_connection)
        depth = self.to_depth(concat_outputs)

        scaled_depth = self.near * (1 - depth) + self.far * depth
        return scaled_depth
