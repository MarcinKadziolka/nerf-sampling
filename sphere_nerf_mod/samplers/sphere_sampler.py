"""
Implements baseline sampling network torch module
"""
import torch
from torch import nn
import torch.nn.functional as F
from nerf_pytorch.run_nerf_helpers import get_embedder
from .utils import scale_points_with_weights


class SphereSampler(nn.Module):
    """
    Baseline sampling network
    """
    def __init__(
            self,
            origin_channels = 3,
            direction_channels = 3,
            output_channels = 40,
            far = 6,
            near = 2
    ):
        super(SphereSampler, self).__init__()
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
                nn.Linear(self.origin_dims, self.w1),
                nn.Linear(self.w1,self.w2),
                nn.Linear(self.w2,self.w2),
                nn.Linear(self.w2,self.w2),
                nn.Linear(self.w2,self.w2),
                nn.Linear(self.w2, self.w2),
            ]
        )

        self.direction_layers = nn.ModuleList(
            [
                nn.Linear(self.origin_dims, self.w1),
                nn.Linear(self.w1,self.w2),
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
            ]
        )

        self.last = nn.Linear(self.w2, self.output_channels)

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor, intersections: torch.Tensor):
        """
        For given ray origins and directions returns points sampled along ray.
        `self.output_channels` points per ray.
        Points are within [near, far] distance from the origin.
        """
        embedded_origin = self.origin_embedder(rays_o)
        embedded_direction = self.direction_embedder(intersections)

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

        z_vals = self.near * (1 - outputs) + self.far * outputs
        z_vals, _ = z_vals.sort(dim=-1)


        return scale_points_with_weights(z_vals, rays_o, rays_d), z_vals
    