"""Blender trainer module - trainer for blender data."""

from nerf_pytorch.trainers import Blender
import torch

from sphere_nerf_mod.lines import Lines
from sphere_nerf_mod.spheres import Spheres


class SphereBlenderTrainer(Blender.BlenderTrainer):
    """Trainer for blender data."""

    def __init__(
            self,
            spheres: Spheres = None,
            **kwargs
    ):
        """Initialize the blender trainer.

        In addition to original nerf_pytorch BlenderTrainer,
        the trainer contains the spheres used in the training process.
        """
        super().__init__(
            **kwargs
        )
        self.spheres = spheres

    def sample_points(
        self,
        z_vals_mid,
        weights,
        perturb,
        pytest,
        rays_d,
        rays_o
    ) -> (torch.Tensor, torch.Tensor):
        """Create rays as Lines object and sample points."""
        z_samples, original_nerf_points = self._sample_points(
            z_vals_mid=z_vals_mid,
            weights=weights,
            perturb=perturb,
            pytest=pytest,
            rays_o=rays_o,
            rays_d=rays_d,
            n_importance=self.N_importance - self.spheres.get_number()
        )

        rays_origins = rays_o
        rays_directions = rays_d
        rays = Lines(rays_origins, rays_directions)
        sphere_nerf_points = self.sample_points_on_spheres(
            rays
        ).swapaxes(0, 1)

        z_sphere = rays.transform_points_to_single_number_representation(
            sphere_nerf_points
        )

        return torch.hstack((z_samples, z_sphere)), torch.hstack(
            (original_nerf_points, sphere_nerf_points)
        )

    def sample_points_on_spheres(
            self,
            rays: Lines,
            point_coordinate_if_nan: float = 100
    ) -> torch.Tensor:
        """Sample points on given rays.

        Samples points on rays - one point per sphere.
        The points are sampled by first searching for intersection
        points between the rays and spheres.
        Then, in case when a ray and a sphere intersect at two points,
        the point that is closer to the ray's origin is sampled.
        In case, when a ray and a sphere do not intersect,
        a point (100, 100, 100) is sampled.
        Args:
            rays: camera rays represented as Lines class object
            point_coordinate_if_nan: replacing the nan points with
             replacement value by coordinates.
        Return:
            Tensor with dimensions [spheres, rays, 3].

        """
        intersection_points = rays.find_intersection_points_with_sphere(
            self.spheres
        )
        selected_points = rays.select_closest_point_to_origin(
            intersection_points
        )

        return torch.nan_to_num(
            selected_points, nan=point_coordinate_if_nan
        )
