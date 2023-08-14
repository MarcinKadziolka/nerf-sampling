"""Blender trainer module - trainer for blender data."""

from nerf_pytorch.trainers import Blender
import torch

from sphere_nerf_mod.lines import Lines
from sphere_nerf_mod.spheres import Spheres


POINT_COORDINATE_IF_NAN = 100


class BlenderTrainer(Blender.BlenderTrainer):
    """Trainer for blender data."""

    def __init__(self, spheres: Spheres):
        """Initialize the blender trainer.

        In addition to original nerf_pytorch BlenderTrainer,
        the trainer contains the spheres used in the training process.
        """
        # parameters are missing
        # super().__init__()
        self.spheres = spheres

    def sample_points(self, rays: Lines) -> torch.Tensor:
        """Sample ray points.

        Samples points on rays - one point per sphere.
        The points are sampled by first searching for intersection
        points between the rays and spheres.
        Then, in case when a ray and a sphere intersect at two points,
        the point that is closer to the ray's origin is sampled.
        In case, when a ray and a sphere do not intersect,
        a point (100, 100, 100) is sampled.
        Args:
            rays: camera rays represented as Lines class object
        Return:
            Tensor with dimensions [spheres, rays, 3].

        """
        intersection_points = rays.find_intersection_points_with_sphere(
            self.spheres
        )
        selected_points = rays.select_closest_point_to_origin(
            intersection_points
        )
        # replacing the nan points with replacement value by coordinates,
        # I don't know how to do where on the points, not individual values.
        return torch.where(
            ~torch.isnan(selected_points),
            selected_points,
            POINT_COORDINATE_IF_NAN
        )
