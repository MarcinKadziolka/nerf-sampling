"""Blender trainer module - trainer for blender data."""

from nerf_pytorch.trainers import Blender
import torch
import torch.nn.functional as F
import numpy as np

from sphere_nerf_mod.lines import Lines
from sphere_nerf_mod.spheres import Spheres

from sphere_nerf_mod.models import (
    SphereConcatNeRF, SphereMoreViewdirsNeRF, SphereMoreViewsNeRF,
    SphereWithoutViewsNeRF, SphereTwoRGBNeRF)


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
        z_vals,
        pytest,
        rays_d,
        rays_o,
        network_fn,
        network_fine,
        network_query_fn,
        viewdirs,
        raw_noise_std,
        white_bkgd,
        **kwargs
    ):
        """Sample points on given rays.

        Define method how to sample points on given rays.
        Main idea behind SphereNerf is sampling points on spheres.
        The points are sampled by choosing the nearest point from the
        intersection points between the rays and surrounding the
        generated object spheres.
        """
        rays_origins = rays_o
        rays_directions = rays_d
        rays = Lines(rays_origins, rays_directions)
        sphere_nerf_points = self.sample_points_on_spheres(
            rays
        ).swapaxes(0, 1)

        z_sphere = rays.transform_points_to_single_number_representation(
            sphere_nerf_points
        )

        z_vals, _ = torch.sort(torch.cat([z_vals, z_sphere], -1), -1)
        _rays_d = rays_d[..., None, :]
        pts = rays_o[..., None, :] + _rays_d * z_vals[..., :, None]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, _, _ = self.raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        return None, None, None, rgb_map, disp_map, acc_map, raw

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

        In case, when a ray and a sphere do not intersect, a point, which
        all three coordinates are ``point_coordinate_if_nan`` is sampled.
        Args:
            rays: Lines - the camera rays.
            point_coordinate_if_nan: float - point coordinate value used to
             create points in place of nan points.
        Return:
            Tensor containing sampled points with shape (spheres, rays, 3).

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

    def create_nerf_model(self):
        """Create default NeRF model."""
        return self._create_nerf_model(model=SphereMoreViewsNeRF)

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)

        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
        weights = alpha * torch.cumprod(torch.cat(
            [torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,:-1]

        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        # rgb_map = torch.sum(raw[..., 3, None] * rgb, -1)

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map
