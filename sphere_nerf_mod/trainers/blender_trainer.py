"""Blender trainer module - trainer for blender data."""

from nerf_pytorch.trainers import Blender
import torch
from typing import Optional, Tuple
import torch.nn.functional as F

from sphere_nerf_mod.lines import Lines
from sphere_nerf_mod.spheres import Spheres

from sphere_nerf_mod.models import (
    SphereMoreViewsNeRFV2,
)

from sphere_nerf_mod.utils import (
    change_cartesian_to_spherical,
    calc_pts
)


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
        pytest: bool,
        rays_d: torch.Tensor,
        rays_o: torch.Tensor,
        network_fn,
        network_fine,
        network_query_fn,
        viewdirs: torch.Tensor,
        raw_noise_std: float,
        white_bkgd: bool,
        cartesian_to_spherical: bool = False,
        n_copies: int = 1,
        **kwargs
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor],
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, Optional[torch.Tensor]
    ]:
        """Sample points on given rays.

        This method defines how to sample points on given rays.
         The main idea behind SphereNerf is sampling points on spheres.
        The points are sampled by choosing the nearest point
        from the intersection points between the rays and the spheres
        surrounding the generated object.

        Args:
            pytest (bool): A flag indicating whether this is a pytest run.
            rays_d (torch.Tensor): The ray directions.
            rays_o (torch.Tensor): The ray origins.
            network_fn:
                The main network function.
            network_fine:
                An optional network for fine details.
            network_query_fn:
                The network query function.
            viewdirs (torch.Tensor):
                The view directions.
            raw_noise_std (float):
                The standard deviation for raw noise.
            white_bkgd (bool):
                A flag indicating whether the background is white.
            cartesian_to_spherical (bool, optional):
                A flag to convert points from Cartesian to spherical coord.
            n_copies (int, optional):
                The number of copies for each point.

        Returns:
            - None, None, None: Placeholder for optional return values.
            - rgb_map (torch.Tensor): The RGB map.
            - disp_map (torch.Tensor): The disparity map.
            - acc_map (torch.Tensor): The accumulation map.
            - raw (torch.Tensor): The raw output from the network.

        """
        # Rename the input variables for clarity
        rays_origins = rays_o
        rays_directions = rays_d
        rays = Lines(rays_origins, rays_directions)

        # Sample points on spheres and transform them into a
        # single number representation
        z_sphere, sphere_nerf_points = self.sample_points_on_spheres(
            rays
        )

        sphere_nerf_points = sphere_nerf_points.swapaxes(0, 1)  # [n_rays, m_spheres/n_points, 3]
        z_sphere = z_sphere.swapaxes(0, 1)  # [m_spheres/n_points, n_rays]

        z_vals, _ = torch.sort(z_sphere, -1)
        pts = sphere_nerf_points

        # Convert points from Cartesian to spherical coordinates if required
        if cartesian_to_spherical:
            pts = change_cartesian_to_spherical(
                x=pts[:, :, 0],
                y=pts[:, :, 1],
                z=pts[:, :, 2]
            )

        # Prepare copies of points and view directions
        final_pts, final_viewdirs = self._prepare_samples(
            pts, viewdirs, n_copies
        )

        # Choose the network function
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(final_pts, final_viewdirs, run_fn)

        # Sum the raw output along the second dimension
        raw = torch.sum(raw, -2)

        # Convert raw output to RGB, disparity, and accumulation maps
        rgb_map, disp_map, acc_map, _, _ = self.raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd,
            pytest=pytest
        )

        return None, None, None, rgb_map, disp_map, acc_map, raw, None

    @staticmethod
    def _prepare_samples(pts, viewdirs, n_copies):
        """Prepare samples to n canals.

        Returns:
        - final_pts -- n_copies of final_pts with fourth_dim
        - final_viewdirs -- n_copies of final_viewdirs with fourth_dim

        """
        n_copies_tensor = pts.unsqueeze(-1).repeat(1, 1, 1, n_copies)
        ind = torch.arange(n_copies).view(1, 1, 1, n_copies)
        _pts_shape = pts.shape
        fourth_dim = ind.expand(_pts_shape[0], _pts_shape[1], 1, n_copies)
        final_pts = torch.cat([n_copies_tensor, fourth_dim / n_copies], dim=2)
        final_pts = final_pts.swapaxes(2, 3)

        n_copies_tensor = viewdirs.unsqueeze(-1).repeat(1, 1, n_copies)
        ind = torch.ones(1, 1, n_copies)
        _viewdirs_shape = viewdirs.shape
        fourth_dim = ind.expand(_viewdirs_shape[0], 1, n_copies)
        final_viewdirs = torch.cat([n_copies_tensor, fourth_dim], dim=1)
        final_viewdirs = final_viewdirs.swapaxes(1, 2)
        return final_pts, final_viewdirs

    def sample_points_on_spheres(
        self,
        rays: Lines,
        point_coordinate_if_nan: float = 100
    ) -> (torch.Tensor, torch.Tensor):
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
            z_vals,
            Tensor containing sampled points with shape (spheres, rays, 3).

        """
        z_vals, intersection_points = rays.find_intersection_points_with_sphere(
            self.spheres
        )

        # select_closest_point_to_origin
        z_vals = z_vals[:, :, 0]

        # not intersection
        z_vals = torch.nan_to_num(z_vals, nan=100)

        selected_points = calc_pts(
            origin=rays.origin,
            t=z_vals,
            direction=rays.direction
        )

        return z_vals, selected_points

    def create_nerf_model(self):
        """Create NeRF model."""
        return self._create_nerf_model(model=SphereMoreViewsNeRFV2)

    def raw2outputs(
        self, raw, z_vals, rays_d,
        raw_noise_std=0, white_bkgd=False,
        pytest=False, as_in_original_nerf=False, **kwargs
    ):
        """Transforms model's predictions to semantically meaningful values.

        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
            raw_noise_std: std of noise added to raw
            white_bkgd: flag, if img have white background,
            pytest: flag, if it is tested
                (based on original nerf implementation)
            as_in_original_nerf: bool, flag, if rgb calc as in original
            nerf using integer
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples].
                Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.

        """

        if as_in_original_nerf:

            raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

            dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

            rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
            noise = 0.

            alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
            # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
            weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,
                              :-1]
            rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

            depth_map = torch.sum(weights * z_vals, -1)
            disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
            acc_map = torch.sum(weights, -1)

            if white_bkgd:
                rgb_map = rgb_map + (1. - acc_map[..., None])

        else:
            rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
            alpha = torch.ones_like(raw[..., 3]) * 0.25

            weights = alpha * torch.cumprod(torch.cat(
                [torch.ones(
                    (alpha.shape[0], 1)
                ), 1. - alpha + 1e-10], -1), -1)[:, :-1]

            _z_vals = torch.ones_like(z_vals)
            depth_map = torch.sum(weights * _z_vals, -1)
            disp_map = 1. / torch.max(
                1e-10 * torch.ones_like(depth_map),
                depth_map / torch.sum(weights, -1)
            )

            rgb_map = torch.mean(rgb, dim=1)
            acc_map = torch.sum(weights, -1)

        return rgb_map, disp_map, acc_map, weights, depth_map

