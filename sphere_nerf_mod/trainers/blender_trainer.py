"""Blender trainer module - trainer for blender data."""

from nerf_pytorch.trainers import Blender
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from sphere_nerf_mod.lines import Lines
from sphere_nerf_mod.spheres import Spheres

from sphere_nerf_mod.models import (
    SphereNeRF, MoreDirectionVectorInfo, SphereMoreViewsNeRF,
    sphereWithoutViews, SphereTwoRGB, SphereMoreViewsNeRFV2,
    SphereMoreViewsNeRFV6
)

from sphere_nerf_mod.utils import change_cartesian_to_spherical
from sphere_nerf_mod.utils import reflected


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
    ) -> Tuple[Optional[torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor],
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
                A flag to convert points from Cartesian to spherical coordinates.
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

        #rays_o_reflected, rays_d_reflected = reflected(
        #    spheres=self.spheres,
        #    img_height=self.H,
        #    img_width=self.W,
        #    base_camera=self.c2w
        #)

        reflected_rays = Lines(
            torch.zeros_like(rays_origins),
            -rays_directions
        )

        # Sample points on spheres and transform them into a
        # single number representation
        sphere_nerf_points = self.sample_points_on_spheres(
            reflected_rays
        ).swapaxes(0, 1)  # [n_rays, m_spheres/n_points, 3]

        z_sphere = reflected_rays.transform_points_to_single_number_representation(
            sphere_nerf_points
        )

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
        return self._create_nerf_model(model=SphereMoreViewsNeRFV2)

    def raw2outputs(
        self,
        raw: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0,
        white_bkgd: bool = False,
        pytest: bool = False
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Transforms model's predictions to semantically meaningful values.

        Args:
            raw (torch.Tensor):
            Model's prediction, shape [num_rays, num_samples along ray, 4].
            z_vals (torch.Tensor):
            Integration time, shape [num_rays, num_samples along ray].
            rays_d (torch.Tensor):
            Direction of each ray, shape [num_rays, 3].
            raw_noise_std (float, optional):
            Standard deviation for raw noise (default: 0).
            white_bkgd (bool, optional):
            Flag indicating white background (default: False).
            pytest (bool, optional):
            Flag indicating whether this is a pytest run (default: False).

        Returns:
            Tuple of torch.Tensors:
            - rgb_map (torch.Tensor):
                Estimated RGB color of a ray, shape [num_rays, 3].
            - disp_map (torch.Tensor):
                Disparity map, the inverse of the depth map, shape [num_rays].
            - acc_map (torch.Tensor):
                Sum of weights along each ray, shape [num_rays].
            - weights (torch.Tensor):
                Weights assigned to each sampled color,
                shape [num_rays, num_samples].
            - depth_map (torch.Tensor):
                Estimated distance to object, shape [num_rays].
        """
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

        # Compute distances between z values and scale by ray direction
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1
        )  # [num_rays, num_samples]
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # Apply sigmoid to the first 3 values of raw to get RGB
        rgb = torch.sigmoid(raw[..., :3])  # [num_rays, num_samples, 3]

        # Generate noise based on raw_noise_std if applicable
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

            # Overwrite randomly sampled data if in pytest mode
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)

        # Calculate alpha values using raw2alpha function
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [num_rays, num_samples]

        # Compute weights using cumulative product
        weights = alpha * torch.cumprod(torch.cat(
            [torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]

        # Compute the final RGB map
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [num_rays, 3]
        #n_spheres = raw.shape[1]
        #rgb_map = torch.sum(raw[..., 3, None] * rgb, -1) / n_spheres
        #aa = rgb_map.max()
        #bb
        #rgb_map = torch.where(rgb_map < 0, 0, rgb_map)

        # Compute the depth map and disparity map
        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(
            1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
        )

        # Compute the accumulation map
        acc_map = torch.sum(weights, -1)

        # If using a white background, adjust the RGB map
        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map