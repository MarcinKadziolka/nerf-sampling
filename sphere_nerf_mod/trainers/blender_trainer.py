"""Blender trainer module - trainer for blender data."""

from nerf_pytorch.trainers import Blender
import torch

from sphere_nerf_mod.lines import Lines
from sphere_nerf_mod.spheres import Spheres

from sphere_nerf_mod.models import (
    SphereNeRF, MoreDirectionVectorInfo, SphereMoreViewsNeRF,
    SphereWithoutViewsNeRF, SphereTwoRGB, SphereMoreViewsNeRFV2,
    SphereMoreViewsNeRFV6
)

from sphere_nerf_mod.utils import change_cartesian_to_spherical


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
        pts: [N_rand, n_spheres, 3D] -> [1024, 5, 3]
        -> [N_rand, n_spheres, 3D] + inf o kanale
        """
        rays_origins = rays_o
        m = min(rays_origins)
        mm = min(rays_d)
        rays_directions = rays_d
        rays = Lines(rays_origins, rays_directions)
        sphere_nerf_points = self.sample_points_on_spheres(
            rays
        ).swapaxes(0, 1)

        z_sphere = rays.transform_points_to_single_number_representation(
            sphere_nerf_points
        )

        z_vals, _ = torch.sort(torch.cat([z_vals, z_sphere], -1), -1)
        # _rays_d = rays_d[..., None, :]
        # pts = rays_o[..., None, :] + _rays_d * z_vals[..., :, None]
        pts = sphere_nerf_points

        #pts = change_cartesian_to_spherical(
        #    x=pts[:, :, 0],
        #    y=pts[:, :, 1],
        #    z=pts[:, :, 2]
        #)
        n_copies = 1
        n_copies_tensor = pts.unsqueeze(-1).repeat(1, 1, 1, n_copies)
        ind = torch.arange(n_copies).view(1, 1, 1, n_copies)
        _pts_shape = pts.shape
        fourth_dim = ind.expand(
            _pts_shape[0], _pts_shape[1], 1, n_copies
        )
        final_pts = torch.cat([n_copies_tensor, fourth_dim/n_copies], dim=2)
        final_pts = final_pts.swapaxes(2, 3)

        n_copies_tensor = viewdirs.unsqueeze(-1).repeat(1, 1, n_copies)
        ind = torch.arange(n_copies).view(1, 1, n_copies)
        ind = torch.ones(1, 1, n_copies)
        _viewdirs_shape = viewdirs.shape
        fourth_dim = ind.expand(
            _viewdirs_shape[0], 1, n_copies
        )
        final_viewdirs = torch.cat([n_copies_tensor, fourth_dim], dim=1)
        final_viewdirs = final_viewdirs.swapaxes(1, 2)

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(final_pts, final_viewdirs, run_fn)

        raw = torch.sum(raw, -2)
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
        return self._create_nerf_model(model=SphereMoreViewsNeRFV6)
