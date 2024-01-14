"""Blender trainer module - trainer for blender data."""

from typing import Optional, Tuple
from nerf_pytorch.trainers import Blender
import torch
import torch.nn.functional as F

from sphere_nerf_mod.lines import Lines
from sphere_nerf_mod.spheres import Spheres
from nerf_pytorch.nerf_utils import NeRF, create_nerf
from sphere_nerf_mod.samplers.sphere_sampler import SphereSampler

def get_closer_intersection_or_nan(origins: torch.Tensor, directions: torch.Tensor, sphere_z_vals: torch.Tensor):
     """
     Takes solution closer to the origin or if there are no solution returns NaN
     """
     x1 = sphere_z_vals[:, 0]
     x2 = sphere_z_vals[:, 1]
     x1 = torch.reshape(x1, [x1.shape[0], 1]) # [rays, 1]
     x2 = torch.reshape(x2, [x2.shape[0], 1]) # [rays, 1]
     closer_solution = torch.fmin(x1, x2)

     return origins + closer_solution*directions # [rays, 3]

def split_batch(intersections: torch.Tensor, rays_o: torch.Tensor, rays_d: torch.Tensor, viewdirs: torch.Tensor):
    """
    Sometimes there might be no intersection and result of quadratic equation is NaN
    In this case we need to split tensors based on NaN values in intersection tensor.
    Later tensor will be again merged based on computed mask to maintain order.
    """

    indices = torch.arange(intersections.size(0))
    
    isnan = torch.isnan(intersections)
    nan_mask = torch.any(isnan, dim=1)

    valid_indices = indices[~nan_mask]
    invalid_indices = indices[nan_mask]

    valid_rays_o = rays_o[~nan_mask]
    valid_rays_d = rays_d[~nan_mask]
    valid_intersections = intersections[~nan_mask]

    invalid_rays_o = rays_o[nan_mask]
    invalid_rays_d = rays_d[nan_mask]
    invalid_intersections = intersections[nan_mask]

    valid_viewdirs = viewdirs[~nan_mask]

    return valid_rays_o, valid_rays_d, valid_intersections, invalid_rays_o, invalid_rays_d, invalid_intersections, valid_indices, invalid_indices, valid_viewdirs


def merge_batch(valid_part, invalid_part, valid_indices, invalid_indices, dim):
    result = torch.empty(dim)
    result[valid_indices] = valid_part.float()
    if invalid_part.shape[0] > 0:
        result[invalid_indices] = invalid_part.float()

    return result

def swap_nans_in_missing_batch_elements(dim, N_samples):
    fake_rgb_map = torch.full(dim, 255)
    fake_disp_map = torch.full((dim[0],), 0)
    fake_acc_map = torch.full((dim[0],), 0)
    fake_weights = torch.full((dim[0], N_samples), 0)
    fake_depth_map = torch.full((dim[0],), 0)
    fake_raw = torch.full((dim[0], N_samples, 3), 0)
    fake_z_vals = torch.full((dim[0], N_samples), 0)

    return fake_rgb_map, fake_disp_map, fake_acc_map, fake_weights, fake_depth_map, fake_raw, fake_z_vals

class SphereSamplingTrainer(Blender.BlenderTrainer):
    """Trainer for blender data."""

    def __init__(
            self,
            spheres: Spheres = None,
            as_in_original_nerf = False,
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
        self.as_in_original_nerf = as_in_original_nerf
        # Fine network is not used in this approach, we aim to learn sampling network which points are valuable
        self.N_importance = 0

    def sample_main_points(
        self,
        N_samples,
        viewdirs,
        network_fn,
        network_query_fn,
        rays_o,
        rays_d,
        raw_noise_std,
        white_bkgd,
        pytest,
        sampling_network,
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
        rays = Lines(rays_o, rays_d)
        batch_size = rays_o.shape[0]

        sphere_z_vals, _ = rays.find_intersection_points_with_sphere(self.spheres)
        sphere_z_vals = torch.reshape(sphere_z_vals, sphere_z_vals.shape[1:])

        # Get closer intersection or NaN if there is no intersection
        intersections = get_closer_intersection_or_nan(rays_o, rays_d, sphere_z_vals)

        # Divide batch into two tensor with NaNs and without NaNs. Later it will be used to maintain order of values in tensor
        valid_rays_o, valid_rays_d, valid_intersections, _,\
        invalid_rays_d, _, valid_indices, invalid_indices, valid_viewdirs = split_batch(intersections, rays_o, rays_d, viewdirs)


        final_rgb_map, final_disp_map, final_acc_map, final_depth_map = None, None, None, None
        final_raw = None
        final_weights = None
        final_z_vals = None

        if N_samples > 0:

            pts, z_vals = sampling_network.forward(valid_rays_o, valid_rays_d, valid_intersections)

            print("SHAPE======")
            print(valid_rays_d.shape)
            print(invalid_rays_d.shape)
            raw = network_query_fn(pts, valid_viewdirs, network_fn)
            rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                pytest=pytest
            )
            fake_rgb_map, fake_disp_map, fake_acc_map, fake_weights, fake_depth_map, fake_raw, fake_z_vals = swap_nans_in_missing_batch_elements(invalid_rays_d.shape, N_samples)

            final_rgb_map = merge_batch(rgb_map, fake_rgb_map, valid_indices, invalid_indices, [batch_size, 3])
            final_disp_map = merge_batch(disp_map, fake_disp_map, valid_indices, invalid_indices, [batch_size])
            final_acc_map = merge_batch(acc_map, fake_acc_map, valid_indices, invalid_indices, [batch_size])
            final_weights = merge_batch(weights, fake_weights, valid_indices, invalid_indices, [batch_size, N_samples])
            final_depth_map = merge_batch(depth_map, fake_depth_map, valid_indices, invalid_indices, [batch_size])
            final_raw = merge_batch(raw, fake_raw, valid_indices, invalid_indices, [batch_size, N_samples, 4])
            final_z_vals = merge_batch(z_vals, fake_z_vals, valid_indices, invalid_indices, [batch_size, N_samples])

        return final_rgb_map, final_disp_map, final_acc_map, final_weights, final_depth_map, final_z_vals, final_weights, final_raw
 
    def create_nerf_model(self):
        """Custom create_nerf_model function that adds sampler to the model"""
        render_kwargs_train, render_kwargs_test, start, grad_vars, _ = create_nerf(self, NeRF)
        self.global_step = start
        self.start = start

        bds_dict = {
            'near': self.near,
            'far': self.far,
        }
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)

        # Inject sampler
        sampling_network = SphereSampler(
            output_channels=self.N_samples
        )

        # Add samplet to grad_vars
        grad_vars += list(sampling_network.parameters())


        # Create optimizer
        optimizer = torch.optim.Adam(params=grad_vars, lr=self.lrate, betas=(0.9, 0.999))

        # Add sampler to model dicts
        render_kwargs_train['sampling_network'] = sampling_network
        render_kwargs_test['sampling_network'] = sampling_network

        # Pick integral approximation method
        render_kwargs_train['as_in_original_nerf'] = self.as_in_original_nerf
        render_kwargs_test['as_in_original_nerf'] = self.as_in_original_nerf

        return optimizer, render_kwargs_train, render_kwargs_test 
    
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