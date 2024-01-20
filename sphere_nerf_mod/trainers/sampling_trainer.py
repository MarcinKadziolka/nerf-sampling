from nerf_pytorch.trainers import Blender
import torch
from nerf_pytorch.nerf_utils import NeRF, create_nerf
from sphere_nerf_mod.samplers.baseline_sampler import BaselineSampler
from safetensors.torch import save_file
import os
import torch.nn.functional as F
import numpy as np

class SamplingTrainer(Blender.BlenderTrainer):
    """Trainer for blender data."""

    def __init__(
            self,
            as_in_original_nerf = False,
            use_noise = True,
            noise_size = 10,
            **kwargs
    ):
        """Initialize the sampling trainer.

        In addition to original nerf_pytorch BlenderTrainer,
        the trainer contains the spheres used in the training process.
        """
        super().__init__(
            **kwargs
        )
        self.as_in_original_nerf = as_in_original_nerf
        # Fine network is not used in this approach, we aim to learn sampling network which points are valuable
        self.N_importance = 0
        self.use_noise = use_noise
        self.noise_size = noise_size

        if use_noise:
            print(f"Using noise {self.noise_size}")
   
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
        sampling_network = BaselineSampler(
            output_channels=self.N_samples,
            noise_size=self.noise_size if self.use_noise else None
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

        render_kwargs_train['model_mode'] = 'train'
        render_kwargs_test['model_mode'] = 'test'

        return optimizer, render_kwargs_train, render_kwargs_test
    
    def save_rays_data(self, rays_o, pts, alpha):
        """
        Saves rays data for later visualization
        """
        n_rays = rays_o.shape[0]
        if self.global_step % self.i_testset != 0:
            return
        
        filename = os.path.join(self.basedir, self.expname, f'{self.expname}_{self.global_step}.safetensors')

        tensors = {
            'origins': rays_o.contiguous(),
            'pts': pts.contiguous(),
            'alpha': alpha.contiguous(),
        }

        save_file(tensors, filename)

    
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
    ):
        """
        Custom method for sampling `N_samples` points from coarse network. 
        Uses sampling network to get points on the ray
        """
        rgb_map, disp_map, acc_map, depth_map = None, None, None, None
        raw = None
        weights = None
        z_vals = None

        if N_samples > 0:

            pts, z_vals = sampling_network.forward(rays_o, rays_d)


            raw = network_query_fn(pts, viewdirs, network_fn)
            rgb_map, disp_map, acc_map, weights, depth_map, alpha = self.raw2outputs(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                pytest=pytest
            )

            self.save_rays_data(rays_o, pts, alpha)

        return rgb_map, disp_map, acc_map, weights, depth_map, z_vals, weights, raw
    
    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, **kwargs):
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
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,
                          :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map, alpha