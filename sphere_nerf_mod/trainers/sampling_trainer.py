from nerf_pytorch.trainers import Blender
import torch
from nerf_pytorch.nerf_utils import NeRF, create_nerf
from sphere_nerf_mod.samplers.baseline_sampler import BaselineSampler
from safetensors.torch import save_file
import os

class SamplingTrainer(Blender.BlenderTrainer):
    """Trainer for blender data."""

    def __init__(
            self,
            as_in_original_nerf = False,
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

        render_kwargs_train['model_mode'] = 'train'
        render_kwargs_test['model_mode'] = 'test'

        return optimizer, render_kwargs_train, render_kwargs_test
    
    def save_rays_data(self, rays_o, pts):
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

            self.save_rays_data(rays_o, pts)

            raw = network_query_fn(pts, viewdirs, network_fn)
            rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                pytest=pytest
            )
        return rgb_map, disp_map, acc_map, weights, depth_map, z_vals, weights, raw