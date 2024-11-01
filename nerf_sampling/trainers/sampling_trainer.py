import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from nerf_sampling.nerf_pytorch import nerf_utils, utils
from nerf_sampling.nerf_pytorch.nerf_utils import create_nerf
from nerf_sampling.nerf_pytorch.run_nerf_helpers import NeRF
from nerf_sampling.nerf_pytorch.trainers import Blender
from nerf_sampling.depth_nets.depth_net import DepthNet


class DepthNetTrainer(Blender.BlenderTrainer):
    """Trainer for blender data."""

    def __init__(
        self,
        distance=None,
        sampling_mode=None,
        n_depth_samples=None,
        depth_net_path: Optional[str] = None,
        n_layers: int = 6,
        layer_width: int = 256,
        sphere_radius: float = 2.0,
        **kwargs,
    ):
        """Initialize the sampling trainer.

        In addition to original nerf_pytorch BlenderTrainer,
        the trainer contains the spheres used in the training process.

        Args:
            n_layers: number of layers for sampling network.
                denotes number of layers both individually of origin and direction
                and number of layers for concatenated embeddings of origin and direction
            layer_width: number of neurons in each layer.
            **kwargs: other arguments passed to BlenderTrainer or Trainer.
        """
        self.n_layers = n_layers
        self.layer_width = layer_width
        self.depth_net_path = depth_net_path
        self.sphere_radius = sphere_radius
        self.distance = distance
        self.n_depth_samples = n_depth_samples
        self.sampling_mode = sampling_mode
        print(f"{self.n_layers=}")
        print(f"{self.layer_width=}")
        super().__init__(**kwargs)
        # Fine network is not used in this approach, we aim to train sampling network which points are valuable

    def create_nerf_model(self):
        """Custom create_nerf_model function that adds depth_net to the model."""
        render_kwargs_train, render_kwargs_test, _start, grad_vars, optimizer = (
            create_nerf(self, NeRF)
        )

        bds_dict = {
            "near": self.near,
            "far": self.far,
        }
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)

        # Inject depth_net
        hidden_sizes = [self.layer_width for _ in range(self.n_layers)]
        cat_hidden_sizes = [self.layer_width for _ in range(self.n_layers)]
        depth_network = DepthNet(
            hidden_sizes=hidden_sizes,
            cat_hidden_sizes=cat_hidden_sizes,
            sphere_radius=self.sphere_radius,
        )

        sampling_params = list(depth_network.parameters())

        sampling_optimizer = torch.optim.Adam(
            params=sampling_params, lr=self.depth_net_lr
        )

        # Load checkpoints
        basedir = self.basedir
        expname = self.expname
        if self.depth_net_path is not None and self.depth_net_path != "None":
            ckpts = [self.depth_net_path]
        else:
            ckpts = [
                os.path.join(basedir, expname, f)
                for f in sorted(os.listdir(os.path.join(basedir, expname)))
                if "tar" in f
            ]
        print("Found ckpts", ckpts)
        start = None
        if len(ckpts) > 0 and not self.no_reload:
            ckpt_path = ckpts[-1]
            print("Reloading from", ckpt_path)
            ckpt = torch.load(ckpt_path)
            start = ckpt["global_step"]
            # Load model
            utils.load_depth_network(depth_network, sampling_optimizer, ckpt)

        if start is not None:
            self.global_step = start  # start
            self.start = start  # start
        else:
            self.global_step = 0
            self.start = 0

        # Add depth_net to model dicts
        render_kwargs_train["depth_network"] = depth_network
        render_kwargs_test["depth_network"] = depth_network

        render_kwargs_train["model_mode"] = "train"
        render_kwargs_test["model_mode"] = "test"

        return (
            optimizer,
            sampling_optimizer,
            render_kwargs_train,
            render_kwargs_test,
        )

    def save_rays_data(self, rays_o, pts, alpha):
        """Saves rays data for later visualization."""
        n_rays = rays_o.shape[0]

        filename = os.path.join(
            self.basedir, self.expname, f"{self.expname}_{self.global_step}.safetensors"
        )

        tensors = {
            "origins": rays_o.contiguous(),
            "pts": pts.contiguous(),
            "alpha": alpha.contiguous(),
        }

        save_file(tensors, filename)

    def sample_main_points(
        self,
        rays_o,
        rays_d,
        depth_network,
    ):
        """Custom method for sampling `N_samples` points from coarse network.

        Uses sampling network to get points on the ray
        """
        pts, z_vals = depth_network.forward(rays_o, rays_d)
        return pts, z_vals

    def raw2outputs(
        self,
        raw,
        z_vals,
        rays_d,
        raw_noise_std=0,
        white_bkgd=True,
        pytest=False,
        **kwargs,
    ):
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
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists, torch.tensor([1e10]).expand(dists[..., :1].shape)], -1
        )  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        noise = 0.0
        if raw_noise_std > 0.0:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.tensor(noise)

        density = raw[..., 3]
        alphas = nerf_utils.raw2alpha(density + noise, dists)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        # weights = alpha * cumprod(1.0 - alpha)
        # weights = (1 - exp(-sigma_i * dists_i)) * cumprod(1.0 - (1 - exp(-sigma_i * dists_i)))
        weights = (
            alphas
            * torch.cumprod(
                torch.cat([torch.ones((alphas.shape[0], 1)), 1.0 - alphas + 1e-10], -1),
                -1,
            )[:, :-1]
        )  # [N_rays, N_samples]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1.0 / torch.max(
            1e-10 * torch.ones_like(depth_map),
            depth_map / (torch.sum(weights, -1) + 1e-10),
        )
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        if weights.shape[-1] == 0:
            rgb_map = torch.sum(rgb, -2)  # [N_rays, 3]

        return (
            rgb_map,
            disp_map,
            acc_map,
            depth_map,
            density,
            alphas,
            weights,
        )
