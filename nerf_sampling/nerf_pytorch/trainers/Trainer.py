import os
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm, trange

from nerf_sampling.nerf_pytorch import loss_functions, nerf_utils, utils
from nerf_sampling.nerf_pytorch.loss_functions import SamplerLossInput
from nerf_sampling.nerf_pytorch.visualize import visualize_rays_pts


class Trainer:
    def __init__(
        self,
        dataset_type,
        basedir,
        expname,
        no_batching,
        datadir,
        device="cpu",
        render_test=False,
        config_path=None,
        N_rand=32 * 32 * 4,
        render_only=False,
        chunk=1024 * 32,
        render_factor=0,
        multires=10,
        i_embed=0,
        multires_views=4,
        netchunk=1024 * 64,
        lrate=5e-4,
        lrate_decay=250,
        use_viewdirs=True,
        N_importance=0,
        netdepth=8,
        netwidth=256,
        netdepth_fine=8,
        netwidth_fine=256,
        ft_path=None,
        perturb=1.0,
        raw_noise_std=0.0,
        N_samples=64,
        lindisp=True,
        precrop_iters=0,
        precrop_frac=0.5,
        i_weights=10000,
        i_testset=100,
        i_video=5000,
        i_print=100,
        input_dims_embed: int = 1,
        save_train_set_render: bool = True,
        sampler_lr: float = 0.0001,
        train_sampler_only: bool = False,
        trial: Optional[optuna.trial.Trial] = None,
        single_image=False,
        single_ray=False,
    ):
        self.start = None
        self.dataset_type = dataset_type
        self.render_test = render_test
        self.render_only = render_only
        self.basedir = basedir
        self.expname = expname
        self.config_path = config_path
        self.device = device
        self.chunk = chunk
        self.render_factor = render_factor
        self.N_rand = N_rand
        self.no_batching = no_batching
        self.use_batching = not self.no_batching
        self.datadir = datadir
        self.multires = multires
        self.i_embed = i_embed
        self.multires_views = multires_views
        self.netwidth_fine = netwidth_fine
        self.netchunk = netchunk
        self.lrate = lrate
        self.lrate_decay = lrate_decay
        self.use_viewdirs = use_viewdirs
        self.N_importance = N_importance
        self.netdepth = netdepth
        self.netwidth = netwidth
        self.netdepth_fine = netdepth_fine
        self.netwidth_fine = netwidth_fine
        self.ft_path = ft_path
        self.perturb = perturb
        self.raw_noise_std = raw_noise_std
        self.N_samples = N_samples
        self.lindisp = lindisp
        self.precrop_iters = precrop_iters
        self.precrop_frac = precrop_frac
        self.i_weights = i_weights
        self.i_testset = i_testset
        self.i_video = i_video
        self.i_print = i_print
        self.input_dims_embed = input_dims_embed
        self.save_train_set_render = save_train_set_render
        self.no_reload = False
        self.K = None
        self.global_step = None
        self.W = None
        self.H = None
        self.c2w = None

        self.sampler_lr = sampler_lr
        self.train_sampler_only = train_sampler_only
        self.trial = trial

        self.single_image = single_image
        self.single_ray = single_ray
        print(f"{self}")
        print(f"{self.N_samples=}")
        print(f"{self.N_importance=}")
        print(f"{self.sampler_lr=}")
        print(f"{self.train_sampler_only=}")

    def load_data(self):
        """Load data and prepare poses."""
        pass

    def cast_intrinsics_to_right_types(self, hwf):
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        if self.K is None:
            self.K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

        self.H = H
        self.W = W
        return hwf

    def create_log_dir_and_copy_the_config_file(self):
        basedir = self.basedir
        expname = self.expname
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        f = os.path.join(basedir, expname, "args.txt")
        with open(f, "w") as file:
            dict = self.__dict__
            for arg in dict:
                file.write("{} = {}\n".format(arg, dict[arg]))
        if self.config_path is not None:
            f = os.path.join(basedir, expname, "config.txt")
            with open(f, "w") as file:
                file.write(open(self.config_path, "r").read())

    def create_nerf_model(self):
        return self._create_nerf_model(model=nerf_utils.run_nerf_helpers.NeRF)

    def _create_nerf_model(self, model):
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = (
            nerf_utils.create_nerf(self, model=model)
        )
        self.global_step = start
        self.start = start

        bds_dict = {
            "near": self.near,
            "far": self.far,
        }
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)

        return optimizer, render_kwargs_train, render_kwargs_test

    def render(
        self, render_test, images, i_test, render_poses, hwf, render_kwargs_test
    ):
        with torch.no_grad():
            if render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(
                self.basedir,
                self.expname,
                "renderonly_{}_{:06d}".format(
                    "test" if render_test else "path", self.global_step
                ),
            )

            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", render_poses.shape)

            rgbs, _ = nerf_utils.render_path(
                render_poses,
                hwf,
                self.K,
                self.chunk,
                render_kwargs_test,
                step=self.global_step,
                gt_imgs=images,
                savedir=testsavedir,
                render_factor=self.render_factor,
            )
            print("Done rendering", testsavedir)
            imageio.mimwrite(
                os.path.join(testsavedir, "video.mp4"),
                nerf_utils.run_nerf_helpers.to8b(rgbs),
                fps=30,
                quality=8,
            )

    def prepare_raybatch_tensor_if_batching_random_rays(self, poses, images, i_train):
        i_batch = None
        rays_rgb = None

        if self.use_batching:
            # For random ray batching
            print("get rays")
            rays = np.stack(
                [
                    nerf_utils.run_nerf_helpers.get_rays_np(self.H, self.W, self.K, p)
                    for p in poses[:, :3, :4]
                ],
                0,
            )  # [N, ro+rd, H, W, 3]
            print("done, concats")
            rays_rgb = np.concatenate(
                [rays, images[:, None]], 1
            )  # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = np.transpose(
                rays_rgb, [0, 2, 3, 1, 4]
            )  # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
            rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb = rays_rgb.astype(np.float32)
            print("shuffle rays")
            np.random.shuffle(rays_rgb)

            print("done")
            i_batch = 0

        # Move training data to GPU
        if self.use_batching:
            images = torch.tensor(images).to(self.device)
        poses = torch.tensor(poses).to(self.device)
        if self.use_batching:
            rays_rgb = torch.tensor(rays_rgb).to(self.device)

        return images, poses, rays_rgb, i_batch

    def log(
        self,
        i: int,
        render_poses,
        hwf,
        poses,
        i_test,
        i_train,
        images,
        loss,
        sampler_loss,
        psnr,
        render_kwargs_train,
        render_kwargs_test,
        optimizer,
        sampling_optimizer,
        logs,
    ):
        """Handle logging and saving logic."""
        if i % self.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                self.basedir, self.expname, "testset_{:06d}".format(i)
            )
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", poses[i_test].shape)
            with torch.no_grad():
                target_s = images[i_test]
                rgbs, _ = nerf_utils.render_path(
                    torch.tensor(poses[i_test]).to(self.device),
                    hwf,
                    self.K,
                    self.chunk,
                    render_kwargs_test,
                    step=self.global_step,
                    wandb_log=True,
                    gt_imgs=target_s,
                    savedir=testsavedir,
                )

                img_loss = nerf_utils.run_nerf_helpers.img2mse(
                    torch.tensor(rgbs), torch.tensor(target_s)
                )
                loss = img_loss
                test_psnr = nerf_utils.run_nerf_helpers.mse2psnr(img_loss)

            print("Saved test set")

        if i % self.i_testset == 0 and i > 0 and self.save_train_set_render:
            testsavedir = os.path.join(
                self.basedir, self.expname, "trainset_{:06d}".format(i)
            )
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", poses[i_train[:10]].shape)
            with torch.no_grad():
                target_s = images[i_test]
                rgbs, _ = nerf_utils.render_path(
                    torch.tensor(poses[i_train[:10]]).to(self.device),
                    hwf,
                    self.K,
                    self.chunk,
                    render_kwargs_test,
                    step=self.global_step,
                    gt_imgs=target_s,
                    savedir=testsavedir,
                )

            print("Saved train set")

        if i % self.i_weights == 0:
            path = os.path.join(self.basedir, self.expname, "{:06d}.tar".format(i))
            utils.save_state(
                global_step=self.global_step,
                network_fn=render_kwargs_train["network_fn"],
                network_fine=render_kwargs_train["network_fine"],
                optimizer=optimizer,
                sampling_network=render_kwargs_train["sampling_network"],
                sampling_optimizer=sampling_optimizer,
                path=path,
            )
        if i % self.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = nerf_utils.render_path(
                    render_poses,
                    hwf,
                    self.K,
                    self.chunk,
                    render_kwargs_test,
                    step=self.global_step,
                )
            print("Done, saving", rgbs.shape, disps.shape)
            moviebase = os.path.join(
                self.basedir, self.expname, "{}_spiral_{:06d}_".format(self.expname, i)
            )
            imageio.mimwrite(
                moviebase + "rgb.mp4",
                nerf_utils.run_nerf_helpers.to8b(rgbs),
                fps=30,
                quality=8,
            )
            imageio.mimwrite(
                moviebase + "disp.mp4",
                nerf_utils.run_nerf_helpers.to8b(disps / np.max(disps)),
                fps=30,
                quality=8,
            )

        if i % self.i_print == 0:
            sampler_density = logs["sampler_density"]
            sampler_alphas = logs["sampler_alphas"]
            sampler_weights = logs["sampler_weights"]
            fine_density = torch.max(logs["fine_density"], 1, keepdims=True)[0]
            sampler_loss = sampler_loss.item()
            info = f"Iter: {i} Loss: {loss.item()}, Sampler Loss: {sampler_loss}, SMean/FMean: {torch.mean(sampler_density):.2f}/{torch.mean(fine_density):.2f}, PSNR: {psnr.item():.5f}"
            wandb.log(
                {
                    "Loss": loss.item(),
                    "Sampler loss": sampler_loss,
                    "PSNR": psnr.item(),
                    "Mean density": torch.mean(sampler_density),
                    "Max density": torch.max(sampler_density),
                    "Mean alphas": torch.mean(sampler_alphas),
                    "Mean weights": torch.mean(sampler_weights),
                },
                step=self.global_step,
            )
            tqdm.write(info)
            f = os.path.join(self.basedir, self.expname, "psnr.txt")
            with open(f, "a") as file:
                file.write(f"{info}\n")

            if self.trial is not None:
                self.trial.report(psnr.item(), self.global_step)
                if self.trial.should_prune():
                    wandb.run.summary["state"] = "pruned"
                    wandb.finish(quiet=True)
                    raise optuna.exceptions.TrialPruned()

    def sample_random_ray_batch(self, rays_rgb, i_batch, i_train, images, poses, i):
        if self.use_batching:
            # Random over all images
            batch = rays_rgb[i_batch : i_batch + self.N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += self.N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            if self.single_image:
                img_i = 0  # set fixed image
            else:
                img_i = np.random.choice(i_train)

            target = images[img_i]
            target = torch.tensor(target)
            pose = poses[img_i, :3, :4]
            self.c2w = pose.clone().detach()

            if self.N_rand is not None:
                rays_o, rays_d = nerf_utils.run_nerf_helpers.get_rays(
                    self.H, self.W, self.K, self.c2w
                )  # (H, W, 3), (H, W, 3)

                if i < self.precrop_iters:
                    dH = int(self.H // 2 * self.precrop_frac)
                    dW = int(self.W // 2 * self.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(
                                self.H // 2 - dH, self.H // 2 + dH - 1, 2 * dH
                            ),
                            torch.linspace(
                                self.W // 2 - dW, self.W // 2 + dW - 1, 2 * dW
                            ),
                        ),
                        -1,
                    )
                    if i == self.start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {self.precrop_iters}"
                        )
                else:
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(0, self.H - 1, self.H),
                            torch.linspace(0, self.W - 1, self.W),
                        ),
                        -1,
                    )  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                if self.single_ray:
                    select_inds = torch.tensor([91])
                    # (N_rand,)
                else:
                    select_inds = np.random.choice(
                        coords.shape[0], size=[self.N_rand], replace=False
                    )  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[
                    select_coords[:, 0], select_coords[:, 1]
                ]  # (N_rand, 3)

        return rays_rgb, i_batch, batch_rays, target_s

    def core_optimization_loop(
        self,
        optimizer,
        sampling_optimizer,
        render_kwargs_train,
        batch_rays,
        i,
        target_s,
    ):
        """Runs rendering and backpropagates."""
        rgb, disp, acc, extras = nerf_utils.render(
            self.H,
            self.W,
            self.K,
            chunk=self.chunk,
            rays=batch_rays,
            verbose=i < 10,
            retraw=True,
            **render_kwargs_train,
        )

        # optimizer.zero_grad()
        sampling_optimizer.zero_grad()
        img_loss = nerf_utils.run_nerf_helpers.img2mse(rgb, target_s)
        loss = img_loss

        psnr = nerf_utils.run_nerf_helpers.mse2psnr(img_loss)

        psnr0 = None
        if "rgb0" in extras:
            img_loss0 = nerf_utils.run_nerf_helpers.img2mse(extras["rgb0"], target_s)
            loss = loss + img_loss0
            psnr0 = nerf_utils.run_nerf_helpers.mse2psnr(img_loss0)
        sampler_loss = F.mse_loss(extras["pts"], extras["max_pts"])
        sampler_loss.backward()
        # loss.backward()
        # optimizer.step()
        sampling_optimizer.step()
        logs = {
            "density": extras["density"],
            "alphas": extras["alphas"],
            "weights": extras["weights"],
        }
        return logs, loss, sampler_loss, psnr, psnr0

    def update_learning_rate(self, optimizer):
        decay_rate = 0.1
        decay_steps = self.lrate_decay * 1000
        new_lrate = self.lrate * (decay_rate ** (self.global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate

    def _sample_points(
        self, z_vals_mid, weights, perturb, pytest, rays_d, rays_o, n_importance=None
    ):

        if n_importance is None:
            n_importance = self.N_importance
        z_vals_mid = z_vals_mid
        weights = weights
        perturb = perturb
        pytest = pytest

        z_samples = nerf_utils.run_nerf_helpers.sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            n_importance,
            det=(perturb == 0.0),
            pytest=pytest,
        )

        z_samples = z_samples.detach()

        pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_samples[..., :, None]
        )  # [N_rays, N_importance, 3]
        return z_samples, pts

    def sample_coarse_points(
        self,
        near,
        far,
        perturb,
        N_rays,
        N_samples,
        viewdirs,
        network_fn,
        network_query_fn,
        rays_o,
        rays_d,
        raw_noise_std,
        white_bkgd,
        pytest,
        lindisp,
        **kwargs,
    ):

        rgb_map, disp_map, acc_map, depth_map, alphas_map = None, None, None, None, None
        raw = None
        weights = None
        z_vals = None

        if N_samples > 0:
            t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
            if not lindisp:
                z_vals = near * (1.0 - t_vals) + far * t_vals
            else:
                z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)

            z_vals = z_vals.expand([N_rays, N_samples])

            if perturb > 0.0:
                # get intervals between samples
                mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape)

                # Pytest, overwrite u with numpy's fixed random numbers
                if pytest:
                    np.random.seed(0)
                    t_rand = np.random.rand(*list(z_vals.shape))
                    t_rand = torch.tensor(t_rand)

                z_vals = lower + (upper - lower) * t_rand

            pts = (
                rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            )  # [N_rays, N_samples, 3]
            raw = network_query_fn(pts, viewdirs, network_fn)

            rgb_map, disp_map, acc_map, depth_map, density, alphas, weights = (
                self.raw2outputs(
                    raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
                )
            )

        return (
            rgb_map,
            disp_map,
            acc_map,
            weights,
            depth_map,
            z_vals,
            weights,
            raw,
            alphas_map,
        )

    def sample_fine_points(
        self,
        z_vals,
        weights,
        perturb,
        pytest,
        rays_d,
        rays_o,
        rgb_map,
        disp_map,
        acc_map,
        network_fn,
        network_fine,
        network_query_fn,
        viewdirs,
        raw_noise_std,
        white_bkgd,
    ):
        rgb_map_0, disp_map_0, acc_map_0, raw = None, None, None, None
        z_samples = None

        if self.N_importance > 0:
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples, pts = self._sample_points(
                z_vals_mid=z_vals_mid,
                weights=weights,
                perturb=perturb,
                pytest=pytest,
                rays_o=rays_o,
                rays_d=rays_d,
            )

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            run_fn = network_fn if network_fine is None else network_fine

            raw = network_query_fn(pts, viewdirs, run_fn)

            rgb_map, disp_map, acc_map, depth_map, density, alphas, weights = (
                self.raw2outputs(
                    raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
                )
            )

        return (
            rgb_map_0,
            disp_map_0,
            acc_map_0,
            rgb_map,
            disp_map,
            acc_map,
            raw,
            z_samples,
            pts,
            density,
            alphas,
            weights,
        )

    def train(self, N_iters=200000 + 1):
        hwf, poses, i_test, i_val, i_train, images, render_poses = self.load_data()

        if self.render_test:
            render_poses = np.array(poses[i_test])
            render_poses = torch.tensor(render_poses).to(self.device)

        hwf = self.cast_intrinsics_to_right_types(hwf=hwf)
        self.create_log_dir_and_copy_the_config_file()
        optimizer, sampling_optimizer, render_kwargs_train, render_kwargs_test = (
            self.create_nerf_model()
        )
        if self.train_sampler_only:
            if render_kwargs_train["network_fn"] is not None:
                utils.freeze_model(render_kwargs_train["network_fn"])
            if render_kwargs_train["network_fine"] is not None:
                utils.freeze_model(render_kwargs_train["network_fine"])

        if self.render_only:
            self.render(
                self.render_test, images, i_test, render_poses, hwf, render_kwargs_test
            )
            return self.render_only

        images, poses, rays_rgb, i_batch = (
            self.prepare_raybatch_tensor_if_batching_random_rays(poses, images, i_train)
        )

        print("Begin")
        print("TRAIN views are", i_train)
        print("TEST views are", i_test)
        print("VAL views are", i_val)

        start = self.start + 1
        for i in trange(start, N_iters):
            rays_rgb, i_batch, batch_rays, target_s = self.sample_random_ray_batch(
                rays_rgb, i_batch, i_train, images, poses, i
            )

            logs, loss, sampler_loss, psnr, psnr0 = self.core_optimization_loop(
                optimizer,
                sampling_optimizer,
                render_kwargs_train,
                batch_rays,
                i,
                target_s,
            )
            self.update_learning_rate(optimizer)

            self.log(
                i=i,
                render_poses=render_poses,
                hwf=hwf,
                poses=poses,
                i_test=i_test,
                i_train=i_train,
                images=images,
                loss=loss,
                sampler_loss=sampler_loss,
                psnr=psnr,
                render_kwargs_train=render_kwargs_train,
                render_kwargs_test=render_kwargs_test,
                optimizer=optimizer,
                sampling_optimizer=sampling_optimizer,
                logs=logs,
            )

            self.global_step += 1
        return psnr

    def run_network(
        self, inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64
    ):
        """Prepares inputs and applies network 'fn'."""
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = embed_fn(inputs_flat)

        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        outputs_flat = nerf_utils.batchify(fn, netchunk)(embedded)
        outputs = torch.reshape(
            outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
        )
        return outputs
