"""Utility functions for running NeRF training."""

import os
import pickle
import random
import time
from typing import Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from nerf_sampling.nerf_pytorch import run_nerf_helpers, utils, visualize
from nerf_sampling.nerf_pytorch.loss_functions import gaussian_log_likelihood
from nerf_sampling.nerf_pytorch.utils import sample_points_around_mean
from nerf_sampling.depth_nets.depth_net import DepthNet

np.random.seed(0)
DEBUG = False


def raw2alpha(raw, dists):
    """Converts raw density to alpha values.

    In NeRF paper (https://arxiv.org/pdf/2003.08934)
    section 4, page 6, equation (3)

    ^C(r) = sum(T_i * (1 - exp(-sigma_i * delta_i)) * color_i),

    where T_i = exp(-sum(sigma_j * delta_j)) -> probability of not hitting anything
    and delta_i = t_{i+1} - t_i -> the distance between adjacent samples

    This function for calculating ^C(r) from the set of (color_i, sigma_i) values
    is trivially differentiable and reduces to traditional alpha compositing
    with alpha values alpha_i = 1 − exp(−sigma_i * delta_i)
    """
    return 1.0 - torch.exp(-F.relu(raw) * dists)


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat(
            [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0
        )

    return ret


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_returned = {}
    for i in range(0, rays_flat.shape[0], chunk):
        torch.cuda.empty_cache()
        returned = render_rays(rays_flat[i : i + chunk], **kwargs)
        for key in returned:
            if key not in all_returned:
                all_returned[key] = []
            all_returned[key].append(returned[key])

    all_returned = {key: torch.cat(all_returned[key], 0) for key in all_returned}
    return all_returned


def batchify_rays_test(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_returned = {}
    for i in range(0, rays_flat.shape[0], chunk):
        torch.cuda.empty_cache()
        returned = render_rays_test(rays_flat[i : i + chunk], **kwargs)
        for key in returned:
            if key not in all_returned:
                all_returned[key] = []
            all_returned[key].append(returned[key])

    all_returned = {key: torch.cat(all_returned[key], 0) for key in all_returned}
    return all_returned


def render(
    H,
    W,
    K,
    chunk=1024 * 32,
    rays: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    c2w=None,
    ndc=True,
    near=0.0,
    far=1.0,
    use_viewdirs=False,
    c2w_staticcam=None,
    **kwargs,
):
    """Render rays.

    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
        camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      alphas_map:
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = run_nerf_helpers.get_rays(H, W, K, c2w)
    elif rays is not None:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = run_nerf_helpers.get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = run_nerf_helpers.ndc_rays(H, W, K[0][0], 1.0, rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(
        rays_d[..., :1]
    )
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_returned = batchify_rays(rays, chunk, **kwargs)
    for key in all_returned:
        k_sh = list(sh[:-1]) + list(all_returned[key].shape[1:])
        all_returned[key] = torch.reshape(all_returned[key], k_sh)

    key_extract = ["depth_net_rgb_map", "depth_net_disp_map"]
    ret_list = [all_returned[key] for key in key_extract]
    ret_dict = {
        key: all_returned[key] for key in all_returned if key not in key_extract
    }
    ret_dict["rays_o"] = rays_o
    ret_dict["rays_d"] = rays_d
    return ret_list + [ret_dict]


def render_test(
    H,
    W,
    K,
    chunk=1024 * 32,
    rays: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    c2w=None,
    ndc=True,
    near=0.0,
    far=1.0,
    use_viewdirs=False,
    c2w_staticcam=None,
    **kwargs,
):
    """Render rays.

    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
        camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      alphas_map:
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = run_nerf_helpers.get_rays(H, W, K, c2w)
    elif rays is not None:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = run_nerf_helpers.get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = run_nerf_helpers.ndc_rays(H, W, K[0][0], 1.0, rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(
        rays_d[..., :1]
    )
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_returned = batchify_rays_test(rays, chunk, **kwargs)
    for key in all_returned:
        k_sh = list(sh[:-1]) + list(all_returned[key].shape[1:])
        all_returned[key] = torch.reshape(all_returned[key], k_sh)

    key_extract = ["depth_net_rgb_map", "depth_net_disp_map"]
    ret_list = [all_returned[key] for key in key_extract]
    ret_dict = {
        key: all_returned[key] for key in all_returned if key not in key_extract
    }
    ret_dict["rays_o"] = rays_o
    ret_dict["rays_d"] = rays_d
    return ret_list + [ret_dict]


def render_path(
    render_poses,
    hwf,
    K,
    chunk,
    render_kwargs,
    step,
    wandb_log=False,
    plot_object=False,
    gt_imgs=None,
    savedir=None,
    render_factor=0,
):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    all_pts = []
    densities = []
    alphas = []
    weights = []
    psnr_info = None
    total_psnr = 0
    n_render_poses = render_poses.shape[0]
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        depth_net_rgb, depth_net_disp, depth_net_extras = render(
            H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs
        )
        rgbs.append(depth_net_rgb.cpu().numpy())
        disps.append(depth_net_disp.cpu().numpy())
        if i == 0:
            print(depth_net_rgb.shape, depth_net_disp.shape)

        if gt_imgs is not None and render_factor == 0:
            psnr = -10.0 * np.log10(
                np.mean(np.square(depth_net_rgb.cpu().numpy() - gt_imgs[i]))
            )
            psnr_info = f"{i:03d}.png, PSNR: {psnr}"
            mse = None
            if depth_net_extras["max_z_vals"] is not None:
                mse = F.mse_loss(
                    depth_net_extras["max_z_vals"], depth_net_extras["depth_net_z_vals"]
                )
                total_mse += mse
                psnr_info += f", MSE: {mse}"
            total_psnr += psnr
            print(psnr_info)

        if savedir is not None:
            rgb8 = run_nerf_helpers.to8b(rgbs[-1])
            filename = os.path.join(savedir, "{:03d}.png".format(i))
            imageio.imwrite(filename, rgb8)

            if psnr_info is not None:
                f = os.path.join(savedir, "psnr.txt")
                with open(f, "a") as file:
                    file.write(f"{psnr_info}\n")
                if i == n_render_poses - 1:
                    to_write = f"Avg of {n_render_poses} images\n:PSNR: {total_psnr/n_render_poses}\n"
                    if total_mse > 0:
                        to_write += f"MSE: {total_mse/n_render_poses}"
                    with open(f, "a") as file:
                        file.write(to_write)

            if plot_object:
                pts = depth_net_extras["depth_net_pts"]  # [H, W, N_samples, 3]
                density = depth_net_extras["depth_net_density"]
                indices = utils.get_dense_indices(
                    density, min_density=torch.mean(density)
                )
                dense_points = pts[indices]
                densities.append(density[indices])
                all_pts.append(dense_points)
        if wandb_log:
            pts = torch.flatten(
                depth_net_extras["depth_net_pts"], end_dim=1
            )  # [H*W, N_samples, 3]
            max_pts = torch.flatten(
                depth_net_extras["max_pts"], end_dim=1
            )  # [H*W, N_samples, 3]
            rays_o = depth_net_extras["rays_o"]  # [H*W, 3]
            rays_d = depth_net_extras["rays_d"]  # [H*W, 3]
            indices = random.sample(range(len(rays_o)), k=5)
            rays_fig, rays_ax = visualize.visualize_rays_pts(
                rays_o=rays_o[indices].cpu(),
                rays_d=rays_d[indices].cpu(),
                pts=pts[indices],
                c=[[(0.0, 0.0, 1.0)]],
                title="{:03d}.png, y_pred: blue, y: black".format(i),
            )
            visualize._plot_points(
                rays_ax,
                max_pts[indices],
                c=[[(0.0, 0.0, 0.0)]],
            )
            wandb.log(
                {
                    f"Ray plot {step}": wandb.Image(rays_fig),
                }
            )
            plt.close(rays_fig)
    if plot_object and savedir is not None:
        all_pts = torch.cat(all_pts)  # [n, 3]
        densities = torch.cat(densities)  # [n, 1]
        for k in [1e4, 2e4, 3e4, 5e4, 6e4]:
            points_to_plot = utils.get_random_points(all_pts, k=int(k))  # [k, 3]
            fig, _ = visualize.plot_points(points_to_plot.unsqueeze(0), s=10)
            pickle.dump(
                fig, open(os.path.join(savedir, f"excavator{k}.fig.pickle"), "wb")
            )

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps, total_psnr / n_render_poses


def render_path_test(
    render_poses,
    hwf,
    K,
    chunk,
    render_kwargs,
    step,
    wandb_log=False,
    plot_object=False,
    gt_imgs=None,
    savedir=None,
    render_factor=0,
):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    all_pts = []
    densities = []
    alphas = []
    weights = []
    psnr_info = None
    total_psnr = 0
    total_mse = 0
    n_render_poses = render_poses.shape[0]
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        depth_net_rgb, depth_net_disp, depth_net_extras = render_test(
            H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs
        )
        rgbs.append(depth_net_rgb.cpu().numpy())
        disps.append(depth_net_disp.cpu().numpy())
        if i == 0:
            print(depth_net_rgb.shape, depth_net_disp.shape)

        if gt_imgs is not None and render_factor == 0:
            psnr = -10.0 * np.log10(
                np.mean(np.square(depth_net_rgb.cpu().numpy() - gt_imgs[i]))
            )
            psnr_info = f"{i:03d}.png, PSNR: {psnr}"
            total_psnr += psnr
            print(psnr_info)

        if savedir is not None:
            rgb8 = run_nerf_helpers.to8b(rgbs[-1])
            filename = os.path.join(savedir, "{:03d}.png".format(i))
            imageio.imwrite(filename, rgb8)

            if psnr_info is not None:
                f = os.path.join(savedir, "psnr.txt")
                with open(f, "a") as file:
                    file.write(f"{psnr_info}\n")
                if i == n_render_poses - 1:
                    to_write = f"Avg of {n_render_poses} images\n:PSNR: {total_psnr/n_render_poses}\n"
                    with open(f, "a") as file:
                        file.write(to_write)

            if plot_object:
                pts = depth_net_extras["depth_net_pts"]  # [H, W, N_samples, 3]
                density = depth_net_extras["depth_net_density"]
                indices = utils.get_dense_indices(
                    density, min_density=torch.mean(density)
                )
                dense_points = pts[indices]
                densities.append(density[indices])
                all_pts.append(dense_points)
        if wandb_log:
            pts = torch.flatten(
                depth_net_extras["depth_net_pts"], end_dim=1
            )  # [H*W, N_samples, 3]
            max_pts = torch.flatten(
                depth_net_extras["max_pts"], end_dim=1
            )  # [H*W, N_samples, 3]
            rays_o = depth_net_extras["rays_o"]  # [H*W, 3]
            rays_d = depth_net_extras["rays_d"]  # [H*W, 3]
            indices = random.sample(range(len(rays_o)), k=5)
            rays_fig, rays_ax = visualize.visualize_rays_pts(
                rays_o=rays_o[indices].cpu(),
                rays_d=rays_d[indices].cpu(),
                pts=pts[indices],
                c=[[(0.0, 0.0, 1.0)]],
                title="{:03d}.png, y_pred: blue, y: black".format(i),
            )
            visualize._plot_points(
                rays_ax,
                max_pts[indices],
                c=[[(0.0, 0.0, 0.0)]],
            )
            wandb.log(
                {
                    f"Ray plot {step}": wandb.Image(rays_fig),
                }
            )
            plt.close(rays_fig)
    if plot_object and savedir is not None:
        all_pts = torch.cat(all_pts)  # [n, 3]
        densities = torch.cat(densities)  # [n, 1]
        for k in [1e4, 2e4, 3e4, 5e4, 6e4]:
            points_to_plot = utils.get_random_points(all_pts, k=int(k))  # [k, 3]
            fig, _ = visualize.plot_points(points_to_plot.unsqueeze(0), s=10)
            pickle.dump(
                fig, open(os.path.join(savedir, f"excavator{k}.fig.pickle"), "wb")
            )

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps, total_psnr / n_render_poses


def create_nerf(args, model):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = run_nerf_helpers.get_embedder(
        args.multires, args.i_embed, args.input_dims_embed
    )

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = run_nerf_helpers.get_embedder(
            args.multires_views, args.i_embed, args.input_dims_embed
        )
    # output_ch will equal to 4 either way if view_dirs are used
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model_nerf = model(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs,
    ).to(args.device)
    grad_vars = list(model_nerf.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = model(
            D=args.netdepth_fine,
            W=args.netwidth_fine,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
        ).to(args.device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: args.run_network(
        inputs,
        viewdirs,
        network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk,
    )

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        utils.load_nerf(model_nerf, model_fine, optimizer, ckpt)

    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb": args.perturb,
        "N_importance": args.N_importance,
        "network_fine": model_fine,
        "N_samples": args.N_samples,
        "network_fn": model_nerf,
        "use_viewdirs": args.use_viewdirs,
        "white_bkgd": args.white_bkgd,
        "raw_noise_std": args.raw_noise_std,
        "trainer": args,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != "llff" or args.no_ndc:
        print("Not ndc!")
        render_kwargs_train["ndc"] = False
        render_kwargs_train["lindisp"] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return (render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer)


def sample_as_in_NeRF(
    ray_batch,
    network_fn,
    network_fine,
    network_query_fn,
    N_samples,
    trainer,
    perturb,
    raw_noise_std,
    lindisp,
    white_bkgd,
    kwargs,
    pytest,
):
    """Samples points along rays as in Neural Radiance Fields (NeRF).

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.

    Returns:
      fine_density: torch.Tensor. Density values for all points (coarse + fine)
      fine_z_vals: torch.Tensor. Z values for all points (coarse + fine)
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
    (
        coarse_rgb_map,
        coarse_disp_map,
        coarse_acc_map,
        coarse_weights,
        coarse_depth_map,
        coarse_z_vals,
        coarse_weights,
        coarse_raw,
        coarse_alphas_map,
    ) = trainer.sample_coarse_points(
        near=near,
        far=far,
        perturb=perturb,
        N_rays=N_rays,
        N_samples=N_samples,
        viewdirs=viewdirs,
        network_fn=network_fn,
        network_query_fn=network_query_fn,
        rays_o=rays_o,
        rays_d=rays_d,
        raw_noise_std=raw_noise_std,
        white_bkgd=white_bkgd,
        pytest=pytest,
        lindisp=lindisp,
        kwargs=kwargs,
    )
    (
        rgb_map_0,
        disp_map_0,
        acc_map_0,
        fine_rgb_map,
        fine_disp_map,
        fine_acc_map,
        fine_raw,
        fine_z_vals,
        fine_pts,
        fine_density,
        fine_alphas,
        fine_weights,
    ) = trainer.sample_fine_points(
        z_vals=coarse_z_vals,
        weights=coarse_weights,
        perturb=perturb,
        pytest=pytest,
        rays_d=rays_d,
        rays_o=rays_o,
        rgb_map=coarse_rgb_map,
        disp_map=coarse_disp_map,
        acc_map=coarse_acc_map,
        network_fn=network_fn,
        network_fine=network_fine,
        network_query_fn=network_query_fn,
        viewdirs=viewdirs,
        raw_noise_std=raw_noise_std,
        white_bkgd=white_bkgd,
    )
    return fine_density, fine_z_vals, fine_pts, fine_rgb_map, fine_weights, fine_alphas


def render_rays(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    trainer,
    retraw=True,
    lindisp=False,
    perturb=0.0,
    N_importance=0,
    network_fine=None,
    white_bkgd=False,
    raw_noise_std=0.0,
    verbose=False,
    pytest=False,
    **kwargs,
) -> tuple[dict[str, torch.FloatTensor], torch.FloatTensor]:
    """Volumetric rendering.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    fine_density, fine_z_vals, fine_pts, fine_rgb_map, fine_weights, fine_alphas = (
        sample_as_in_NeRF(
            ray_batch=ray_batch,
            N_samples=N_samples,
            network_fn=network_fn,
            network_fine=network_fine,
            network_query_fn=network_query_fn,
            trainer=trainer,
            perturb=perturb,
            raw_noise_std=raw_noise_std,
            lindisp=lindisp,
            white_bkgd=white_bkgd,
            pytest=pytest,
            kwargs=kwargs,
        )
    )
    top_indices = fine_weights.argmax(dim=1, keepdim=True)
    max_z_vals = torch.gather(fine_z_vals, 1, top_indices)
    max_pts = rays_o[..., None, :] + rays_d[..., None, :] * max_z_vals[..., :, None]
    depth_net_z_vals = kwargs["depth_network"](rays_o, rays_d)
    depth_net_pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * depth_net_z_vals[..., :, None]
    )
    if network_fine is not None:
        depth_net_raw = network_query_fn(depth_net_pts, viewdirs, network_fine)
    else:
        depth_net_raw = network_query_fn(depth_net_pts, viewdirs, network_fn)
    (
        depth_net_rgb_map,
        depth_net_disp_map,
        depth_net_acc_map,
        depth_net_depth_map,
        depth_net_density,
        depth_net_alphas,
        depth_net_weights,
    ) = trainer.raw2outputs(
        raw=depth_net_raw,
        z_vals=depth_net_z_vals,
        rays_d=rays_d,
        raw_noise=raw_noise_std,
        white_bkdg=white_bkgd,
        pytest=pytest,
    )

    ret = {
        "depth_net_rgb_map": depth_net_rgb_map,
        "depth_net_disp_map": depth_net_disp_map,
        "depth_net_z_vals": depth_net_z_vals,
        "max_z_vals": max_z_vals,
        "depth_net_pts": depth_net_pts.cpu(),
        "max_pts": max_pts.cpu(),
    }

    if retraw:
        ret["raw"] = depth_net_raw.cpu()

    for key in ret:
        if (torch.isnan(ret[key]).any() or torch.isinf(ret[key]).any()) and DEBUG:
            print(f"! [Numerical Error] {key} contains nan or inf.")

    return ret


def render_rays_test(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    trainer,
    retraw=True,
    lindisp=False,
    perturb=0.0,
    N_importance=0,
    network_fine=None,
    white_bkgd=False,
    raw_noise_std=0.0,
    verbose=False,
    pytest=False,
    **kwargs,
) -> tuple[dict[str, torch.FloatTensor], torch.FloatTensor]:
    """Volumetric rendering.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    depth_net_z_vals = kwargs["depth_network"](rays_o, rays_d)
    depth_net_pts, depth_net_z_vals = sample_points_around_mean(
        rays_o=rays_o,
        rays_d=rays_d,
        mean=depth_net_z_vals,
        n_samples=trainer.n_depth_samples,
        mode=trainer.sampling_mode,
        std=trainer.distance,
    )
    if network_fine is not None:
        depth_net_raw = network_query_fn(depth_net_pts, viewdirs, network_fine)
    else:
        depth_net_raw = network_query_fn(depth_net_pts, viewdirs, network_fn)
    (
        depth_net_rgb_map,
        depth_net_disp_map,
        depth_net_acc_map,
        depth_net_depth_map,
        depth_net_density,
        depth_net_alphas,
        depth_net_weights,
    ) = trainer.raw2outputs(
        raw=depth_net_raw,
        z_vals=depth_net_z_vals,
        rays_d=rays_d,
        raw_noise=raw_noise_std,
        white_bkdg=white_bkgd,
        pytest=pytest,
    )

    ret = {
        "depth_net_rgb_map": depth_net_rgb_map,
        "depth_net_disp_map": depth_net_disp_map,
        "depth_net_z_vals": depth_net_z_vals,
        "depth_net_pts": depth_net_pts.cpu(),
    }

    if retraw:
        ret["raw"] = depth_net_raw.cpu()

    for key in ret:
        if (torch.isnan(ret[key]).any() or torch.isinf(ret[key]).any()) and DEBUG:
            print(f"! [Numerical Error] {key} contains nan or inf.")

    return ret


def config_parser():
    """Handle console arguments logic."""
    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config_path", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument(
        "--basedir", type=str, default="./logs/", help="where to store ckpts and logs"
    )
    parser.add_argument(
        "--datadir", type=str, default="./data/llff/fern", help="input data directory"
    )

    # training options
    parser.add_argument("--netdepth", type=int, default=8, help="layers in network")
    parser.add_argument("--netwidth", type=int, default=256, help="channels per layer")
    parser.add_argument(
        "--netdepth_fine", type=int, default=8, help="layers in fine network"
    )
    parser.add_argument(
        "--netwidth_fine",
        type=int,
        default=256,
        help="channels per layer in fine network",
    )
    parser.add_argument(
        "--N_rand",
        type=int,
        default=32 * 32 * 4,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument("--lrate", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=250,
        help="exponential learning rate decay (in 1000 steps)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1024 * 32,
        help="number of rays processed in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--netchunk",
        type=int,
        default=1024 * 64,
        help="number of pts sent through network in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--no_batching",
        action="store_true",
        help="only take random rays from 1 image at a time",
    )
    parser.add_argument(
        "--no_reload", action="store_true", help="do not reload weights from saved ckpt"
    )
    parser.add_argument(
        "--ft_path",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )
    parser.add_argument(
        "--input_dims_embed", type=int, default=None, help="input_dims in get_embedder"
    )

    # rendering options
    parser.add_argument(
        "--N_samples", type=int, default=64, help="number of coarse samples per ray"
    )
    parser.add_argument(
        "--N_importance",
        type=int,
        default=0,
        help="number of additional fine samples per ray",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    parser.add_argument(
        "--use_viewdirs", action="store_true", help="use full 5D input instead of 3D"
    )
    parser.add_argument(
        "--i_embed",
        type=int,
        default=0,
        help="set 0 for default positional encoding, -1 for none",
    )
    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )

    parser.add_argument(
        "--render_only",
        action="store_true",
        help="do not optimize, reload weights and render out render_poses path",
    )
    parser.add_argument(
        "--render_test",
        action="store_true",
        help="render the test set instead of render_poses path",
    )
    parser.add_argument(
        "--render_factor",
        type=int,
        default=0,
        help="downsampling factor to speed up rendering, set 4 or 8 for fast preview",
    )

    # training options
    parser.add_argument(
        "--precrop_iters",
        type=int,
        default=0,
        help="number of steps to train on central crops",
    )
    parser.add_argument(
        "--precrop_frac",
        type=float,
        default=0.5,
        help="fraction of img taken for central crops",
    )

    # dataset options
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="llff",
        help="options: llff / blender / deepvoxels",
    )
    parser.add_argument(
        "--testskip",
        type=int,
        default=8,
        help="will load 1/N images from test/val sets, useful for large datasets like deepvoxels",
    )

    ## deepvoxels flags
    parser.add_argument(
        "--shape",
        type=str,
        default="greek",
        help="options : armchair / cube / greek / vase",
    )

    ## blender flags
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help="set to render synthetic data on a white bkgd (always use for dvoxels)",
    )
    parser.add_argument(
        "--half_res",
        action="store_true",
        help="load blender synthetic data at 400x400 instead of 800x800",
    )

    ## llff flags
    parser.add_argument(
        "--factor", type=int, default=8, help="downsample factor for LLFF images"
    )
    parser.add_argument(
        "--no_ndc",
        action="store_true",
        help="do not use normalized device coordinates (set for non-forward facing scenes)",
    )
    parser.add_argument(
        "--lindisp",
        action="store_true",
        help="sampling linearly in disparity rather than depth",
    )
    parser.add_argument(
        "--spherify", action="store_true", help="set for spherical 360 scenes"
    )
    parser.add_argument(
        "--llffhold",
        type=int,
        default=8,
        help="will take every 1/N images as LLFF test set, paper uses 8",
    )

    # logging/saving options
    parser.add_argument(
        "--i_print",
        type=int,
        default=100,
        help="frequency of console printout and metric loggin",
    )
    parser.add_argument(
        "--i_img", type=int, default=500, help="frequency of tensorboard image logging"
    )
    parser.add_argument(
        "--i_weights", type=int, default=10000, help="frequency of weight ckpt saving"
    )
    parser.add_argument(
        "--i_testset", type=int, default=50000, help="frequency of testset saving"
    )
    parser.add_argument(
        "--i_video",
        type=int,
        default=50000,
        help="frequency of render_poses video saving",
    )

    return parser
