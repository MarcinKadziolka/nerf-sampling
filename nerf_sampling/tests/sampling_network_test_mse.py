from nerf_sampling.samplers.baseline_sampler import BaselineSampler
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from nerf_sampling.nerf_pytorch import visualize
import click


@click.command()
@click.option("--plot", default=False, is_flag=True, help="Plot every 10 iters.")
@click.option("--no_final_plot", default=True, is_flag=True, help="Don't plot results.")
def train(**kwargs):
    """Mock train function to test sampler network."""
    plot = kwargs["plot"]
    final_plot = kwargs["no_final_plot"]

    target_z_vals = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])
    n_samples = 1
    sampling_network = BaselineSampler(n_main_samples=n_samples)
    rays_o = torch.zeros(len(target_z_vals), 3, dtype=torch.float)
    for i, ray in enumerate(rays_o):
        ray[2] += i
    rays_d = torch.tensor([[1, 0, 0]], dtype=torch.float).repeat(len(target_z_vals), 1)
    optim = torch.optim.Adam(sampling_network.parameters())

    for i in range(100):
        pts, z_vals = sampling_network(rays_o, rays_d)
        expanded_target_z_vals = target_z_vals.unsqueeze(1).expand(-1, z_vals.shape[1])
        loss = F.mse_loss(expanded_target_z_vals, z_vals)

        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % 10 == 0:
            print(
                f"Step {i}, Loss: {loss.item()}, mean pts = {torch.mean(pts, dim=1)}, target = {target_z_vals}"
            )
            if plot:
                for pt in pts:
                    visualize.visualize_rays_pts(rays_o, rays_d, pts)

    if final_plot:
        pts, _ = sampling_network(rays_o, rays_d)
        visualize.visualize_rays_pts(
            rays_o, rays_d, pts.cpu().detach(), title=f"{target_z_vals=}"
        )
        plt.show()


if __name__ == "__main__":
    train()
