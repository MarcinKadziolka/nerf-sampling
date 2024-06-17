from nerf_sampling.samplers.baseline_sampler import BaselineSampler
import matplotlib.pyplot as plt
import torch
from nerf_sampling.nerf_pytorch import visualize, loss_functions
import click


def calculate_densities(pts, target_center, gaussian_width):
    """Differentiable density distribution around targets."""
    densities = torch.exp(-0.5 * ((pts[:, 0] - target_center) ** 2) / gaussian_width**2)
    return densities


@click.command()
@click.option("--plot", default=False, is_flag=True, help="Plot every 10 iters.")
@click.option("--no_final_plot", default=True, is_flag=True, help="Don't plot results.")
def train(**kwargs):
    """Mock train function to test sampler network."""
    plot = kwargs["plot"]
    final_plot = kwargs["no_final_plot"]

    target_centers = [2, 3, 4, 5, 6]
    n_samples = 64
    loss_fn = loss_functions.mean_density_loss
    gaussian_width = 0.5
    sampling_network = BaselineSampler(n_samples=n_samples)
    rays_o = torch.zeros(len(target_centers), 3, dtype=torch.float)
    for i, ray in enumerate(rays_o):
        ray[2] += i
    rays_d = torch.tensor([[1, 0, 0]], dtype=torch.float).repeat(len(target_centers), 1)
    optim = torch.optim.Adam(sampling_network.parameters())

    for i in range(500):
        pts, _ = sampling_network(rays_o, rays_d)
        loss = 0
        for pt, target_center in zip(pts, target_centers):
            densities = calculate_densities(pt, target_center, gaussian_width)
            loss += loss_fn(densities.unsqueeze(0))

        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % 10 == 0:
            print(
                f"Step {i}, Loss: {loss.item()}, mean pts = {torch.mean(pts, dim=1)}, target = {target_centers}"
            )
            if plot:
                for pt in pts:
                    visualize.visualize_rays_pts(rays_o, rays_d, pts)
                visualize.plot_histogram(densities)

    if final_plot:
        pts, _ = sampling_network(rays_o, rays_d)
        for pt, target_center in zip(pts, target_centers):
            densities = calculate_densities(pt, target_center, gaussian_width)
        visualize.visualize_rays_pts(
            rays_o, rays_d, pts.cpu().detach(), title=f"{target_centers=}"
        )
        visualize.plot_histogram(densities.cpu().detach())
        plt.show()


if __name__ == "__main__":
    train()
