import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from nerf_sampling.samplers.baseline_sampler import BaselineSampler
from nerf_sampling.nerf_pytorch import visualize
import click

# Define constants for image dimensions
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 400


@click.command()
@click.option("--plot", default=False, is_flag=True, help="Plot every 10 iters.")
@click.option("--no_final_plot", default=True, is_flag=True, help="Don't plot results.")
def train(**kwargs):
    """Train function to test sampler network with a camera and rays."""
    torch.manual_seed(42)
    plot = kwargs["plot"]
    final_plot = kwargs["no_final_plot"]

    # Define camera origin (assuming a pinhole camera model)
    camera_origin = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)

    # Create a grid of rays
    i_coords, j_coords = torch.meshgrid(
        torch.linspace(-1, 1, IMAGE_WIDTH), torch.linspace(-1, 1, IMAGE_HEIGHT)
    )
    rays_d = torch.stack(
        [i_coords.flatten(), j_coords.flatten(), torch.ones_like(i_coords.flatten())],
        dim=-1,
    )
    rays_d = F.normalize(rays_d, dim=-1)
    rays_o = camera_origin.expand_as(rays_d)

    # Initialize z-values with random values between 2.0 and 6.0
    z_vals = 2.0 + 4.0 * torch.rand(IMAGE_WIDTH * IMAGE_HEIGHT)

    # Initialize the sampling network
    n_main_samples = 1
    n_noise_samples = 0
    sampling_network = BaselineSampler(
        n_main_samples=n_main_samples, n_noise_samples=n_noise_samples
    )

    # Optimizer
    optim = torch.optim.Adam(sampling_network.parameters())

    for step in range(100000):
        (main_pts, main_z_vals), (noise_pts, noise_z_vals) = sampling_network(
            rays_o, rays_d
        )
        expanded_target_z_vals = z_vals.unsqueeze(1).expand(-1, main_z_vals.shape[1])
        loss = F.mse_loss(expanded_target_z_vals, main_z_vals)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
            if plot:
                visualize.visualize_rays_pts(rays_o, rays_d, main_pts)

    if final_plot:
        main_pts, _ = sampling_network(rays_o, rays_d)
        visualize.visualize_rays_pts(
            rays_o,
            rays_d,
            main_pts.cpu().detach(),
            title=f"Final visualization {z_vals=}",
        )
        plt.show()


if __name__ == "__main__":
    train()
