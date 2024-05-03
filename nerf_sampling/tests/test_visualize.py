from nerf_sampling.nerf_pytorch.visualize import *


def main(args):
    """Run example visualization of rays and points."""
    test_wandb = args.wandb
    save = args.save
    rays_o = torch.zeros((6, 3))
    rays_d = torch.Tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    )
    pts = torch.Tensor(
        [
            [[1, 0, 0]],
            [[0, 2, 0]],
            [[0, 0, 3]],
            [[-4, 0, 0]],
            [[0, -5, 0]],
            [[0, 0, -6]],
        ]
    )
    density = torch.Tensor([[-100], [0], [5], [10], [15], [100]])
    points_fig, _ = plot_points(pts, c=density)
    rays_fig = visualize_rays_pts(rays_o, rays_d, pts, n_rays=6, c=density)
    densities = torch.Tensor([10, 20, 30, 40, 50, 60, 70, 80])
    histogram_fig = plot_histogram(densities=densities)
    if save:
        pickle.dump(rays_fig, open("rays_fig.fig.pickle", "wb"))
        pickle.dump(points_fig, open("points_fig.fig.pickle", "wb"))
    elif test_wandb:
        wandb.init(project="nerf-sampling")
        wandb.log(
            {
                "Test ray plot": wandb.Image(rays_fig),
                "Test histogram": wandb.Image(histogram_fig),
            }
        )
    else:
        plt.show()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    main(args)
