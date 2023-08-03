"""Script for run sphere nerf."""
import click
import torch

from nerf_pytorch.run_nerf import train


@click.command()
@click.option(
    "--model",
    help="Name of selected model",
    type=str,
    default="nerf"
)
def main(
    model: str,
):
    if model == "nerf":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        train()


if __name__ == "__main__":
    main()
