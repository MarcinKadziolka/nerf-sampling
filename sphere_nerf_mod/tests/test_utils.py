"""Tests for all utils module functions."""
import torch

from sphere_nerf_mod.utils import solve_quadratic_equation


def test_solve_quadratic_equation():
    """Test solving quadratic equation."""
    assert torch.equal(
        solve_quadratic_equation(
            torch.Tensor([1]),
            torch.Tensor([2]),
            torch.Tensor([1])
        ),
        torch.Tensor([[-1], [torch.nan]])
    )
    assert torch.equal(
        solve_quadratic_equation(
            torch.Tensor([1, 4, 5]),
            torch.Tensor([1, 4, 6]),
            torch.Tensor([1, 1, 1])
        ),
        torch.Tensor(
            [[torch.nan, -0.5, -1],
             [torch.nan, torch.nan, -0.2]]
        )
    )


if __name__ == "__main__":
    test_solve_quadratic_equation()