"""Tests for all utils module functions."""
import torch

from sphere_nerf_mod.utils import solve_quadratic_equation


def test_solve_quadratic_equation():
    """Test solving quadratic equation."""
    assert torch.isclose(
        solve_quadratic_equation(
            torch.Tensor([1]),
            torch.Tensor([2]),
            torch.Tensor([1])
        ),
        torch.Tensor([[-1], [-1]]),
        equal_nan=True
    ).all()
    assert torch.isclose(
        solve_quadratic_equation(
            torch.Tensor([[1, 4, 5],
                          [1, 4, 5]]),
            torch.Tensor([[1, 4, 6],
                          [1, 4, 6]]),
            torch.Tensor([[1, 1, 1],
                          [1, 1, 1]])
        ),
        torch.Tensor(
            [
                [
                    [torch.nan, -0.5, -1],
                    [torch.nan, -0.5, -0.2]
                ],
                [
                    [torch.nan, -0.5, -1],
                    [torch.nan, -0.5, -1]
                ],
                [
                    [torch.nan, -0.5, -0.2],
                    [torch.nan, -0.5, -0.2]
                ]
            ]
        ),
        equal_nan=True
    ).all()
    assert torch.isclose(
        solve_quadratic_equation(
            torch.Tensor([1, 4, 5, 1, 4, 5]),
            torch.Tensor([1, 4, 6, 1, 4, 6]),
            torch.Tensor([1, 1, 1, 1, 1, 1])
        ),
        torch.Tensor(
            [
                    [torch.nan, -0.5, -1, torch.nan, -0.5, -1],
                    [torch.nan, -0.5, -0.2, torch.nan, -0.5, -0.2]
            ]
        ),
        equal_nan=True
    ).all()


if __name__ == "__main__":
    test_solve_quadratic_equation()
