"""Tests for all utils module functions."""

from sphere_nerf_mod.utils import solve_quadratic_equation


def test_solve_quadratic_equation():
    """Test solving quadratic equation."""
    assert solve_quadratic_equation(1, 2, 1) == [-1]
    assert solve_quadratic_equation(1, 1, 1) == []
    assert solve_quadratic_equation(1, 7, 6) == [-6, -1]
