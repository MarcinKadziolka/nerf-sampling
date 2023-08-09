"""Utils module - contains useful low-level utility functions."""

import math


def solve_quadratic_equation(a: float, b: float, c: float) -> list[float]:
    """Solve quadratic equation ax^2 + bx + c = 0."""
    delta = b * b - 4 * a * c
    if delta > 0:
        sqrt_delta = math.sqrt(delta)
        return [(-b - sqrt_delta) / (2 * a), (-b + sqrt_delta) / (2 * a)]
    elif delta == 0:
        return [-b / (2 * a)]
    else:
        return []
