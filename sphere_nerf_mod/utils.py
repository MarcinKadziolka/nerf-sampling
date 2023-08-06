import math


def solve_quadratic_equation(a: float, b: float, c: float) -> [float]:
    delta = b*b - 4*a*c
    if delta > 0:
        sqrt_delta = math.sqrt(delta)
        return [(-b - sqrt_delta) / (2*a), (-b + sqrt_delta) / (2*a)]
    elif delta == 0:
        return [-b / (2*a)]
    else:
        return []
