"""Main module.

Contains high-level functions, which implement SphereNeRF's behavior.
"""

import torch

from sphere_nerf_mod.line import Line
from sphere_nerf_mod.sphere import Sphere


# returns absolute point coordinates, it's possible that we need
# to return the distance to the ray's origin point
def get_points_on_ray_for_all_spheres(
        ray: Line, spheres: list[Sphere]) -> list[torch.Tensor(1, 3)]:
    """Finds intersection points of the ray with all spheres."""
    points = []
    for sphere in spheres:
        intersection_points = ray.find_intersection_points_with_sphere(sphere)
        if len(intersection_points) == 2:
            points.append(
                ray.select_closest_point_to_origin(intersection_points))
        elif len(intersection_points) == 1:
            points.append(intersection_points[0])
        else:
            points.append(torch.zeros((1, 3)))
    return points
