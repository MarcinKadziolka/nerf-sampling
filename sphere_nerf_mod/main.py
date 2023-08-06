import torch

from sphere_nerf_mod.line import Line
from sphere_nerf_mod.sphere import Sphere

spheres = [Sphere(torch.zeros((3, 1)), 0.1 * r) for r in range(1, 11)]


# returns absolute point coordinates, it's possible that we need
# to return the distance to the ray's origin point
def get_points_on_ray_for_all_spheres(ray: Line, spheres: [Sphere]):
    points = []
    for sphere in spheres:
        intersection_points = ray.find_intersection_points_with_sphere(sphere)
        if len(intersection_points) == 2:
            points.append(ray.select_closest_point_to_origin(intersection_points))
        elif len(intersection_points) == 1:
            points.append(intersection_points[0])
        else:
            points.append(torch.zeros((3, 1)))
    return points
