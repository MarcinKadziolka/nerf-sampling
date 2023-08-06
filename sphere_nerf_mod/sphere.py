import torch


class Sphere:
    def __init__(self, center: torch.Tensor(1, 3), radius: float):
        self.center = center
        self.radius = radius
        