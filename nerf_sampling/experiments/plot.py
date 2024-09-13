import matplotlib.pyplot as plt
from nerf_sampling.nerf_pytorch import visualize
from nerf_sampling.nerf_pytorch import utils
import torch
from nerf_sampling.definitions import ROOT_DIR
import os

dir_path = f"{ROOT_DIR}/logs/lego_render/renderonly_test_099999"
scene_data_path = os.path.join(dir_path, "scene_data.pt")
scene_data = torch.load(scene_data_path)

all_pts = scene_data["all_pts"].cpu()
all_weights = scene_data["all_weights"].cpu()
k = 5e4
min_indices = utils.get_min_indices(all_weights, torch.tensor([1]))
points_to_plot = all_pts[min_indices]
points_to_plot = utils.get_random_points(points_to_plot, k=int(k))  # [k, 3]
fig, _ = visualize.plot_points(points_to_plot.unsqueeze(0), s=10)
plt.show()

fig_name = "3d_points.png"
fig.savefig(os.path.join(dir_path, fig_name), dpi=600, bbox_inches="tight")
