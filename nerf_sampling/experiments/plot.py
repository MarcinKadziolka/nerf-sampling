import matplotlib.pyplot as plt
from nerf_sampling.nerf_pytorch import visualize
from nerf_sampling.nerf_pytorch import utils
import torch
from nerf_sampling.definitions import ROOT_DIR
import os

dataset_name = "chair"
# dir_path = f"{ROOT_DIR}/logs/{dataset_name}/{dataset_name}_depth_net_render_n_samples_2_distance_0.01_sampling_mode_uniform/renderonly_test_199999"
dir_path = f"{ROOT_DIR}/logs/{dataset_name}/{dataset_name}_nerf_max_render/renderonly_test_199999"
scene_data_path = os.path.join(dir_path, "scene_data.pt")
scene_data = torch.load(scene_data_path)

all_pts = scene_data["all_pts"].cpu()
all_weights = scene_data["all_weights"].cpu()
k = 5e4
min_indices = utils.get_min_indices(all_weights, torch.tensor([0.0]))
points_to_plot = all_pts
points_to_plot = all_pts[min_indices]
random_indices = utils.get_random_indices(points_to_plot, k=int(k))  # [k, 3]
points_to_plot = points_to_plot[random_indices]
fig, _ = visualize.plot_points(points_to_plot.unsqueeze(0), s=10)
plt.show()

# fig_name = f"3d_points_{k}.png"
# fig.savefig(os.path.join(dir_path, fig_name), dpi=600, bbox_inches="tight")
