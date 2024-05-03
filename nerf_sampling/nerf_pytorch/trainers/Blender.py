import torch

from nerf_sampling.nerf_pytorch.trainers.Trainer import Trainer
from nerf_sampling.nerf_pytorch.load_blender import load_blender_data


class BlenderTrainer(Trainer):
    def __init__(self, half_res, white_bkgd, testskip=8, near=2.0, far=6.0, **kwargs):

        self.half_res = half_res
        self.testskip = testskip
        self.white_bkgd = white_bkgd

        self.near = near
        self.far = far

        super().__init__(**kwargs)

    def load_data(self):
        images, poses, render_poses, hwf, i_split = load_blender_data(
            self.datadir, self.half_res, self.testskip
        )
        print("Loaded blender", images.shape, render_poses.shape, hwf, self.datadir)
        i_train, i_val, i_test = i_split

        if self.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]

        render_poses = render_poses.clone().detach()
        return hwf, poses, i_test, i_val, i_train, images, render_poses
