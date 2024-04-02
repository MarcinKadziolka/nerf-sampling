import numpy as np

from nerf_pytorch.trainers.Trainer import Trainer
from nerf_pytorch.load_deepvoxels import load_dv_data


class DeepvoxelsTrainer(Trainer):
    def __init__(
            self,
            dataset_type,
            render_test,
            basedir,
            expname,
            config_path,
            device,
            render_factor,
            chunk,
            N_rand,
            no_batching,
            testskip,
            datadir,
            **kwargs
    ):
        self.testskip = testskip
        self.near = None
        self.far = None

        super().__init__(
            dataset_type=dataset_type,
            render_test=render_test,
            basedir=basedir,
            expname=expname,
            config_path=config_path,
            device=device,
            render_factor=render_factor,
            chunk=chunk,
            N_rand=N_rand,
            no_batching=no_batching,
            datadir=datadir,
            **kwargs
        )

    def load_data(self):
        images, poses, render_poses, hwf, i_split = load_dv_data(
            scene=self.shape,
            basedir=self.datadir,
            testskip=self.testskip
        )

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, self.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.

        self.near = near
        self.far = far

        return hwf, poses, i_test, i_val, i_train, images