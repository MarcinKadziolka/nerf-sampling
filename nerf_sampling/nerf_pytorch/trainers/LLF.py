import numpy as np

from nerf_sampling.nerf_pytorch.trainers.Trainer import Trainer
from nerf_sampling.nerf_pytorch.load_llff import load_llff_data


class LLFTrainer(Trainer):

    def __init__(
        self,
        dataset_type,
        llffhold,
        no_ndc,
        spherify,
        render_test,
        basedir,
        expname,
        config_path,
        device,
        render_factor,
        chunk,
        N_rand,
        no_batching,
        datadir,
    ):
        self.far = None
        self.near = None
        self.llffhold = llffhold
        self.no_ndc = no_ndc
        self.spherify = spherify

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
        )

    def load_data(self):
        images, poses, bds, render_poses, i_test = load_llff_data(
            self.datadir,
            self.factor,
            recenter=True,
            bd_factor=0.75,
            spherify=self.spherify,
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print("Loaded llff", images.shape, render_poses.shape, hwf, self.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if self.llffhold > 0:
            print("Auto LLFF holdout,", self.llffhold)
            i_test = np.arange(images.shape[0])[:: self.llffhold]

        i_val = i_test
        i_train = np.array(
            [
                i
                for i in np.arange(int(images.shape[0]))
                if (i not in i_test and i not in i_val)
            ]
        )

        print("DEFINING BOUNDS")
        if self.no_ndc:
            near = np.ndarray.min(bds) * 0.9
            far = np.ndarray.max(bds) * 1.0

        else:
            near = 0.0
            far = 1.0
        print("NEAR FAR", near, far)

        self.near = near
        self.far = far

        return hwf, poses, i_test, i_val, i_train, images
