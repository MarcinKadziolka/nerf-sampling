from nerf_pytorch.trainers.Trainer import Trainer
from nerf_pytorch.load_LINEMOD import load_LINEMOD_data


class LinemodTrainer(Trainer):
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
            white_bkgd,
            datadir,
            **kwargs
    ):
        self.K = None
        self.testskip = testskip
        self.white_bkgd = white_bkgd
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
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(
            self.datadir,
            self.half_res,
            self.testskip
        )
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')

        self.near = near
        self.far = far
        self.K = K

        i_train, i_val, i_test = i_split

        if self.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

        return hwf, poses, i_test, i_val, i_train, images