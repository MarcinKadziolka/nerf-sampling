from tqdm import trange

from nerf_pytorch.nerf_utils import *
from nerf_pytorch.utils import load_obj_from_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(trainer_config, dataset_type=None):
    if dataset_type == "llff":
        trainer_config["module"] = "nerf_pytorch.trainers.LLF"
    elif dataset_type == "blender":
        trainer_config["module"] = "nerf_pytorch.trainers.BlenderTrainer"
    elif dataset_type == "LINEMOD":
        trainer_config["module"] = "nerf_pytorch.trainers.LinemodTrainer"
    elif dataset_type == "deepvoxels":
        trainer_config["module"] = "nerf_pytorch.trainers.DeepvoxelsTrainer"
    else:
        if "module" not in trainer_config:
            raise "You have to declare module in trainer_config."
        Warning(
            f"You use your own dataset_type. "
            f'Trainer is declared by {trainer_config["module"]}'
        )

    trainer = load_obj_from_config(cfg=trainer_config)
    trainer.train()


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    parser = config_parser()
    args = parser.parse_args()
    trainer_config = {
        "kwargs": {
            "dataset_type": args.dataset_type,
            "render_test": args.render_test,
            "render_only": args.render_only,
            "basedir": args.basedir,
            "expname": args.expname,
            "config_path": args.config_path,
            "device": device,
            "render_factor": args.render_factor,
            "chunk": args.chunk,
            "N_rand": args.N_rand,
            "no_batching": args.no_batching,
            "half_res": args.half_res,
            "testskip": args.testskip,
            "white_bkgd": args.white_bkgd,
            "datadir": args.datadir,
            "multires": args.multires,
            "i_embed": args.i_embed,
            "multires_views": args.multires_views,
            "netchunk": args.netchunk,
            "lrate": args.lrate,
            "input_dims_embed": args.input_dims_embed,
            "lrate_decay": args.lrate_decay,
            "use_viewdirs": args.use_viewdirs,
            "N_importance": args.N_importance,
            "netdepth": args.netdepth,
            "netwidth": args.netwidth,
            "netdepth_fine": args.netdepth_fine,
            "netwidth_fine": args.netwidth_fine,
            "ft_path": args.ft_path,
            "perturb": args.perturb,
            "raw_noise_std": args.raw_noise_std,
            "N_samples": args.N_samples,
            "lindisp": args.lindisp,
            "precrop_iters": args.precrop_iters,
            "precrop_frac": args.precrop_frac,
            "i_weights": args.i_weights,
            "i_testset": args.i_testset,
            "i_video": args.i_video,
            "i_print": args.i_print,
        }
    }

    train(trainer_config, args.dataset_type)
