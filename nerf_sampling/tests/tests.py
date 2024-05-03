import os

import pytest
import torch

from nerf_sampling.nerf_pytorch import utils
from nerf_sampling.nerf_pytorch.run_nerf_helpers import NeRF
from nerf_sampling.samplers import baseline_sampler


def test_freeze_unfreeze_model():
    model = NeRF()
    for param in model.parameters():
        assert param.requires_grad == True

    utils.freeze_model(model)
    for param in model.parameters():
        assert param.requires_grad == False

    utils.unfreeze_model(model)
    for param in model.parameters():
        assert param.requires_grad == True


def test_save_and_load():
    network_fn = NeRF()
    network_fine = None
    optimizer = torch.optim.Adam(params=list(network_fn.parameters()))
    sampling_network = baseline_sampler.BaselineSampler()
    sampling_optimizer = torch.optim.Adam(params=list(sampling_network.parameters()))
    global_step = 2000
    path = "./data.tar"
    utils.save_state(
        global_step=global_step,
        network_fn=network_fn,
        network_fine=network_fine,
        optimizer=optimizer,
        sampling_network=sampling_network,
        sampling_optimizer=sampling_optimizer,
        path=path,
    )
    assert os.path.exists(path)
    ckpt = torch.load(path)
    os.remove(path)

    start = ckpt["global_step"]
    load_network_fn = NeRF()
    load_network_fine = None
    load_optimizer = torch.optim.Adam(params=list(network_fn.parameters()))
    load_sampling_network = baseline_sampler.BaselineSampler()
    load_sampling_optimizer = torch.optim.Adam(
        params=list(sampling_network.parameters())
    )
    assert start == global_step
    utils.load_nerf(load_network_fn, load_network_fine, load_optimizer, ckpt)
    utils.load_sampling_network(load_sampling_network, load_sampling_optimizer, ckpt)

    for p1, p2 in zip(network_fn.parameters(), load_network_fn.parameters()):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(
        sampling_network.parameters(), load_sampling_network.parameters()
    ):
        assert torch.equal(p1, p2)

    for param_group1, param_group2 in zip(
        optimizer.param_groups, load_optimizer.param_groups
    ):
        assert param_group1["lr"] == param_group2["lr"]
        assert torch.allclose(param_group1["params"][0], param_group2["params"][0])

    for param_group1, param_group2 in zip(
        sampling_optimizer.param_groups, load_sampling_optimizer.param_groups
    ):
        assert param_group1["lr"] == param_group2["lr"]
        assert torch.allclose(param_group1["params"][0], param_group2["params"][0])


@pytest.fixture
def test_config():
    return {
        "N_samples": 64,
        "density_in_loss": True,
        "train_sampler_frequency": 10,
    }


def test_update_config_good(test_config):
    n_samples = test_config["N_samples"] - 32
    train_sampler_frequency = test_config["train_sampler_frequency"] + 90
    update = {
        "N_samples": n_samples,
        "train_sampler_frequency": train_sampler_frequency,
    }
    utils.override_config(test_config, update)
    assert test_config["N_samples"] == n_samples
    assert test_config["train_sampler_frequency"] == train_sampler_frequency


def test_update_config_key_does_not_exists(test_config):
    invalid_key = "N_sampels"
    invalid_update = {
        invalid_key: 32,
        "train_sampler_frequency": 2,
    }
    with pytest.raises(KeyError) as exc_info:
        utils.override_config(test_config, invalid_update)

    # Optionally, assert the message of the error
    assert f"Key {invalid_key} does not exist in config" in str(exc_info.value)
