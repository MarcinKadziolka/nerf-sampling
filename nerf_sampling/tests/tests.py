import os

import pytest
import torch

from nerf_sampling.nerf_pytorch import utils
from nerf_sampling.nerf_pytorch.run_nerf_helpers import NeRF
from nerf_sampling.samplers import baseline_sampler
from nerf_sampling.nerf_pytorch.utils import (
    find_intersection_points_with_sphere,
    solve_quadratic_equation,
)


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


class TestBaselineSampler:
    def test_baseline_sampler_layers_and_depth(self):
        hidden_sizes = [16, 32, 64]
        cat_hidden_sizes = [32, 64, 128]
        n_samples = 8
        multires = 5
        n_channels = 3
        # multires * cos/sin * 3 channels + 3 original channels
        expected_embedding_dim = multires * 2 * n_channels + n_channels
        intersection_channels = 6
        intersection_embedding = (
            multires * 2 * intersection_channels + intersection_channels
        )
        sampler = baseline_sampler.BaselineSampler(
            hidden_sizes=hidden_sizes,
            cat_hidden_sizes=cat_hidden_sizes,
            multires=multires,
            origin_channels=n_channels,
            direction_channels=n_channels,
        )

        assert isinstance(sampler.sigmoid, torch.nn.Sigmoid)

        # Check depth of network
        assert len(sampler.origin_layers) == len(hidden_sizes)
        assert len(sampler.origin_layers) == len(hidden_sizes)
        # multiply by 2 to account for ReLU after each layer
        assert len(sampler.cat_layers) == len(cat_hidden_sizes) * 2

        # Check width of network
        # origin layers
        assert (
            sampler.origin_layers[0].in_features
            == sampler.origin_dims + sampler.origin_dims
        )
        assert sampler.origin_layers[0].out_features == hidden_sizes[0]

        assert (
            sampler.origin_layers[1].in_features
            == hidden_sizes[0] + sampler.origin_dims
        )
        assert sampler.origin_layers[1].out_features == hidden_sizes[1]

        assert (
            sampler.origin_layers[2].in_features
            == hidden_sizes[1] + sampler.origin_dims
        )
        assert sampler.origin_layers[2].out_features == hidden_sizes[2]

        # concatenated_layers
        assert sampler.cat_layers[0].in_features == hidden_sizes[-1] * 3 + (
            expected_embedding_dim + expected_embedding_dim + intersection_embedding
        )  # * 2 because we concatenate origin and direction
        assert sampler.cat_layers[0].out_features == cat_hidden_sizes[0]

        assert sampler.cat_layers[2].in_features == cat_hidden_sizes[0]
        assert sampler.cat_layers[2].out_features == cat_hidden_sizes[1]

        assert sampler.cat_layers[4].in_features == cat_hidden_sizes[1]
        assert sampler.cat_layers[4].out_features == cat_hidden_sizes[2]

        assert sampler.to_mean.in_features == cat_hidden_sizes[-1]
        assert sampler.to_mean.out_features == 1

    def test_baseline_sampler_one_layer(self):
        hidden_sizes = [16]
        concatenated_hidden_sizes = [32]
        sampler = baseline_sampler.BaselineSampler(
            hidden_sizes=hidden_sizes,
            cat_hidden_sizes=concatenated_hidden_sizes,
        )
        assert len(sampler.origin_layers) == len(hidden_sizes)
        assert len(sampler.origin_layers) == len(hidden_sizes)
        assert len(sampler.cat_layers) == len(concatenated_hidden_sizes) * 2

    def test_baseline_sampler_output_shape(self):
        n_rays = 4
        n_samples = 5
        rays_o = rays_d = torch.zeros(n_rays, 3)
        sampler = baseline_sampler.BaselineSampler(n_samples=n_samples)
        (pts, z_vals), mean = sampler(rays_o, rays_d)
        assert pts.shape == (n_rays, n_samples, 3)


def test_solve_quadratic_equation():
    """Test solving quadratic equation."""
    assert torch.isclose(
        solve_quadratic_equation(
            torch.Tensor([1]), torch.Tensor([2]), torch.Tensor([1])
        ),
        torch.Tensor([[-1], [-1]]),
        equal_nan=True,
    ).all()
    assert torch.isclose(
        solve_quadratic_equation(
            torch.Tensor([[1, 4, 5], [1, 4, 5]]),
            torch.Tensor([[1, 4, 6], [1, 4, 6]]),
            torch.Tensor([[1, 1, 1], [1, 1, 1]]),
        ),
        torch.Tensor(
            [
                [[torch.nan, -0.5, -1], [torch.nan, -0.5, -1]],
                [[torch.nan, -0.5, -0.2], [torch.nan, -0.5, -0.2]],
            ]
        ),
        equal_nan=True,
    ).all()
    assert torch.isclose(
        solve_quadratic_equation(
            torch.Tensor([1, 4, 5, 1, 4, 5]),
            torch.Tensor([1, 4, 6, 1, 4, 6]),
            torch.Tensor([1, 1, 1, 1, 1, 1]),
        ),
        torch.Tensor(
            [
                [torch.nan, -0.5, -1, torch.nan, -0.5, -1],
                [torch.nan, -0.5, -0.2, torch.nan, -0.5, -0.2],
            ]
        ),
        equal_nan=True,
    ).all()


def test_find_intersection_points_with_sphere_output_shape():
    rays_o = rays_d = torch.zeros(4, 3)
    sphere_radius = torch.tensor([2])
    t, intersection_points = find_intersection_points_with_sphere(
        rays_o, rays_d, sphere_radius
    )
    assert intersection_points.shape == (rays_o.shape[0], 2, 3)


def nan_equal(a, b):
    return torch.allclose(a[~torch.isnan(a)], b[~torch.isnan(b)], equal_nan=True)


def test_intersection_ray_directed_towards_sphere():
    rays_o = torch.tensor([[-3.0, 0.0, 0.0]])
    rays_d = torch.tensor([[1.0, 0.0, 0.0]])
    sphere_radius = torch.tensor([1.0])
    t, intersection_points = find_intersection_points_with_sphere(
        rays_o, rays_d, sphere_radius
    )
    expected = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert nan_equal(intersection_points[0], expected)


def test_no_intersection_ray_parallel_to_sphere():
    rays_o = torch.tensor([[-3.0, 0.0, 0.0]])
    rays_d = torch.tensor([[0.0, 2.0, 0.0]])
    sphere_radius = torch.tensor([1.0])
    t, intersection_points = find_intersection_points_with_sphere(
        rays_o, rays_d, sphere_radius
    )
    expected = torch.tensor(
        [[torch.nan, torch.nan, torch.nan], [torch.nan, torch.nan, torch.nan]]
    )
    assert nan_equal(intersection_points[0], expected)


def test_intersection_ray_directed_away_from_sphere():
    rays_o = torch.tensor([[-3.0, 0.0, 0.0]])
    rays_d = torch.tensor([[-1.0, 0.0, 0.0]])
    sphere_radius = torch.tensor([1.0])
    t, intersection_points = find_intersection_points_with_sphere(
        rays_o, rays_d, sphere_radius
    )
    expected = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    assert nan_equal(intersection_points[0], expected)


def test_tangential_ray_intersects_at_one_point():
    rays_o = torch.tensor([[-3.0, 1.0, 0.0]])
    rays_d = torch.tensor([[1.0, 0.0, 0.0]])
    sphere_radius = torch.tensor([1.0])
    t, intersection_points = find_intersection_points_with_sphere(
        rays_o, rays_d, sphere_radius
    )
    expected = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert nan_equal(intersection_points[0], expected)


def test_origin_on_sphere():
    rays_o = torch.tensor([[1.0, 0.0, 0.0]])  # On the sphere surface
    rays_d = torch.tensor([[0.0, 1.0, 0.0]])
    sphere_radius = torch.tensor([1.0])
    t, intersection_points = find_intersection_points_with_sphere(
        rays_o, rays_d, sphere_radius
    )
    expected = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )  # Should return the same point twice
    assert nan_equal(intersection_points[0], expected)


def test_origin_inside_sphere():
    rays_o = torch.tensor([[0.0, 0.0, 0.0]])
    rays_d = torch.tensor([[-1.0, 0.0, 0.0]])
    sphere_radius = torch.tensor([1.0])
    t, intersection_points = find_intersection_points_with_sphere(
        rays_o, rays_d, sphere_radius
    )
    expected = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    )  # Origin inside, two points
    assert nan_equal(intersection_points[0], expected)


def test_origin_on_sphere_moving_inward():
    rays_o = torch.tensor([[1.0, 0.0, 0.0]])
    rays_d = torch.tensor([[-1.0, 0.0, 0.0]])
    sphere_radius = torch.tensor([1.0])
    t, intersection_points = find_intersection_points_with_sphere(
        rays_o, rays_d, sphere_radius
    )
    expected = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    )  # Should intersect at the opposite point
    assert nan_equal(intersection_points[0], expected)


# some hit
# some don't
# expected points
# expected nans
# in the real life there come [n_rays, 3]
# sphere center is 0, 0, 0
# sphere radius is 2
# probably needs some approximation
# what happens if everything is correct?
# what should be the shape of the output?
# what happens in origin is inside the sphere?
# what happens if line is tangent to the sphere?
# what happens if line misses the sphere completly?
# what happens if origin begins on the sphere?
# what happens if origin begins on the end of the sphere?
test_origin_on_sphere()
