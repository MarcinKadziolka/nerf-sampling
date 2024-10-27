import os

import pytest
import torch

from nerf_sampling.nerf_pytorch import utils
from nerf_sampling.nerf_pytorch.run_nerf_helpers import NeRF
from nerf_sampling.depth_nets import depth_net
from nerf_sampling.nerf_pytorch.utils import (
    find_intersection_points_with_sphere,
    sample_gaussian,
    sample_uniform,
    solve_quadratic_equation,
    z_vals_to_points,
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
    depth_network = depth_net.DepthNet()
    sampling_optimizer = torch.optim.Adam(params=list(depth_network.parameters()))
    global_step = 2000
    path = "./data.tar"
    utils.save_state(
        global_step=global_step,
        network_fn=network_fn,
        network_fine=network_fine,
        optimizer=optimizer,
        depth_network=depth_network,
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
    load_depth_network = depth_net.DepthNet()
    load_sampling_optimizer = torch.optim.Adam(
        params=list(load_depth_network.parameters())
    )
    assert start == global_step
    utils.load_nerf(load_network_fn, load_network_fine, load_optimizer, ckpt)
    utils.load_depth_network(load_depth_network, load_sampling_optimizer, ckpt)

    for p1, p2 in zip(network_fn.parameters(), load_network_fn.parameters()):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(depth_network.parameters(), load_depth_network.parameters()):
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
        "train_depth_net_frequency": 10,
    }


def test_update_config_good(test_config):
    n_samples = test_config["N_samples"] - 32
    train_depth_net_frequency = test_config["train_depth_net_frequency"] + 90
    update = {
        "N_samples": n_samples,
        "train_depth_net_frequency": train_depth_net_frequency,
    }
    utils.override_config(test_config, update)
    assert test_config["N_samples"] == n_samples
    assert test_config["train_depth_net_frequency"] == train_depth_net_frequency


def test_update_config_key_does_not_exists(test_config):
    invalid_key = "N_sampels"
    invalid_update = {
        invalid_key: 32,
        "train_depth_net_frequency": 2,
    }
    with pytest.raises(KeyError) as exc_info:
        utils.override_config(test_config, invalid_update)

    # Optionally, assert the message of the error
    assert f"Key {invalid_key} does not exist in config" in str(exc_info.value)


class TestBaselineSampler:
    def test_depth_network_layers_and_depth(self):
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
        depth_network = depth_net.DepthNet(
            hidden_sizes=hidden_sizes,
            cat_hidden_sizes=cat_hidden_sizes,
            multires=multires,
            origin_channels=n_channels,
            direction_channels=n_channels,
        )

        # Check depth of network
        assert len(depth_network.origin_layers) == len(hidden_sizes)
        assert len(depth_network.origin_layers) == len(hidden_sizes)
        # multiply by 2 to account for ReLU after each layer
        assert len(depth_network.cat_layers) == len(cat_hidden_sizes) * 2

        # Check width of network
        # origin layers
        assert (
            depth_network.origin_layers[0].in_features
            == depth_network.origin_dims + depth_network.origin_dims
        )
        assert depth_network.origin_layers[0].out_features == hidden_sizes[0]

        assert (
            depth_network.origin_layers[1].in_features
            == hidden_sizes[0] + depth_network.origin_dims
        )
        assert depth_network.origin_layers[1].out_features == hidden_sizes[1]

        assert (
            depth_network.origin_layers[2].in_features
            == hidden_sizes[1] + depth_network.origin_dims
        )
        assert depth_network.origin_layers[2].out_features == hidden_sizes[2]

        # concatenated_layers
        assert depth_network.cat_layers[0].in_features == hidden_sizes[-1] * 3 + (
            expected_embedding_dim + expected_embedding_dim + intersection_embedding
        )  # * 2 because we concatenate origin and direction
        assert depth_network.cat_layers[0].out_features == cat_hidden_sizes[0]

        assert depth_network.cat_layers[2].in_features == cat_hidden_sizes[0]
        assert depth_network.cat_layers[2].out_features == cat_hidden_sizes[1]

        assert depth_network.cat_layers[4].in_features == cat_hidden_sizes[1]
        assert depth_network.cat_layers[4].out_features == cat_hidden_sizes[2]

        assert depth_network.to_depth[0].in_features == cat_hidden_sizes[-1]
        assert depth_network.to_depth[0].out_features == 1
        assert isinstance(depth_network.to_depth[1], torch.nn.Sigmoid)

    def test_depth_network_one_layer(self):
        hidden_sizes = [16]
        concatenated_hidden_sizes = [32]
        depth_network = depth_net.DepthNet(
            hidden_sizes=hidden_sizes,
            cat_hidden_sizes=concatenated_hidden_sizes,
        )
        assert len(depth_network.origin_layers) == len(hidden_sizes)
        assert len(depth_network.origin_layers) == len(hidden_sizes)
        assert len(depth_network.cat_layers) == len(concatenated_hidden_sizes) * 2

    def test_depth_network_output_shape(self):
        n_rays = 4
        n_samples = 5
        rays_o = rays_d = torch.zeros(n_rays, 3)
        depth_network = depth_net.DepthNet()
        depth_z_vals = depth_network(rays_o, rays_d)
        assert depth_z_vals.shape == (n_rays, 1)


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


def test_z_vals_to_points():
    rays_o = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]])
    rays_d = torch.tensor([[1.0, 0, 0], [0.0, 1.0, 1]])
    z_vals = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 1.5, 3, 4]])
    expected_points = torch.tensor(
        [
            [[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]],
            [[1.0, 1, 1], [1.0, 1.5, 1.5], [1.0, 3, 3], [1.0, 4, 4]],
        ]
    )
    points = z_vals_to_points(rays_o=rays_o, rays_d=rays_d, z_vals=z_vals)
    assert points.shape == (2, 4, 3)
    assert torch.equal(expected_points, points)


def test_sample_gaussian():
    n_rays = 5
    n_samples = 10
    mean = torch.zeros(n_rays, 1, dtype=torch.float32)  # mean of zero
    std = torch.tensor(1.0)  # standard deviation of 1
    assert mean.shape == (n_rays, 1)
    samples = sample_gaussian(n_samples, mean, std)

    assert samples.shape == (
        n_rays,
        n_samples,
    ), f"Expected shape {(n_rays, n_samples)}, but got {samples.shape}"

    assert any(
        torch.allclose(samples[:, i], mean[:, 0]) for i in range(n_samples)
    ), "Mean value not found in the samples."

    # Check that values are within reasonable bounds for a Gaussian with std=1
    max_allowed = mean + 4 * std
    min_allowed = mean - 4 * std
    assert torch.all(
        samples <= max_allowed
    ), "Samples have values too high for expected Gaussian distribution."
    assert torch.all(
        samples >= min_allowed
    ), "Samples have values too low for expected Gaussian distribution."
