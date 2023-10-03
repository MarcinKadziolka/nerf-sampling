import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SphereTwoRGB(nn.Module):
    def __init__(
        self, D=8, W=256, input_ch=3, input_ch_views=3,
            output_ch=4, skips=[4], first_rgb_output_layer=4, use_viewdirs=False
    ):
        """
        #TODO add docstring
        """
        super(SphereTwoRGB, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.first_rgb_output_layer = first_rgb_output_layer
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [nn.Linear(W, W) for _ in range(first_rgb_output_layer)]
        )

        self.views_linears = nn.ModuleList(
            [nn.Linear(W, W)]
            + [nn.Linear(W + input_ch_views, W) for _ in range(D - first_rgb_output_layer - 1)]
            + [nn.Linear(W + input_ch_views, W // 2)]
        )

        if use_viewdirs:
            self.sigma_linear = nn.Linear(W // 2, 1)
            self.first_rgb_linear = nn.Linear(W, 3)
            self.second_rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        """
        #TODO add docstring
        """
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)

        if self.use_viewdirs:
            first_rgb = self.first_rgb_linear(h)

            for i, l in enumerate(self.views_linears):
                if i > 0:
                    h = torch.cat([h, input_views], -1)
                h = self.views_linears[i](h)
                h = F.relu(h)

            second_rgb = self.second_rgb_linear(h)
            sigma = self.sigma_linear(h)

            final_rgb = first_rgb * second_rgb
            outputs = torch.cat([final_rgb, sigma], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        """
        #TODO add docstring
        """
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear + 1]))
