import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MoreDirectionVectorInfo(nn.Module):
    def __init__(
        # self, D=8, W=256, input_ch=3,
        # input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False
        self, D=16, W=512, input_ch=3,
        input_ch_views=3, output_ch=4, skips=[4, 8, 12], use_viewdirs=False
    ):
        """
        #TODO add docstring
        """
        super(MoreDirectionVectorInfo, self).__init__()
        D = 16
        W = 512
        skips = [4, 8, 12]
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        input_dim = input_ch + input_ch_views

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_dim, W)]
            + [nn.Linear(W + input_ch_views, W)
                if i not in self.skips
                else nn.Linear(W + input_dim, W)
                for i in range(D - 1)])

        # self.views_linears = nn.ModuleList([nn.Linear(W + input_ch_views, W // 2)])
        self.views_linears = nn.ModuleList(
            [nn.Linear(W + input_ch_views, W // 2)]
            + [nn.Linear(W // 2, W // 2) for i in range(D // 4)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W + input_ch_views, W)
            self.alpha_linear = nn.Linear(W + input_ch_views, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        """
        #TODO add docstring
        """
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = torch.cat([input_pts, input_views], -1)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, input_views, h], -1)
            else:
                h = torch.cat([input_views, h], dim=-1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([input_views, feature], dim=-1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
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
