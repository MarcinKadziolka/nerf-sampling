import torch
import torch.nn as nn
import torch.nn.functional as F


class SphereMoreViewsNeRF(nn.Module):
    def __init__(
        self, input_ch=3,
        input_ch_views=3, output_ch=4, use_viewdirs=True, **kwargs
    ):
        """
        #TODO add docstring
        """
        super(SphereMoreViewsNeRF, self).__init__()
        self.D = 3
        self.W = 256
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.use_viewdirs = use_viewdirs
        input_dim = input_ch + input_ch_views

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_dim, self.W)] + [nn.Linear(self.W + input_ch_views, self.W)
                                         for i in range(self.D - 1)])

        self.views_linears = nn.ModuleList([nn.Linear(self.W, self.W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(self.W + input_ch_views, self.W)
            self.alpha_linear = nn.Linear(self.W + input_ch_views, 1)
            self.rgb_linear = nn.Linear(self.W // 2, 3)
        else:
            self.output_linear = nn.Linear(self.W, output_ch)

    def forward(self, x):
        """Do a forward pass through the network."""

        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views],
            dim=-1
        )
        h = torch.cat([input_pts, input_views], -1)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            h = torch.cat([input_views, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = feature

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs
