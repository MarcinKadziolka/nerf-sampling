import torch
import torch.nn as nn
import torch.nn.functional as F


class SphereWithoutViewsNeRF(nn.Module):
    def __init__(
        self, input_ch=3, skips=None,
        input_ch_views=3, output_ch=4, use_viewdirs=True, **kwargs
    ):
        """
        #TODO add docstring
        """
        super(SphereWithoutViewsNeRF, self).__init__()
        self.D = 2
        self.W = 256
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.use_viewdirs = use_viewdirs
        input_dim = input_ch + input_ch_views

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W),
             nn.Linear(W, W),
             nn.Linear(W, W)
             ])
        self.pts_linears1 = nn.ModuleList([nn.Linear(W + input_ch_views, W),
             nn.Linear(W, W)])

        self.pts_linears2 = nn.ModuleList(
            [nn.Linear(input_ch, W),
             nn.Linear(W, W),
             nn.Linear(W, W)
             ])
        self.pts_linears3 = nn.ModuleList([nn.Linear(W + input_ch, W),
                  nn.Linear(W, W)])

        self.views_linears = nn.ModuleList([nn.Linear(W, W // 2)])

        self.lin = nn.Linear(W, W)

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        """Do a forward pass through the network."""

        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views],
            dim=-1
        )
        h1 = torch.cat([input_pts, input_views], -1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)

        h = torch.cat([input_views, h], -1)
        for i, l in enumerate(self.pts_linears1):
            h = self.pts_linears1[i](h)
            h = F.relu(h)

        h2 = input_pts
        for i, l in enumerate(self.pts_linears2):
            h2 = self.pts_linears2[i](h2)
            h2 = F.relu(h2)

        h2 = torch.cat([input_pts, h2], -1)
        for i, l in enumerate(self.pts_linears3):
            h2 = self.pts_linears3[i](h2)
            h2 = F.relu(h2)

        h = self.lin(h2 * h)

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
