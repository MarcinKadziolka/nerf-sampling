import torch
import torch.nn as nn
import torch.nn.functional as F


class SphereTwoRGBNeRF(nn.Module):
    def __init__(
        self, D=8, W=256, input_ch=3, input_ch_views=3,
            output_ch=4, skips=[4], first_rgb_output_layer=4,
            use_viewdirs=False
    ):
        """
        Create a model with two RGB outputs.

        The model first outputs predicted RGB values after pts_linears layers,
        then after views_linears it outputs predicted RGB the second time,
        along with predicted sigma value.
        """
        super(SphereTwoRGBNeRF, self).__init__()
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
            + [nn.Linear(W + input_ch_views, W)
               for _ in range(D - first_rgb_output_layer - 1)]
            + [nn.Linear(W + input_ch_views, W // 2)]
        )

        if use_viewdirs:
            self.sigma_linear = nn.Linear(W // 2, 1)
            self.first_rgb_linear = nn.Linear(W, 3)
            self.second_rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        """Do a forward pass through the network."""

        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views],
            dim=-1
        )
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
