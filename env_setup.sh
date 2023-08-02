#!/bin/bash
cd sphere_nerf_mod
git clone https://github.com/yenchenlin/nerf-pytorch.git
mv nerf-pytorch nerf_pytorch
cd nerf_pytorch
touch __init__.py