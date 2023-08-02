#!/bin/bash
cd sphere_nerf_mod
git clone https://github.com/yenchenlin/nerf-pytorch.git
git checkout 63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd
mv nerf-pytorch nerf_pytorch
cd nerf_pytorch
touch __init__.py