# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from network import DepthEncoder, DepthDecoder, VAESampler

class VAENet(torch.nn.Module):
    def __init__(self, latent_dim: int, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.depth_encoder = DepthEncoder(latent_dim, in_channel=in_channels)

        self.vae_sampler = VAESampler(latent_dim, latent_dim)

        self.depth_decoder = DepthDecoder(latent_dim, out_dim=out_channels)

    def forward(self, depth: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.depth_encoder(depth)

        feat, mu, logvar = self.vae_sampler(feat)

        depth_out = self.depth_decoder(feat)

        out_dict: dict[str, torch.Tensor] = {
            "depth": depth_out,
            "mu": mu,
            "logvar": logvar,
        }

        return out_dict
    
if __name__ == "__main__":
    latent_dim = 64

    vae = VAENet(latent_dim)
    depth = torch.randn(4, 1, 40, 64)
    out_dict = vae(depth)
    print("Depth shape:", out_dict['depth'].shape)
    print("Mu shape:", out_dict['mu'].shape)
    print("Logvar shape:", out_dict['logvar'].shape)
