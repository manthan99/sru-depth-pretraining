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
    def __init__(self, latent_dim):
        super(VAENet, self).__init__()
        self.depth_encoder = DepthEncoder(latent_dim)
        
        self.vae_sampler = VAESampler(latent_dim, latent_dim)

        self.depth_decoder = DepthDecoder(latent_dim)

    def forward(self, depth):
        feat = self.depth_encoder(depth)
        
        feat, mu, logvar = self.vae_sampler(feat)
        
        depth_out = self.depth_decoder(feat)
        
        out_dict = {
            "depth": depth_out,
            'vae': (mu, logvar)
        }

        return out_dict
    
if __name__ == "__main__":
    latent_dim = 64
    
    vae = VAENet(latent_dim)
    depth = torch.randn(4, 1, 40, 64)
    out_dict = vae(depth)
    print("Depth shape:", out_dict['depth'].shape)
    print("Mu shape:", out_dict['vae'][0].shape)
    print("Logvar shape:", out_dict['vae'][1].shape)
