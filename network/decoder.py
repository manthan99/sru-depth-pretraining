# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

import torch
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation

class VAEDecoder(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(VAEDecoder, self).__init__()
        self.input_dim = input_dim
        self.conv = Conv2dNormActivation(input_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.decoder = nn.Sequential(
            # Layer 0
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            # Layer 1
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            # Layer 2
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            # Final Output Layer
            nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1)
        ) # Expand the spatial dimensions by a factor of 2**4=16
    
    def forward(self, z):
        z =  self.conv(z)
        img = self.decoder(z)
        return img
    
    
class RGBDecoder(VAEDecoder):
    def __init__(self, input_dim):
        super(RGBDecoder, self).__init__(input_dim, 3)
        
class DepthDecoder(VAEDecoder):
    def __init__(self, input_dim):
        super(DepthDecoder, self).__init__(input_dim, 1)
    
# Test
if __name__ == "__main__":
    input_dim = 64

    rgb_vae_decoder = RGBDecoder(input_dim)
    depth_vae_decoder = DepthDecoder(input_dim)
    
    random_feat = torch.rand((1, input_dim, 14, 14))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    rgb_vae_decoder.to(device)
    rgb_img = rgb_vae_decoder(random_feat.to(device))
    
    print("RGB Decoder output shape:", rgb_img.shape)
    
    depth_vae_decoder.to(device)
    depth_img = depth_vae_decoder(random_feat.to(device))
    
    print("Depth Decoder output shape:", depth_img.shape)
