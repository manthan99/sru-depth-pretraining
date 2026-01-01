# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

import torch
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation

class VAESampler(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAESampler, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.conv = Conv2dNormActivation(input_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Convolutional Layers for 2D mean and logvar
        self.mean_layers = nn.Sequential(
            Conv2dNormActivation(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)
        )
        
        self.logvar_layers = nn.Sequential(
            Conv2dNormActivation(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)
        )
    
    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = self.conv(x)
        mu = self.mean_layers(x)
        logvar = self.logvar_layers(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
if __name__ == "__main__":
    latent_dim = 256
    input_dim = 3
    vae = VAESampler(input_dim, latent_dim)
    x = torch.randn(4, 3, 224, 224)
    z, mu, logvar = vae(x)
    print("Latent shape:", z.shape)
    print("Mean shape:", mu.shape)
    print("Logvar shape:", logvar.shape)