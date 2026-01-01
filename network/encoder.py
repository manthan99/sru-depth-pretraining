# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models import regnet_x_400mf, resnet18
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.regnet import RegNet_X_400MF_Weights


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# RegNetX-400MF encoder
# class Encoder(nn.Module):
#     def __init__(self, in_channel, out_channel, pos_enc_dim=8, pretrained=True):
#         super(Encoder, self).__init__()
#         # pixel positional encoding
#         self.pos_enc = PositionalEncodingPermute2D(pos_enc_dim)
#         # Feature encoder
#         self.feat_conv = nn.Sequential(
#             Conv2dNormActivation(in_channel + pos_enc_dim, 64, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=True)
#         )
#         weigths = RegNet_X_400MF_Weights.DEFAULT if pretrained else None
#         self.encoder = regnet_x_400mf(weights=weigths)
#         # Remove classification head from the encoder
#         self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
#         # Modify the first layer to accept the number of channels in the input image
#         self.encoder[0][0] = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
#         # Add a 1x1 convolution to map the output to the desired number of channels
#         self.final_conv = nn.Conv2d(400, out_channel, kernel_size=1, bias=True)

#     def forward(self, x):
#         pos_enc = self.pos_enc(x)
#         x = torch.cat([x, pos_enc], dim=1) # concatenate positional encoding with input
#         x = self.feat_conv(x) # Feature Fusion
#         # Encoder
#         x = self.encoder(x)
#         x = self.final_conv(x)
#         return x
    
class EncoderFPN(nn.Module):
    def __init__(self, in_channel, out_channel, pretrained=True):
        super(EncoderFPN, self).__init__()
        # pixel positional encoding
        weigths = RegNet_X_400MF_Weights.DEFAULT if pretrained else None
        encoder = regnet_x_400mf(weights=weigths)
        # Remove classification head from the encoder
        encoder = nn.Sequential(*list(encoder.children())[:-2])
        # Modify the first layer to accept the number of channels in the input image
        encoder[0][0] = nn.Conv2d(in_channel, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc = encoder[0]
        self.enc_1 = encoder[1][:2]
        self.enc_2 = encoder[1][2]
        self.enc_3 = encoder[1][3]
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork([64, 160, 400], out_channel)

    def forward(self, x):
        out = OrderedDict()
        x = self.enc(x)
        out['feat1'] = self.enc_1(x)
        out['feat2'] = self.enc_2(out['feat1'])
        out['feat3'] = self.enc_3(out['feat2'])
        
        out = self.fpn(out)
        
        return out['feat1']
    
class FuseEncoder(EncoderFPN):
    def __init__(self, out_channel, pretrained=True):
        super(FuseEncoder, self).__init__(4, out_channel, pretrained)
        
    def forward(self, rgb, depth):
        # check if depth has channel dimension
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        x = torch.cat([rgb, depth], dim=1)
        return super(FuseEncoder, self).forward(x)
    
# RGB image encoder
class RGBEncoder(EncoderFPN):
    def __init__(self, out_channel, pretrained=True):
        super(RGBEncoder, self).__init__(3, out_channel, pretrained)
        
# Depth image encoder
class DepthEncoder(EncoderFPN):
    def __init__(self, out_channel, pretrained=True):
        super(DepthEncoder, self).__init__(1, out_channel, pretrained)
        
    def forward(self, x):
        # check if depth has channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return super(DepthEncoder, self).forward(x)
        
        
# Test
if __name__ == "__main__":
    output_dim = 64
    rgb_encoder = RGBEncoder(output_dim)
    depth_encoder = DepthEncoder(output_dim)
    random_rgb_image = torch.rand((1, 3, 40, 64))
    random_depth_image = torch.rand((1, 1, 40, 64))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_encoder.to(device)
    depth_encoder.to(device)
    
    rgb_output = rgb_encoder(random_rgb_image.to(device))
    depth_output = depth_encoder(random_depth_image.to(device))
    
    print("RGB Encoder output shape:", rgb_output.shape)
    print("Depth Encoder output shape:", depth_output.shape)