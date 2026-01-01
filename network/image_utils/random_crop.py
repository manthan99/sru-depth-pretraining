# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

import torch
import torch.nn as nn
import torchvision.transforms as T
import random

class RandomCropDownsample(nn.Module):
    def __init__(self, crop_min, crop_max, final_size): # (H, W)
        """
        Randomly crop an image within a range and downsample to final size.

        Parameters:
            crop_min (tuple): Minimum crop size for (height, width).
            crop_max (tuple): Maximum crop size for (height, width).
            final_size (tuple): Final size for downsampling (height, width).
        """
        super(RandomCropDownsample, self).__init__()
        self.crop_min = crop_min
        self.crop_max = crop_max
        self.final_size = final_size
        self.resize_transform = T.Resize(size=self.final_size, antialias=False, interpolation=T.InterpolationMode.BILINEAR)
        
    def forward(self, image):
        # Ensure input is a batch of images (B, C, H, W)
        if len(image.shape) == 3: # Add channel dimension if missing
            image = image.unsqueeze(1)

        # Use the same random crop size for the entire batch
        crop_height = random.randint(self.crop_min[0], self.crop_max[0])
        crop_transform = T.RandomCrop(size=(crop_height, self.crop_max[1])) # make the width the maximum crop width

        # Apply the crop transform to the entire batch
        cropped_images = crop_transform(image)

        # Downsample to final size (H, W)
        downsampled_images = self.resize_transform(cropped_images)

        return downsampled_images

# Example usage
if __name__ == "__main__":
    # random generator an image with batch dimension
    batch_size = 2
    ori_h, ori_w = 640, 640
    channels = 1
    image = torch.rand((batch_size, channels, ori_h, ori_w))

    # Define crop min and max (H, W)
    crop_min = (200, 200)  # Minimum crop height and width
    crop_max = (500, 500)  # Maximum crop height and width

    # Create the module
    random_crop_downsample = RandomCropDownsample(crop_min, crop_max)

    # Crop and downsample
    output_image = random_crop_downsample(image)

    # Print tensor shape
    print(output_image.shape)  # Should be [C, 64, 40] or [B, C, 64, 40]