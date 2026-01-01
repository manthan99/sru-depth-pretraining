# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
        if len(image.shape) == 2: # Add channel dimension if missing
            image = image.unsqueeze(0).unsqueeze(0)

        # Use the same random crop size for the entire batch
        crop_height = random.randint(self.crop_min[0], self.crop_max[0])
        crop_transform = T.RandomCrop(size=(crop_height, self.crop_max[1])) # make the width the maximum crop width

        # Apply the crop transform to the entire batch
        cropped_images = crop_transform(image)

        # Downsample to final size (H, W)
        downsampled_images = self.resize_transform(cropped_images)

        return downsampled_images.squeeze()


# Dataset class to load depth images from a directory
class DepthImageDataset(Dataset):
    def __init__(self, dataloader_config, is_cluster=False):
        if is_cluster:
            image_dir = dataloader_config.get('euler_data_root', './data/tartanair-v2')
            cluster_type = "euler"
        else:
            image_dir = dataloader_config.get('local_data_root', './data/tartanair-v2')
            cluster_type = "local"
        self.image_dir = image_dir
        # List image files containing 'depth' in the name and having '.png' extension
        self.image_files = []
        # thresholds for valid depth pixel filtering
        self.min_depth = dataloader_config.get('min_depth', 0.1)
        self.max_depth = dataloader_config.get('max_depth', 10.0)
        self.valid_ratio_threshold = dataloader_config.get('valid_ratio_threshold', 0.10)

        # Get depth image list filename from config, fallback to default cluster-based naming
        depth_image_list = dataloader_config.get('depth_image_list', f'{cluster_type}_depth_images.txt')

        # try to load the list from the txt file
        try:
            with open(depth_image_list, 'r') as f:
                self.image_files = f.read().splitlines()

            print(f"Loaded {len(self.image_files)} depth images from {depth_image_list}")
        except:
            print("depth_images.txt not found, loading images from directory: ", image_dir)
            # if the list is empty, load the images from the directory
            for root, _, files in os.walk(image_dir):
                for f in files:
                    # if 'depth' in f.lower() and (f.endswith('.png') or f.endswith('.pt')):    
                    if 'depth' in f.lower() and (f.endswith('.png')):
                        self.image_files.append(os.path.join(root, f))
                        
            # record pre-filter count and print the total number of depth images found
            pre_filter_count = len(self.image_files)
            print(f"Found {pre_filter_count} depth images")

            # filter images by requiring sufficient valid pixel ratio
            filtered_files = []
            for file_path in tqdm(self.image_files, desc="Pre-check depth images", total=pre_filter_count):
                ratio = self._compute_valid_ratio(file_path, self.min_depth, self.max_depth)
                if ratio is None:
                    continue
                if ratio >= self.valid_ratio_threshold:
                    filtered_files.append(file_path)
            self.image_files = filtered_files

            print(
                f"Kept {len(self.image_files)} of {pre_filter_count} images after valid-pixel filter (>={int(self.valid_ratio_threshold * 100)}% in ({self.min_depth}, {self.max_depth}))"
            )
            
            # save the list into a txt file
            with open(depth_image_list, 'w') as f:
                for item in self.image_files:
                    f.write("%s\n" % item)
                
        # init random crop downsample module
        crop_min = (dataloader_config['crop_min'], dataloader_config['crop_min'])
        crop_max = (dataloader_config['new_image_shape_hw'][0], dataloader_config['new_image_shape_hw'][1])
        final_size = (dataloader_config['resize_shape_hw'][0], dataloader_config['resize_shape_hw'][1])
        self.random_cropper = RandomCropDownsample(crop_min, crop_max, final_size)

    def _compute_valid_ratio(self, file_path, min_depth, max_depth):
        try:
            if not file_path.endswith('.png'):
                return None
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                return None
            depth = self.depth_rgba_float32(image)
            if depth is None:
                return None
            # compute valid pixel ratio: finite and within (min_depth, max_depth)
            valid = np.isfinite(depth)
            valid &= (depth > float(min_depth))
            valid &= (depth < float(max_depth))
            total = depth.size if hasattr(depth, 'size') else 0
            if total == 0:
                return None
            return float(valid.sum()) / float(total)
        except Exception:
            return None

    def __len__(self):
        return len(self.image_files)
    
    def depth_rgba_float32(self, depth_rgba):
        depth = depth_rgba.view("<f4")
        return np.squeeze(depth, axis=-1)

    def __getitem__(self, idx):
        file_path = self.image_files[idx]
        
        # if file_path.endswith('.pt'):
        #     depth = torch.load(file_path, weights_only=True).squeeze()
        if file_path.endswith('.png'):
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            # Convert RGBA depth image to float32 depth values
            depth = self.depth_rgba_float32(image)
            # Convert to torch tensor
            depth = torch.from_numpy(depth)
            # random crop and resize the depth image
            depth = self.random_cropper(depth)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        return depth

# Usage
if __name__ == "__main__":
    import yaml
    import matplotlib.pyplot as plt
    
    def load_config(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    config = load_config('/home/fanyang1/Projects/world_model_pretrain/config/pretrain.yaml')
    
    dataloader_config = config.get('dataloader', {})

    min_depth = dataloader_config.get('min_depth', 0.1)
    max_depth = dataloader_config.get('max_depth', 10.0)
    batch_size = dataloader_config.get('batch_size', 64)

    # Create the dataset and data loader
    depth_dataset = DepthImageDataset(dataloader_config, is_cluster=False)
    data_loader = DataLoader(depth_dataset, batch_size=batch_size, shuffle=True)

    # Iterate through the data loader
    for batch_idx, depth_scans in enumerate(data_loader):
        print(f"Batch {batch_idx}: {depth_scans.shape}")
        # Clamp the depth values to the specified range
        depth_scans = torch.clamp(depth_scans, min_depth, max_depth)
        # handle nan values
        depth_scans[torch.isnan(depth_scans)] = 0.0
        
        # Print max and min value of the depth scans
        print(f"Max value: {depth_scans.max()}")
        print(f"Min value: {depth_scans.min()}")

        # Visualize the depth scans
        for i in range(depth_scans.shape[0]):
            depth_scan = depth_scans[i].squeeze().numpy()
            plt.imshow(depth_scan, cmap='viridis')
            plt.title(f"Batch {batch_idx}, Sample {i}")
            plt.colorbar()
            plt.show()
