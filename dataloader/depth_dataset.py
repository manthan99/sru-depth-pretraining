# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

"""Unified dataloader for depth image and height scan VAE pretraining.

This module provides a PyTorch Dataset for loading either:
- Depth images from cameras (e.g., RealSense, ZED)
- Height scan images from terrain generators

Both data types are stored as float32 RGBA PNGs where each float32 value
is encoded as 4 RGBA bytes.

Usage:
    from depth_dataset import DepthImageDataset
    from torch.utils.data import DataLoader

    # For depth images
    config = {
        'data_type': 'depth',
        'local_data_root': './data/tartanair-v2',
        'min_depth': 0.25,
        'max_depth': 10.0,
        ...
    }

    # For height scans
    config = {
        'data_type': 'heightscan',
        'local_data_root': './data/heightscan',
        'min_depth': -2.0,
        'max_depth': 2.0,
        ...
    }

    dataset = DepthImageDataset(config)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
"""

import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Optional, List, Tuple


class RandomCropDownsample(nn.Module):
    """Randomly crop an image within a range and downsample to final size."""

    def __init__(self, crop_min: Tuple[int, int], crop_max: Tuple[int, int], final_size: Tuple[int, int]):
        """Initialize random crop and downsample module.

        Args:
            crop_min: Minimum crop size (height, width).
            crop_max: Maximum crop size (height, width).
            final_size: Final size for downsampling (height, width).
        """
        super(RandomCropDownsample, self).__init__()
        self.crop_min = crop_min
        self.crop_max = crop_max
        self.final_size = final_size
        self.resize_transform = T.Resize(
            size=self.final_size,
            antialias=False,
            interpolation=T.InterpolationMode.BILINEAR
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random crop and resize.

        Args:
            image: Input tensor of shape (H, W) or (C, H, W) or (B, C, H, W)

        Returns:
            Cropped and resized tensor
        """
        # Ensure input is at least 4D (B, C, H, W)
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)

        # Use the same random crop size for the entire batch
        # Only randomize height, keep width at max (consistent with original behavior)
        crop_height = random.randint(self.crop_min[0], self.crop_max[0])
        crop_transform = T.RandomCrop(size=(crop_height, self.crop_max[1]))

        # Apply the crop transform
        cropped_images = crop_transform(image)

        # Downsample to final size (H, W)
        downsampled_images = self.resize_transform(cropped_images)

        return downsampled_images.squeeze()


class DepthImageDataset(Dataset):
    """Unified dataset for loading depth images or height scans.

    Supports two data types configured via 'data_type' config key:
    - 'depth': Camera depth images (default)
    - 'heightscan': Terrain height scan images
    """

    # File patterns for different data types
    FILE_PATTERNS = {
        'depth': ['depth'],
        'heightscan': ['heightscan', 'height_scan', 'terrain'],
    }

    def __init__(self, dataloader_config: dict, is_cluster: bool = False):
        """Initialize the dataset.

        Args:
            dataloader_config: Configuration dictionary with keys:
                - data_type: 'depth' or 'heightscan' (default: 'depth')
                - local_data_root / euler_data_root: Data directory path
                - min_depth: Minimum valid value
                - max_depth: Maximum valid value
                - valid_ratio_threshold: Min fraction of valid pixels
                - crop_min: Minimum crop size
                - new_image_shape_hw: [H, W] for crop max
                - resize_shape_hw: [H, W] for final output
                - use_augmentation: Whether to apply augmentation (default: True)
                - image_list: Optional custom image list filename
            is_cluster: Whether running on cluster (uses euler_data_root)
        """
        # Determine data type
        self.data_type = dataloader_config.get('data_type', 'depth')
        if self.data_type not in self.FILE_PATTERNS:
            raise ValueError(f"Unknown data_type: {self.data_type}. Must be one of {list(self.FILE_PATTERNS.keys())}")

        # Set up paths
        if is_cluster:
            image_dir = dataloader_config.get('euler_data_root', './data')
            cluster_type = "euler"
        else:
            image_dir = dataloader_config.get('local_data_root', './data')
            cluster_type = "local"

        self.image_dir = image_dir
        self.image_files: List[str] = []

        # Value thresholds for valid pixel filtering
        self.min_depth = dataloader_config.get('min_depth', 0.1)
        self.max_depth = dataloader_config.get('max_depth', 10.0)
        self.valid_ratio_threshold = dataloader_config.get('valid_ratio_threshold', 0.10)

        # Get image list filename from config
        default_list_name = f'{cluster_type}_{self.data_type}_images.txt'
        image_list_file = dataloader_config.get('image_list', default_list_name)

        # Try to load the list from file
        try:
            with open(image_list_file, 'r') as f:
                self.image_files = f.read().splitlines()
            print(f"Loaded {len(self.image_files)} {self.data_type} images from {image_list_file}")
        except FileNotFoundError:
            print(f"{image_list_file} not found, loading images from directory: {image_dir}")
            self._load_images_from_directory(image_dir, image_list_file)

        # Initialize random crop downsample module
        crop_min = (dataloader_config.get('crop_min', 32), dataloader_config.get('crop_min', 32))
        crop_max = (
            dataloader_config.get('new_image_shape_hw', [64, 64])[0],
            dataloader_config.get('new_image_shape_hw', [64, 64])[1]
        )
        final_size = (
            dataloader_config.get('resize_shape_hw', [64, 64])[0],
            dataloader_config.get('resize_shape_hw', [64, 64])[1]
        )
        self.random_cropper = RandomCropDownsample(crop_min, crop_max, final_size)

        # Whether to use augmentation
        self.use_augmentation = dataloader_config.get('use_augmentation', True)

    def _matches_file_pattern(self, filename: str) -> bool:
        """Check if filename matches patterns for current data type."""
        filename_lower = filename.lower()
        patterns = self.FILE_PATTERNS[self.data_type]
        return any(pattern in filename_lower for pattern in patterns)

    def _load_images_from_directory(self, image_dir: str, list_filename: str):
        """Load image file paths from directory and filter by valid ratio."""
        # Find all matching PNG files
        for root, _, files in os.walk(image_dir):
            for f in files:
                if self._matches_file_pattern(f) and f.endswith('.png'):
                    self.image_files.append(os.path.join(root, f))

        pre_filter_count = len(self.image_files)
        print(f"Found {pre_filter_count} {self.data_type} images")

        if pre_filter_count == 0:
            print(f"Warning: No {self.data_type} images found!")
            return

        # Filter images by requiring sufficient valid pixel ratio
        filtered_files = []
        for file_path in tqdm(self.image_files, desc=f"Pre-check {self.data_type} images"):
            ratio = self._compute_valid_ratio(file_path, self.min_depth, self.max_depth)
            if ratio is None:
                continue
            if ratio >= self.valid_ratio_threshold:
                filtered_files.append(file_path)

        self.image_files = filtered_files
        print(
            f"Kept {len(self.image_files)} of {pre_filter_count} images after valid-pixel filter "
            f"(>={int(self.valid_ratio_threshold * 100)}% in ({self.min_depth}, {self.max_depth}))"
        )

        # Save the list to a txt file
        with open(list_filename, 'w') as f:
            for item in self.image_files:
                f.write(f"{item}\n")
        print(f"Saved image list to {list_filename}")

    def _compute_valid_ratio(self, file_path: str, min_depth: float, max_depth: float) -> Optional[float]:
        """Compute the ratio of valid (finite, in-range) pixels."""
        try:
            if not file_path.endswith('.png'):
                return None
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                return None
            values = self._rgba_to_float32(image)
            if values is None:
                return None
            # Compute valid pixel ratio: finite and within range
            valid = np.isfinite(values)
            valid &= (values > float(min_depth))
            valid &= (values < float(max_depth))
            total = values.size if hasattr(values, 'size') else 0
            if total == 0:
                return None
            return float(valid.sum()) / float(total)
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self.image_files)

    def _rgba_to_float32(self, rgba_image: np.ndarray) -> np.ndarray:
        """Convert RGBA encoded PNG back to float32 values."""
        values = rgba_image.view("<f4")
        return np.squeeze(values, axis=-1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample.

        Args:
            idx: Sample index

        Returns:
            Tensor of shape (H, W)
        """
        file_path = self.image_files[idx]

        if file_path.endswith('.png'):
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            # Convert RGBA to float32 values
            values = self._rgba_to_float32(image)
            # Convert to torch tensor
            values = torch.from_numpy(values)

            # Apply random crop and resize if augmentation enabled
            if self.use_augmentation:
                values = self.random_cropper(values)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        return values

# Usage example and testing
if __name__ == "__main__":
    import yaml
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Test depth/heightscan dataloader')
    parser.add_argument('--config', type=str, default='config/pretrain.yaml',
                        help='Path to config file')
    parser.add_argument('--data-type', type=str, choices=['depth', 'heightscan'],
                        help='Override data_type in config')
    args = parser.parse_args()

    def load_config(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    config = load_config(args.config)
    dataloader_config = config.get('dataloader', {})

    # Override data_type if specified
    if args.data_type:
        dataloader_config['data_type'] = args.data_type

    data_type = dataloader_config.get('data_type', 'depth')
    min_depth = dataloader_config.get('min_depth', 0.1)
    max_depth = dataloader_config.get('max_depth', 10.0)
    batch_size = dataloader_config.get('batch_size', 64)

    print(f"Testing {data_type} dataloader...")

    # Create the dataset and data loader
    dataset = DepthImageDataset(dataloader_config, is_cluster=False)

    if len(dataset) == 0:
        print(f"No {data_type} images found!")
        exit(1)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Iterate through the data loader
    for batch_idx, samples in enumerate(data_loader):
        print(f"Batch {batch_idx}: {samples.shape}")

        # Clamp values to the specified range
        samples = torch.clamp(samples, min_depth, max_depth)

        # Handle nan values
        samples[torch.isnan(samples)] = 0.0

        # Print max and min values
        print(f"Max value: {samples.max():.3f}")
        print(f"Min value: {samples.min():.3f}")

        # Visualize first few samples
        num_vis = min(4, samples.shape[0])
        fig, axes = plt.subplots(1, num_vis, figsize=(4 * num_vis, 4))

        for i in range(num_vis):
            sample = samples[i].squeeze().numpy()
            ax = axes[i] if num_vis > 1 else axes
            im = ax.imshow(sample, cmap='viridis', vmin=min_depth, vmax=max_depth)
            ax.set_title(f"Sample {i}")
            plt.colorbar(im, ax=ax)

        plt.suptitle(f"{data_type.capitalize()} - Batch {batch_idx}")
        plt.tight_layout()
        plt.savefig(f"{data_type}_batch_{batch_idx}.png", dpi=100)
        plt.show()

        # Only visualize first batch
        if batch_idx >= 0:
            break
