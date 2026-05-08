import json
import os
import random
from typing import Callable, List, Optional, Sequence

import cv2
import glob
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import webdataset as wds
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from network import DepthNoise, DepthNoiseBaseline

IMAGENET_MEAN = [0.3347, 0.5781, 0.4711]
IMAGENET_STD = [0.2514, 0.3264, 0.3328]

def _select_depth_sample(sample):
    if "png" in sample and "json" in sample:
        return sample["png"], sample["json"]
    if "depth.png" in sample and "meta.json" in sample:
        return sample["depth.png"], sample["meta.json"]
    return None


def _not_none(value):
    return value is not None


class WebDatasetDepthPNG:
    """WebDataset depth PNG loader for SRU VAE pretraining.

    The input path mirrors rl_nav/data/webdataset_vision_png.py: metric depth is
    decoded first, optional sensor noise is applied on that single-channel
    metric depth, and only then the depth map is converted to three channels and
    normalized with the DINOv3/ImageNet statistics.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        images_per_shard: int = 3200,
        shard_pattern: str = "*.tar",
        shuffle_buffer: int = 1000,
        dataset_list_file: Optional[str] = None,
        shard_files: Optional[List[str]] = None,
        resampled: bool = True,
        noise_type: str = "parametric",
        noise_prob: float = 0.0,
        min_depth: float = 0.25,
        max_depth: float = 10.0,
        focal_length: float = 28.0,
        baseline: float = 0.12,
        resize_shape_hw: Sequence[int] = (64, 64),
        new_image_shape_hw: Sequence[int] = (640, 640),
        crop_min: int = 320,
        use_augmentation: bool = True,
        log_space_target: bool = False,
        input_mean: Sequence[float] = IMAGENET_MEAN,
        input_std: Sequence[float] = IMAGENET_STD,
        max_depth_ch0: float = 100.0,
        max_depth_ch1: float = 10.0,
    ):
        self.root = root
        self.shard_files = shard_files or self._get_shard_files(root, shard_pattern, dataset_list_file)
        self.estimated_num_samples = max(1, len(self.shard_files)) * images_per_shard
        self.transform = transform
        self.noise_prob = noise_prob
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.resize_shape_hw = tuple(resize_shape_hw)
        self.new_image_shape_hw = tuple(new_image_shape_hw)
        self.crop_min = crop_min
        self.use_augmentation = use_augmentation
        self.log_space_target = log_space_target
        self.normalize_input = T.Normalize(mean=input_mean, std=input_std)
        self.max_depth_ch0 = max_depth_ch0
        self.max_depth_ch1 = max_depth_ch1

        if noise_type == "baseline":
            self.depth_noise = DepthNoiseBaseline(
                focal_length=focal_length,
                baseline=baseline,
                min_depth=min_depth,
                max_depth=max_depth,
            ).eval()
        elif noise_type == "parametric":
            self.depth_noise = DepthNoise(
                focal_length=focal_length,
                baseline=baseline,
                min_depth=min_depth,
                max_depth=max_depth,
            ).eval()
        elif noise_type in ("none", None):
            self.depth_noise = None
            self.noise_prob = 0.0
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        self.dataset = (
            wds.WebDataset(
                self.shard_files,
                resampled=resampled,
                nodesplitter=wds.split_by_node,
                shardshuffle=True,
                empty_check=False,
            )
            .shuffle(shuffle_buffer)
            .decode()
            .map(_select_depth_sample)
            .select(_not_none)
            .map(self.process_sample)
        )

    def _get_shard_files(
        self,
        root: str,
        shard_pattern: str,
        dataset_list_file: Optional[str],
    ) -> List[str]:
        if dataset_list_file and os.path.exists(dataset_list_file):
            with open(dataset_list_file, "r") as f:
                dataset_folders = [line.strip() for line in f if line.strip()]

            shard_files = []
            for folder in dataset_folders:
                folder_path = os.path.join(root, folder)
                if os.path.exists(folder_path):
                    shard_files.extend(glob.glob(os.path.join(folder_path, shard_pattern)))
            return sorted(shard_files)

        if not os.path.exists(root):
            raise FileNotFoundError(f"WebDataset root does not exist: {root}")

        contains_subfolders = any(os.path.isdir(os.path.join(root, entry)) for entry in os.listdir(root))
        if contains_subfolders:
            return sorted(glob.glob(os.path.join(root, "**", shard_pattern), recursive=True))
        return sorted(glob.glob(os.path.join(root, shard_pattern)))

    def process_sample(self, sample):
        png_data, metadata = sample
        metadata = self._decode_metadata(metadata)
        clean_depth = self.decode_png_metric_depth(png_data, metadata)
        clean_depth = self._spatial_transform(clean_depth)

        input_depth = clean_depth
        if self.depth_noise is not None and random.random() < self.noise_prob:
            with torch.no_grad():
                input_depth = self.depth_noise(clean_depth.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        input_depth = self.convert_metric_to_three_channel_depth(input_depth)
        input_depth = self.normalize_input(input_depth)
        
        target_depth = self.convert_metric_to_three_channel_depth(clean_depth)
        target_depth = self.normalize_input(target_depth)

        if self.transform is not None:
            input_depth = self.transform(input_depth)

        return input_depth.contiguous(), target_depth.contiguous()

    def _decode_metadata(self, metadata):
        if isinstance(metadata, dict):
            return metadata
        if isinstance(metadata, bytes):
            return json.loads(metadata.decode("utf-8", errors="ignore"))
        if isinstance(metadata, str):
            return json.loads(metadata)
        return {}

    def decode_png_metric_depth(self, png_data, metadata, depth_multiplier=None):
        img = self._decode_png_array(png_data)
        if img is None:
            raise ValueError("Failed to decode PNG depth sample")

        img_np = img.astype(np.float32)
        dataset = metadata.get("dataset", "Unknown")

        if depth_multiplier is None:
            depth_multiplier = metadata.get("depth_multiplier", 1.0)
            depth_resolution = metadata.get("depth_resolution", 1.0)
            if depth_resolution != 1.0:
                depth_multiplier = depth_resolution
            if dataset == "MetaGraspNetSyn":
                depth_multiplier = depth_multiplier / 100.0

        if img.dtype == np.uint8:
            if len(img_np.shape) != 2:
                raise ValueError(f"Unsupported 8-bit image shape: {img_np.shape}")
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            metric_depth = self.max_depth_ch1 * np.exp(-5.0 * img_np)
        elif img.dtype == np.uint16:
            if dataset in ("hm3d", "taskonomy"):
                img_np[img_np >= 65530] = 0
            metric_depth = img_np / depth_multiplier
        else:
            raise ValueError(f"Unsupported PNG dtype: {img.dtype}")

        metric_depth = np.nan_to_num(metric_depth, nan=0.0, posinf=0.0, neginf=0.0)
        metric_depth = np.clip(metric_depth, 0.0, self.max_depth_ch0).astype(np.float32)
        return torch.from_numpy(metric_depth)

    def _decode_png_array(self, png_data):
        if isinstance(png_data, np.ndarray):
            return png_data
        if isinstance(png_data, Image.Image):
            return np.array(png_data)
        if isinstance(png_data, bytes):
            img_array = np.frombuffer(png_data, np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        if hasattr(png_data, "read"):
            return cv2.imdecode(np.frombuffer(png_data.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        if isinstance(png_data, str):
            with open(png_data, "rb") as f:
                return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        return None

    def _spatial_transform(self, depth: torch.Tensor) -> torch.Tensor:
        depth = depth.unsqueeze(0)
        depth = TF.resize(
            depth,
            list(self.new_image_shape_hw),
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=False,
        )

        if self.use_augmentation:
            crop_h = random.randint(min(self.crop_min, self.new_image_shape_hw[0]), self.new_image_shape_hw[0])
            crop_w = self.new_image_shape_hw[1]
            i, j, h, w = T.RandomCrop.get_params(depth, output_size=(crop_h, crop_w))
            depth = TF.crop(depth, i, j, h, w)

        depth = TF.resize(
            depth,
            list(self.resize_shape_hw),
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=False,
        )
        return depth.squeeze(0)

    def convert_metric_to_three_channel_depth(self, metric_depth: torch.Tensor) -> torch.Tensor:
        metric_depth = torch.nan_to_num(metric_depth, nan=0.0, posinf=0.0, neginf=0.0)
        metric_depth = torch.clamp(metric_depth, min=0.0, max=self.max_depth_ch0)
        log_depth = torch.log1p(metric_depth)

        max0 = torch.tensor(self.max_depth_ch0, dtype=metric_depth.dtype, device=metric_depth.device)
        max1 = torch.tensor(self.max_depth_ch1, dtype=metric_depth.dtype, device=metric_depth.device)
        channel_1 = log_depth / torch.log1p(max0)
        channel_2 = torch.clamp(log_depth / torch.log1p(max1), 0.0, 1.0)

        min_log_depth = torch.amin(log_depth)
        max_log_depth = torch.amax(log_depth)
        denom = max_log_depth - min_log_depth
        if denom > 0:
            channel_3 = (log_depth - min_log_depth) / denom
        else:
            channel_3 = torch.zeros_like(log_depth)

        return torch.stack([channel_1, channel_2, channel_3], dim=0)

    def normalize_target_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = torch.clamp(depth, min=0.0, max=self.max_depth)
        if self.log_space_target:
            depth = torch.log1p(depth)
        return depth.unsqueeze(0)

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.estimated_num_samples


class WebDatasetDepthDataModule(LightningDataModule):
    def __init__(self, dataloader_config: dict, training_config: dict):
        super().__init__()
        self.dataloader_config = dataloader_config
        self.training_config = training_config
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        common_kwargs = self._dataset_kwargs()
        train_root = self.dataloader_config["train_data_root"]
        val_root = self.dataloader_config.get("val_data_root", train_root)

        self.train_dataset = WebDatasetDepthPNG(
            train_root,
            noise_prob=self.training_config.get("noise_prob", 0.0),
            use_augmentation=self.dataloader_config.get("use_augmentation", True),
            **common_kwargs,
        )
        self.val_dataset = WebDatasetDepthPNG(
            val_root,
            noise_prob=0.0,
            use_augmentation=False,
            **common_kwargs,
        )

    def _dataset_kwargs(self):
        return {
            "images_per_shard": self.dataloader_config.get("images_per_shard", 3200),
            "shard_pattern": self.dataloader_config.get("shard_pattern", "*.tar"),
            "shuffle_buffer": self.dataloader_config.get("shuffle_buffer", 1000),
            "dataset_list_file": self.dataloader_config.get("dataset_list"),
            "resampled": self.dataloader_config.get("resampled", True),
            "noise_type": self.training_config.get("noise_type", "parametric"),
            "min_depth": self.dataloader_config.get("min_depth", 0.25),
            "max_depth": self.dataloader_config.get("max_depth", 10.0),
            "focal_length": self.dataloader_config.get("focal_length", 28.0),
            "baseline": self.dataloader_config.get("baseline", 0.12),
            "resize_shape_hw": self.dataloader_config.get("resize_shape_hw", [40, 64]),
            "new_image_shape_hw": self.dataloader_config.get("new_image_shape_hw", [640, 640]),
            "crop_min": self.dataloader_config.get("crop_min", 320),
            "log_space_target": self.training_config.get("log_space", False),
            "input_mean": self.dataloader_config.get("input_mean", IMAGENET_MEAN),
            "input_std": self.dataloader_config.get("input_std", IMAGENET_STD),
            "max_depth_ch0": self.dataloader_config.get("max_depth_ch0", 100.0),
            "max_depth_ch1": self.dataloader_config.get("max_depth_ch1", 10.0),
        }

    def train_dataloader(self):
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def _dataloader(self, dataset):
        num_workers = self.dataloader_config.get("num_workers", 4)
        return DataLoader(
            dataset.dataset,
            batch_size=self.dataloader_config.get("batch_size", 256),
            num_workers=num_workers,
            pin_memory=self.dataloader_config.get("pin_memory", True),
            persistent_workers=num_workers > 0,
        )
