# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

World model pretraining codebase for depth estimation using Variational Autoencoders (VAEs). Primary focus is single-frame depth estimation (`train_single.py`) using RealSense depth camera data, with additional support for sequence modeling with memory architectures.

## Primary Development Commands

**Main Training** (current branch: `pretrain-realsense`):
- `python train_single.py`: Train single-frame VAE depth estimation model
  - Uses `config/pretrain.yaml`
  - Dataset: `DepthImageDataset` with RealSense depth images
  - Model: `VAENet` (DepthEncoder → VAESampler → DepthDecoder)

**Other Training Scripts**:
- `python train_seq.py`: Train sequence VAE with GRU/SRT memory (uses TartanAir)
- `python train_seq_pose.py`: Sequence model with pose prediction
- `python noise_visualizer.py`: Visualize depth noise models

## train_single.py Architecture

**Data Flow**:
1. Load depth images from text file list (`local_depth_images_realsense.txt`)
2. `DepthImageDataset` reads images, applies preprocessing (crop, resize, normalization)
3. Apply depth noise augmentation with probability `noise_prob`:
   - `DepthNoise` (parametric): Learned noise model based on normalized depth differences
   - `DepthNoiseBaseline`: Fixed baseline noise
4. Forward pass through `VAENet`:
   - `DepthEncoder`: Conv layers extract features to latent space
   - `VAESampler`: Reparameterization trick (mu, logvar → sampled z)
   - `DepthDecoder`: Reconstruct depth from latent
5. Loss computation: Huber reconstruction loss + β-weighted KL divergence
6. Beta scheduling: Linear interpolation from `init_beta` to `final_beta`

**Key Components for train_single.py**:
- `network/vae_net.py`: `VAENet` - single-frame VAE model
- `network/encoder.py`: `DepthEncoder` - convolutional encoder
- `network/decoder.py`: `DepthDecoder` - convolutional decoder
- `network/vae.py`: `VAESampler` - reparameterization module
- `network/noise_utils/depth_noise.py`: `DepthNoise` - parametric noise model
- `network/noise_utils/depth_noise_baseline.py`: `DepthNoiseBaseline`
- `dataloader/depth_dataset.py`: `DepthImageDataset` - single-frame depth loader

**Configuration (config/pretrain.yaml)**:
- `dataloader.resize_shape_hw`: Final depth map size (e.g., [40, 64])
- `dataloader.new_image_shape_hw`: Initial crop size (e.g., [640, 640])
- `dataloader.crop_min`: Minimum crop dimension
- `dataloader.min_depth` / `max_depth`: Depth range for clamping
- `dataloader.valid_ratio_threshold`: Filter images with insufficient valid depth
- `training.noise_type`: `'parametric'` or `'baseline'`
- `training.noise_prob`: Probability of applying noise augmentation
- `training.init_beta` / `final_beta`: KL loss weight scheduling
- `training.epochs`: Number of training epochs
- `model.latent_dim`: VAE latent space dimension

**Loss Function** (in `train_single.py`):
- Reconstruction: Huber loss between predicted and target depth
- KL divergence: `-0.5 * mean(1 + logvar - mu^2 - exp(logvar))`
- Total loss: `recon_loss + beta * kl_loss`

**Utilities**:
- `load_pretrained_weights()`: Load partial weights matching by name and size
- `ssim_similarity()`: SSIM metric for validation

## Sequence Models (Optional Context)

**Architecture Types** (controlled by `model_type` in config):
- `'vae'`: Single-frame (`VAENet`)
- `'gru'`: ConvGRU memory (`VAEGRUNet`)
- `'srt'`: Set Representation Transformer memory (`VAESRTNet`)

**Sequence Model Flow** (train_seq.py):
- Depth encoder → VAE sampler → Memory block (GRU/SRT) → Depth decoder
- Memory maintains hidden state across frames
- Pose encoder embeds SE3 poses (using PyPose library)

**Key Files**:
- `network/vae_srt.py`: Sequence models with memory
- `network/memory.py`: `MemoryBlockConvGRU`, `MemoryBlockConvGRUSRT`
- `dataloader/dataloader.py`: `TartanAirDataLoader` for sequences