# Depth Perception Pretraining - SRU Project

[![Paper](https://img.shields.io/badge/IJRR-2025-blue)](https://journals.sagepub.com/home/ijr)
[![Website](https://img.shields.io/badge/Project-Website-green)](https://michaelfyang.github.io/sru-project-website/)

> **📌 Important Note**: This repository contains the **depth perception pretraining module** for the SRU (Spatially-Enhanced Recurrent Units) project. It provides large-scale self-supervised depth perception pretraining on 100K+ synthetic environments, serving as a foundational component for the broader navigation system described in the [project website](https://michaelfyang.github.io/sru-project-website/).

## About This Repository

This repository provides **VAE-based depth perception pretraining** optimized for RealSense and ZED depth cameras, including pre-trained models and sensor preprocessing pipelines.

**Scope of this repository:**
- ✅ Single-frame depth VAE encoder-decoder architecture
- ✅ Depth noise augmentation (parametric and baseline models)
- ✅ TorchScript model compilation for deployment
- ✅ Real-time depth reconstruction visualization
- ✅ Multi-camera support (RealSense D435, ZED X)
- ✅ Training pipelines with configurable hyperparameters

**Related repositories:**
- 🔗 [sru-pytorch-spatial-learning](https://github.com/michaelfyang/sru-pytorch-spatial-learning) - Core SRU architecture and spatial-temporal memorization experiments
- 🔗 [Main Navigation System](https://michaelfyang.github.io/sru-project-website/) - Complete end-to-end navigation system

## Overview

This project provides a complete pipeline for:
- **Training** a VAE-based depth encoder-decoder model (`train_single.py`)
- **Compiling** models to TorchScript format for deployment (`convert_jit.py`)
- **Real-time visualization** of depth reconstruction using ZED cameras (`vae_viz.py`)

The VAE learns to compress and reconstruct depth images, with applications in robotics, 3D scene understanding, and depth-based world models.

## Project Structure

```
.
├── train_single.py              # Main training script for single-frame VAE
├── convert_jit.py               # Convert trained model to TorchScript format
├── vae_viz.py                   # Real-time depth visualization with ZED camera
├── config/                      # Configuration files
│   ├── pretrain.yaml           # Base configuration
│   ├── pretrain_realsense.yaml # RealSense-specific settings
│   └── pretrain_zedx.yaml      # ZED X camera settings
├── network/                     # VAE architecture components
│   ├── vae_net.py              # Main VAE model
│   ├── encoder.py              # Depth/RGB encoders with RegNet backbone
│   ├── decoder.py              # Decoder for depth reconstruction
│   ├── vae.py                  # VAE sampler (reparameterization trick)
│   ├── noise_utils/            # Depth noise models
│   │   ├── depth_noise.py      # Parametric noise augmentation
│   │   └── depth_noise_baseline.py
│   └── image_utils/            # Image processing utilities
│       ├── random_crop.py      # Cropping and downsampling
│       └── image_warper.py     # Depth warping utilities
└── dataloader/                  # Data loading
    └── depth_dataset.py        # DepthImageDataset for loading depth images
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+ (optional, for GPU acceleration)

### Setup

1. Clone or download this repository
2. Create and activate a conda environment:
```bash
conda create -n depth-vae python=3.10
conda activate depth-vae
```

3. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml numpy matplotlib
pip install pyzed-sl  # For ZED camera support (optional)
```

## Usage

### Training a VAE Model

Train a single-frame depth VAE:

```bash
python train_single.py
```

The script uses configuration from `config/pretrain.yaml`. Key parameters:

- **Dataset**: List of depth images in `local_depth_images_realsense.txt`
- **Model**: `VAENet` with configurable latent dimension
- **Loss**: Huber reconstruction loss + β-weighted KL divergence
- **Noise augmentation**: Optional parametric or baseline depth noise

#### Configuration

This project supports multiple depth camera systems through dedicated configuration files:

**`config/pretrain_realsense.yaml`** - RealSense D435 Configuration
- Optimized for Intel RealSense D435 depth camera
- Calibrated depth ranges and preprocessing parameters
- Use this for training with RealSense D435 data

**`config/pretrain_zedx.yaml`** - ZED X Configuration
- Optimized for Stereolabs ZED X depth camera
- Camera-specific depth calibration and noise parameters
- Use this for training with ZED X data

**`config/pretrain.yaml`** - Base Configuration
```yaml
dataloader:
  resize_shape_hw: [40, 64]      # Output depth map size
  new_image_shape_hw: [640, 640] # Crop size
  min_depth: 0.1                 # Depth clamping
  max_depth: 10.0
  valid_ratio_threshold: 0.3     # Filter images with insufficient valid depth

training:
  noise_type: 'parametric'       # 'parametric' or 'baseline'
  noise_prob: 0.5                # Probability of applying noise
  init_beta: 0.0                 # Initial KL weight
  final_beta: 0.001              # Final KL weight
  epochs: 100
  batch_size: 32
  learning_rate: 1e-3

model:
  latent_dim: 64                 # VAE latent space dimension
```

To use a camera-specific config, modify `train_single.py` to load the desired config file or set the config path as a command-line argument.

### Compiling Models to TorchScript

Convert a trained PyTorch model to TorchScript format for deployment:

```bash
python convert_jit.py
```

By default, this loads `model_save/vae_pretrain_new.pth` and saves to `output/vae_pretrain_new_jit.pt`.

To customize:
```python
from convert_jit import compile_vae_model

compile_vae_model(
    model_path="path/to/your/model.pth",
    output_path="path/to/output.pt",
    latent_dim=64
)
```

### Real-time Visualization with ZED Camera

Visualize depth reconstruction in real-time:

```bash
python vae_viz.py
```

This script:
1. Loads a JIT-compiled VAE model
2. Captures depth frames from a connected ZED camera
3. Encodes and decodes each frame
4. Displays original vs. reconstructed depth side-by-side

**Controls**: Press `Ctrl+C` to stop

## Model Architecture

### VAE Components

1. **Encoder** (`DepthEncoder`)
   - Input: Depth image (1 channel, 40×64 or configurable)
   - Backbone: RegNet-X-400MF with Feature Pyramid Network (FPN)
   - Output: Feature maps → latent mean (μ) and log-variance (log σ²)

2. **Sampler** (`VAESampler`)
   - Reparameterization: z = μ + σ * ε, where ε ~ N(0, I)
   - Enables gradient-based learning of latent parameters

3. **Decoder** (`DepthDecoder`)
   - Input: Latent vector z
   - Transposed convolutions for upsampling
   - Output: Reconstructed depth image (40×64 or configured size)

### Loss Function

```
Loss = Reconstruction Loss + β * KL Divergence
     = Huber(depth_pred, depth_true) - 0.5 * Σ(1 + log σ² - μ² - σ²)
```

- **Reconstruction Loss**: Huber loss for robust depth prediction
- **KL Divergence**: Regularizes latent space to N(0, I)
- **β Scheduling**: Linear interpolation from `init_beta` to `final_beta`

## Noise Modeling

This project implements parallelizable stereo depth perception noise models to simulate realistic sensor artifacts. Noise augmentation is crucial for training robust depth encoders that generalize to real sensor data.

### Parametric Stereo Noise Model

The **parametric noise model** (`DepthNoise` in [network/noise_utils/depth_noise.py](network/noise_utils/depth_noise.py)) simulates realistic stereo camera noise characteristics through disparity-space filtering:

**Sensor Artifacts Simulated**:
1. **Edge Noise**: Depth discontinuities at object boundaries are detected and removed using local disparity filtering
2. **Filling Noise**: Invalid (occluded or untextured) pixels are filled from neighboring valid measurements, simulating occlusion handling
3. **Round Noise**: Disparity quantization (1/32 precision) simulates integer rounding in stereo correlation

**Implementation Details**:
- Converts depth → disparity space (where stereo noise is naturally expressed)
- Computes local mean disparity and identifies outlier pixels
- Applies adaptive thresholding on normalized disparity differences
- Fills invalid pixels from neighboring regions with distance-weighted averaging
- Quantizes disparities with 1/32 precision for realistic sensor quantization
- Converts filtered disparity back to depth space

**Configuration Parameters**:
```yaml
training:
  noise_type: 'parametric'
  noise_prob: 0.5              # 50% of batches receive noise augmentation
```

**Advanced Parameters** (in code):
```python
DepthNoise(
    focal_length=50.0,         # Camera focal length (pixels)
    baseline=0.12,             # Stereo baseline distance (meters)
    min_depth=0.1,             # Minimum measurable depth
    max_depth=10.0,            # Maximum measurable depth
    filter_size=3,             # Local filtering kernel size
    inlier_thred_range=(0.01, 0.05),  # Normalized disparity threshold range
    prob_range=(0.4, 0.6),     # Stochastic matching probability range
    invalid_disp=1e7           # Invalid disparity marker
)
```

### Baseline Noise Model

The **baseline noise model** (`DepthNoiseBaseline` in [network/noise_utils/depth_noise_baseline.py](network/noise_utils/depth_noise_baseline.py)) applies generic depth sensor noise:

**Noise Components**:
- **Gaussian noise**: Additive depth measurement noise
- **Missing data**: Contiguous regions of invalid depth (occlusions, low texture)
- **Salt-and-pepper noise**: Random isolated invalid pixels
- **Spatial shift**: Small translational jitter in depth measurements
- **Gaussian blur**: Optical smoothing effects

**Configuration**:
```yaml
training:
  noise_type: 'baseline'
  noise_prob: 0.5
```

**Advanced Parameters**:
```python
DepthNoiseBaseline(
    focal_length=50.0,
    baseline=0.12,
    gaussian_noise_std=0.1,          # Additive Gaussian noise
    missing_data_prob=0.001,          # Probability of invalid pixels
    salt_pepper_noise_prob=0.02,      # Outlier noise probability
    gaussian_shift_std=0.5,           # Spatial jitter (pixels)
    gaussian_blur_kernel_size=3,      # Blur kernel size
    gaussian_blur_sigma=0.5,          # Blur sigma
    min_depth=0.1,
    max_depth=10.0
)
```

### Using Noise Models in Training

Noise augmentation is applied stochastically during training:

```python
# In train_single.py
depth_noise = DepthNoise(...)  # or DepthNoiseBaseline(...)

for batch in dataloader:
    depth = batch

    # Apply noise augmentation with probability noise_prob
    if random() < noise_prob:
        depth = depth_noise(depth)

    # Forward pass through VAE
    latent = encoder(depth)
    reconstructed = decoder(sampler(latent))
    loss = huber_loss(reconstructed, depth) + beta * kl_loss(...)
```

### Noise Visualization

Visualize and compare noise models:

```bash
python noise_visualizer.py
```

This script:
- Displays original depth images alongside noise-augmented versions
- Compares parametric vs. baseline noise effects
- Helps validate noise parameters match your sensor characteristics

### Best Practices

1. **For real sensor data**: Use parametric noise model to match stereo camera characteristics
2. **For synthetic data**: Use baseline noise as a generic augmentation strategy
3. **Tuning noise_prob**:
   - Low values (0.1-0.3): Lighter regularization, faster convergence
   - Medium values (0.5): Balanced robustness and training speed
   - High values (0.8-1.0): Stronger regularization, may require more epochs
4. **Camera-specific tuning**: Adjust `focal_length` and `baseline` for your camera sensor

## Data Format

### Depth Image Dataset

The `DepthImageDataset` expects:
- A text file listing depth image paths (one per line)
- Depth images in NumPy/OpenCV formats (.npy, .png, etc.)
- Depth values in meters

Example `local_depth_images_realsense.txt`:
```
/path/to/depth_image_001.npy
/path/to/depth_image_002.npy
/path/to/depth_image_003.npy
...
```

## Key Features

✅ **Efficient architecture**: RegNet backbone with FPN for multi-scale feature extraction
✅ **Depth noise augmentation**: Parametric and baseline noise models
✅ **TorchScript compilation**: Export models for C++ deployment
✅ **Real-time visualization**: Monitor training quality with ZED camera feedback
✅ **Flexible configuration**: YAML-based experiment configuration
✅ **GPU support**: Automatic CUDA detection and device placement

## License

MIT License - See [LICENSE](LICENSE) file for details

Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich

## Citation

If you use this codebase in your research, please cite:

```bibtex
@article{yang2025sru,
  author = {Yang, Fan and Frivik, Per and Hoeller, David and Wang, Chen and Cadena, Cesar and Hutter, Marco},
  title = {Spatially-enhanced recurrent memory for long-range mapless navigation via end-to-end reinforcement learning},
  journal = {The International Journal of Robotics Research},
  year = {2025},
  doi = {10.1177/02783649251401926},
  url = {https://doi.org/10.1177/02783649251401926}
}
```

## Contact

**Author**: Fan Yang
**Email**: fanyang1@ethz.ch
**Affiliation**: Robotic Systems Lab, ETH Zurich

## Troubleshooting

### ZED Camera Not Detected
- Verify ZED camera is connected via USB
- Check ZED SDK installation: `pip install pyzed-sl`
- Run ZED SDK diagnostics to test camera

### Model Won't Compile to TorchScript
- Ensure all operations are TorchScript-compatible
- Check PyTorch version compatibility
- Verify model has `eval()` mode properly set

### Poor Reconstruction Quality
- Check data preprocessing (normalization, depth range)
- Increase training epochs
- Adjust β scheduling (increase `final_beta`)
- Verify noise augmentation settings
