# Depth Perception Pretraining - SRU Project

[![Paper](https://img.shields.io/badge/IJRR-2025-blue)](https://journals.sagepub.com/home/ijr)
[![Website](https://img.shields.io/badge/Project-Website-green)](https://michaelfyang.github.io/sru-project-website/)

> **📌 Important Note**: This repository contains the **depth perception pretraining module** for the SRU project, providing large-scale self-supervised depth perception pretraining on 100K+ synthetic environments. See the [project website](https://michaelfyang.github.io/sru-project-website/) for the broader navigation system.

## Overview

VAE-based depth perception pretraining optimized for RealSense and ZED depth cameras. This repository provides:

- **Training**: Single-frame depth VAE encoder-decoder model (`train_single.py`)
- **Deployment**: TorchScript (`convert_jit.py`) and ONNX (`convert_onnx.py`) model export
- **Visualization**: Real-time depth reconstruction with ZED cameras (`vae_viz.py`)

The VAE learns to compress and reconstruct depth images for robotics, 3D scene understanding, and depth-based world models.

### What's Included

- ✅ Single-frame depth VAE encoder-decoder architecture
- ✅ Depth noise augmentation (parametric and baseline models)
- ✅ TorchScript and ONNX export for deployment (Jetson, Intel NUC, generic)
- ✅ Real-time visualization and monitoring
- ✅ Multi-camera support (RealSense D435, ZED X)
- ✅ Flexible YAML-based configuration

### Related Projects

- [sru-pytorch-spatial-learning](https://github.com/michaelfyang/sru-pytorch-spatial-learning) - Core SRU architecture
- [SRU Project Website](https://michaelfyang.github.io/sru-project-website/) - Complete navigation system

## Project Structure

```
.
├── train_single.py              # Main training script for single-frame VAE
├── convert_jit.py               # Export model to TorchScript format
├── convert_onnx.py              # Export model to ONNX format (Jetson/NUC/generic)
├── test_export.py               # Verify exported model correctness
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

**Requirements:**
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+ (optional)

**Setup:**
```bash
# Create conda environment
conda create -n depth-vae python=3.10
conda activate depth-vae

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml numpy matplotlib
pip install pyzed-sl  # Optional: For ZED camera support
```

## Usage

### Training

Train a single-frame depth VAE:
```bash
python train_single.py
```

Uses configuration from `config/pretrain.yaml`. Key components:
- **Dataset**: Depth images listed in `local_depth_images_realsense.txt`
- **Model**: `VAENet` with configurable latent dimension
- **Loss**: Huber reconstruction + β-weighted KL divergence
- **Augmentation**: Optional parametric or baseline depth noise

#### Configuration Files

The project provides camera-specific configurations:

| Config | Camera | Notes |
|--------|--------|-------|
| `pretrain.yaml` | Base settings | Template configuration |
| `pretrain_realsense.yaml` | RealSense D435 | Calibrated for Intel RealSense |
| `pretrain_zedx.yaml` | ZED X | Stereolabs ZED X optimized |

**Base Configuration Example** (`config/pretrain.yaml`):
```yaml
dataloader:
  resize_shape_hw: [40, 64]      # Output depth map size
  new_image_shape_hw: [640, 640] # Initial crop size
  min_depth: 0.1
  max_depth: 10.0
  valid_ratio_threshold: 0.3

training:
  noise_type: 'parametric'       # or 'baseline'
  noise_prob: 0.5
  init_beta: 0.0
  final_beta: 0.001
  epochs: 100
  batch_size: 32
  learning_rate: 1e-3

model:
  latent_dim: 64
```

To use a different config, modify the config path in `train_single.py` or pass it as an argument.

#### Pre-trained Models

Available in `model_save/release_model/`:
- **ZED X**: VAE model optimized for Stereolabs ZED X depth camera
  - Uses `config/pretrain_zedx.yaml`
  - Ready for fine-tuning or deployment via `vae_viz.py`

### Model Export

Export trained models for deployment on various platforms.

#### TorchScript Export

Convert to TorchScript format for C++ deployment:

```bash
# Full VAE model
python convert_jit.py --model_path model_save/vae_pretrain_new.pth

# Encoder-only for robot deployment (outputs latent mu)
python convert_jit.py --model_path model_save/vae_pretrain_new.pth --deploy
```

#### ONNX Export

Export to ONNX format for cross-platform deployment:

```bash
# Generic ONNX (wide compatibility)
python convert_onnx.py --model_path model_save/vae_pretrain_new.pth

# NVIDIA Jetson (TensorRT optimized)
python convert_onnx.py --model_path model_save/vae_pretrain_new.pth --platform jetson

# Intel NUC (OpenVINO/ONNX Runtime)
python convert_onnx.py --model_path model_save/vae_pretrain_new.pth --platform nuc

# Encoder-only deploy mode
python convert_onnx.py --model_path model_save/vae_pretrain_new.pth --platform jetson --deploy
```

**Platform-specific options:**

| Platform | Opset | Dynamic Batch | Use Case |
|----------|-------|---------------|----------|
| `generic` | 14 | Yes | Wide compatibility |
| `jetson` | 17 | No | NVIDIA Jetson + TensorRT |
| `nuc` | 17 | Yes | Intel NUC + OpenVINO |

#### Verify Exports

Test exported models for numerical correctness:

```bash
python test_export.py --model_path model_save/vae_pretrain_new.pth
```

This runs comprehensive tests including:
- Determinism verification
- Batch consistency
- JIT export correctness
- ONNX export correctness (all platforms)
- Deploy mode verification

### Real-time Visualization

Visualize depth reconstruction with ZED camera:

```bash
python vae_viz.py
```

This script:
1. Loads a JIT-compiled VAE model
2. Captures live depth frames from ZED camera
3. Encodes and decodes frames in real-time
4. Displays original vs. reconstructed depth side-by-side

**Stop with:** `Ctrl+C`

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

This project implements realistic stereo depth noise models for training robust depth encoders. Noise augmentation helps models generalize to real sensor data.

### Parametric Noise Model

Located in [network/noise_utils/depth_noise.py](network/noise_utils/depth_noise.py)

Simulates realistic stereo camera artifacts through disparity-space filtering:

- **Edge Noise**: Removes depth discontinuities at object boundaries via local disparity filtering
- **Filling Noise**: Fills invalid pixels from neighboring regions (simulates occlusion handling)
- **Quantization Noise**: Disparity quantization at 1/32 precision (matches stereo hardware)

**Configuration:**
```yaml
training:
  noise_type: 'parametric'
  noise_prob: 0.5
```

**Advanced Parameters:**
```python
DepthNoise(
    focal_length=50.0,
    baseline=0.12,                # Stereo baseline (meters)
    filter_size=3,
    inlier_thred_range=(0.01, 0.05),
    prob_range=(0.4, 0.6),
    min_depth=0.1,
    max_depth=10.0
)
```

### Baseline Noise Model

Located in [network/noise_utils/depth_noise_baseline.py](network/noise_utils/depth_noise_baseline.py)

Generic depth sensor noise for synthetic data augmentation:

- **Gaussian noise**: Additive measurement noise
- **Missing data**: Contiguous invalid regions (occlusions, low texture)
- **Salt-and-pepper**: Random isolated invalid pixels
- **Spatial jitter**: Small translational shifts
- **Gaussian blur**: Optical smoothing

**Configuration:**
```yaml
training:
  noise_type: 'baseline'
  noise_prob: 0.5
```

### Visualization & Tuning

Visualize noise effects:
```bash
python noise_visualizer.py
```

**Best Practices:**
- **Real sensor data**: Use parametric noise to match your camera characteristics
- **Synthetic data**: Use baseline noise as general augmentation
- **Tuning `noise_prob`**:
  - `0.1-0.3`: Light regularization, faster training
  - `0.5`: Balanced robustness and speed
  - `0.8-1.0`: Strong regularization, may need more epochs
- **Camera tuning**: Adjust `focal_length` and `baseline` parameters for your sensor

## Data Format

**Depth Image Dataset** format:
- Text file listing depth image paths (one per line)
- Images in NumPy/OpenCV formats (.npy, .png, etc.)
- Depth values in meters

**Example file** (`local_depth_images_realsense.txt`):
```
/path/to/depth_image_001.npy
/path/to/depth_image_002.npy
/path/to/depth_image_003.npy
...
```

## Key Features

- **Efficient architecture**: RegNet backbone with FPN for multi-scale feature extraction
- **Depth noise augmentation**: Parametric and baseline sensor noise models
- **TorchScript compilation**: Export for C++ deployment
- **Real-time visualization**: Monitor depth reconstruction with ZED camera
- **Flexible configuration**: YAML-based hyperparameter control
- **GPU support**: Automatic CUDA device detection and placement

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

**ZED Camera Not Detected**
- Verify camera is connected via USB
- Install ZED SDK: `pip install pyzed-sl`
- Run ZED diagnostics to test connection

**TorchScript Compilation Issues**
- Ensure all operations are TorchScript-compatible
- Check PyTorch version compatibility
- Verify model is in `eval()` mode

**Poor Reconstruction Quality**
- Verify data preprocessing (normalization, depth range)
- Increase training epochs or adjust learning rate
- Increase `final_beta` to strengthen KL regularization
- Check noise augmentation parameters match your camera
