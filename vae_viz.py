# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

import torch
import numpy as np
import matplotlib.pyplot as plt
import pyzed.sl as sl

# Camera and processing configuration
ZED_RESOLUTION = sl.RESOLUTION.SVGA
ZED_FPS = 60
CAMERA_MIN_DEPTH = 0.2  # meters
CAMERA_MAX_DEPTH = 15.0  # meters

# Depth preprocessing
DEPTH_MIN_CLIP = 0.1  # meters
DEPTH_MAX_CLIP = 10.0  # meters
RESIZE_SHAPE = (40, 64)  # height, width

# Visualization
COLORMAP = "viridis"
FIGURE_SIZE = (12, 6)
UPDATE_PAUSE = 0.001  # seconds


def load_vae_model(model_path: str, device: str = "cpu"):
    """Load a JIT-compiled VAE model.

    Args:
        model_path: Path to the compiled model (.pt file)
        device: Device to load the model on ("cpu" or "cuda")

    Returns:
        Loaded VAE model in eval mode
    """
    try:
        vae_model = torch.jit.load(model_path)
        vae_model.to(device)
        vae_model.eval()
        print(f"Model loaded successfully on {device}")
        return vae_model
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise


def initialize_zed_camera():
    """Initialize and configure the ZED camera.

    Returns:
        Initialized ZED camera object
    """
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.camera_resolution = ZED_RESOLUTION
    init_params.camera_fps = ZED_FPS
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = CAMERA_MIN_DEPTH
    init_params.depth_maximum_distance = CAMERA_MAX_DEPTH

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Failed to open ZED camera")

    return zed


@torch.inference_mode()
def process_depth_image(zed, vae_model, runtime_params, mat_depth, device):
    """Process a depth frame from the ZED camera using the VAE model.

    Args:
        zed: ZED camera object
        vae_model: VAE model for encoding/decoding
        runtime_params: ZED runtime parameters
        mat_depth: ZED depth matrix
        device: Device to process on ("cpu" or "cuda")

    Returns:
        Tuple of (input_depth_np, decoded_depth_np) or (None, None) if grab failed
    """
    if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
        return None, None

    # Retrieve depth image from camera
    zed.retrieve_measure(mat_depth, sl.MEASURE.DEPTH)
    depth_image = mat_depth.get_data()  # float32 array

    # Preprocess depth image
    depth_image = np.clip(depth_image, DEPTH_MIN_CLIP, DEPTH_MAX_CLIP)
    depth_image = np.nan_to_num(depth_image, nan=0.0)

    # Convert to tensor and process through VAE
    depth_tensor = torch.tensor(depth_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    depth_tensor = depth_tensor.to(device)

    # Resize to VAE input dimensions
    depth_tensor = torch.nn.functional.interpolate(
        depth_tensor, size=RESIZE_SHAPE, mode="bilinear", align_corners=True
    )

    # Encode and decode
    encoded = vae_model(depth_tensor)
    decoded = vae_model.decode(encoded)

    # Convert back to numpy for visualization
    input_depth_np = depth_tensor.squeeze().cpu().numpy()
    decoded_depth_np = decoded.squeeze().cpu().numpy()

    return input_depth_np, decoded_depth_np


def normalize_depth_image(depth_image):
    """Normalize depth image to [0, 255] range.

    Args:
        depth_image: Input depth image (numpy array)

    Returns:
        Normalized depth image
    """
    depth_min = depth_image.min()
    depth_max = depth_image.max()
    depth_range = depth_max - depth_min + 1e-6

    return (depth_image - depth_min) / depth_range * 255


def main():
    """Main function to visualize VAE depth reconstruction from ZED camera."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and camera
    vae_model = load_vae_model("output/depth_vae_model_fuse_jit.pt", device=device)
    zed = initialize_zed_camera()

    # Setup ZED runtime parameters
    runtime_params = sl.RuntimeParameters()
    mat_depth = sl.Mat()

    # Setup visualization
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

    try:
        while True:
            input_depth, decoded_depth = process_depth_image(zed, vae_model, runtime_params, mat_depth, device)

            if decoded_depth is not None:
                # Normalize images for visualization
                norm_input = normalize_depth_image(input_depth)
                norm_decoded = normalize_depth_image(decoded_depth)

                # Update plots
                ax1.clear()
                ax1.imshow(norm_input, cmap=COLORMAP)
                ax1.set_title("Original Depth Image")

                ax2.clear()
                ax2.imshow(norm_decoded, cmap=COLORMAP)
                ax2.set_title("Decoded Depth Image")

                plt.draw()
                plt.pause(UPDATE_PAUSE)

    except KeyboardInterrupt:
        print("\nVisualization stopped by user")

    finally:
        plt.close()
        zed.close()


if __name__ == "__main__":
    main()
