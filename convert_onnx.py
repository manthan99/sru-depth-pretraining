# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

"""
ONNX Export Script for VAE Depth Estimation Model

Supports deployment on:
- Jetson platforms (Orin, Xavier, Nano) via TensorRT
- Intel NUC platforms via OpenVINO or ONNX Runtime

Usage:
    python convert_onnx.py --model_path model_save/vae_pretrain.pth --platform jetson
    python convert_onnx.py --model_path model_save/vae_pretrain.pth --platform nuc
    python convert_onnx.py --model_path model_save/vae_pretrain.pth --platform generic
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from network import VAENet


class VAENetInference(nn.Module):
    """Wrapper for VAENet that outputs only depth for inference deployment."""

    def __init__(self, vae_net: VAENet):
        super().__init__()
        self.vae_net = vae_net

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only reconstructed depth.

        Args:
            depth: Input depth tensor [B, 1, H, W]

        Returns:
            Reconstructed depth tensor [B, 1, H, W]
        """
        out_dict = self.vae_net(depth)
        return out_dict["depth"]


class VAENetInferenceWithLatent(nn.Module):
    """Wrapper for VAENet that outputs depth and latent features."""

    def __init__(self, vae_net: VAENet):
        super().__init__()
        self.vae_net = vae_net

    def forward(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning depth and latent statistics.

        Args:
            depth: Input depth tensor [B, 1, H, W]

        Returns:
            Tuple of (reconstructed_depth, mu, logvar)
        """
        out_dict = self.vae_net(depth)
        return out_dict["depth"], out_dict["mu"], out_dict["logvar"]


class VAEEncoderDeploy(nn.Module):
    """Deployment wrapper that outputs only mu from the encoder."""

    def __init__(self, vae_net: VAENet):
        super().__init__()
        self.depth_encoder = vae_net.depth_encoder
        self.vae_sampler = vae_net.vae_sampler

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only mu (latent mean).

        Args:
            depth: Input depth tensor [B, 1, H, W]

        Returns:
            mu tensor [B, latent_dim, H', W']
        """
        feat = self.depth_encoder(depth)
        _, mu, _ = self.vae_sampler(feat)
        return mu


def get_platform_config(platform: str) -> dict:
    """Get ONNX export configuration for target platform.

    Args:
        platform: Target platform ('jetson', 'nuc', or 'generic')

    Returns:
        Dictionary with export configuration
    """
    configs = {
        "jetson": {
            "opset_version": 17,  # TensorRT 8.6+ supports opset 17
            "do_constant_folding": True,
            "dynamic_axes": None,  # Fixed batch size for TensorRT optimization
            "suffix": "_jetson",
            "description": "Optimized for NVIDIA Jetson (TensorRT)",
        },
        "nuc": {
            "opset_version": 17,  # OpenVINO and ONNX Runtime support opset 17
            "do_constant_folding": True,
            "dynamic_axes": {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            "suffix": "_nuc",
            "description": "Optimized for Intel NUC (OpenVINO/ONNX Runtime)",
        },
        "generic": {
            "opset_version": 14,  # Wide compatibility
            "do_constant_folding": True,
            "dynamic_axes": {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            "suffix": "",
            "description": "Generic ONNX with wide compatibility",
        },
    }

    if platform not in configs:
        raise ValueError(f"Unknown platform: {platform}. Choose from: {list(configs.keys())}")

    return configs[platform]


def export_onnx(
    model_path: str,
    output_path: str,
    platform: str = "generic",
    latent_dim: int = 64,
    input_height: int = 40,
    input_width: int = 64,
    batch_size: int = 1,
    include_latent: bool = False,
    deploy: bool = False,
    fp16: bool = False,
) -> None:
    """Export VAE model to ONNX format.

    Args:
        model_path: Path to the saved model weights (.pth file)
        output_path: Path to save the ONNX model (.onnx file)
        platform: Target platform ('jetson', 'nuc', or 'generic')
        latent_dim: Latent dimension of the VAE model
        input_height: Input depth map height
        input_width: Input depth map width
        batch_size: Batch size for export (fixed for Jetson)
        include_latent: Whether to include latent outputs (mu, logvar)
        deploy: If True, export only encoder with mu output for robot deployment
        fp16: Whether to export with FP16 weights (for Jetson)
    """
    config = get_platform_config(platform)
    print(f"\033[34mExporting for: {config['description']}\033[0m")

    # Load the VAE model
    vae_model = VAENet(latent_dim)

    try:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        vae_model.load_state_dict(state_dict, strict=True)
        print(f"\033[32mLoaded model weights from {model_path}\033[0m")
    except Exception as e:
        print(f"\033[31mFailed to load model weights: {e}\033[0m")
        raise

    # Wrap model for inference
    if deploy:
        model = VAEEncoderDeploy(vae_model)
        output_names = ["mu"]
        if config["dynamic_axes"]:
            config["dynamic_axes"]["mu"] = {0: "batch_size"}
            del config["dynamic_axes"]["output"]
        print("\033[34mDeploy mode: exporting encoder only (output: mu)\033[0m")
    elif include_latent:
        model = VAENetInferenceWithLatent(vae_model)
        output_names = ["depth", "mu", "logvar"]
        if config["dynamic_axes"]:
            config["dynamic_axes"]["depth"] = {0: "batch_size"}
            config["dynamic_axes"]["mu"] = {0: "batch_size"}
            config["dynamic_axes"]["logvar"] = {0: "batch_size"}
            del config["dynamic_axes"]["output"]
    else:
        model = VAENetInference(vae_model)
        output_names = ["output"]

    model.eval()

    # Convert to FP16 if requested (useful for Jetson)
    if fp16:
        model = model.half()
        dummy_input = torch.randn(batch_size, 1, input_height, input_width, dtype=torch.float16)
        print("\033[33mExporting with FP16 precision\033[0m")
    else:
        dummy_input = torch.randn(batch_size, 1, input_height, input_width)

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=config["opset_version"],
            do_constant_folding=config["do_constant_folding"],
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=config["dynamic_axes"],
        )
        print(f"\033[32mONNX model exported to {output_path}\033[0m")
    except Exception as e:
        print(f"\033[31mFailed to export ONNX model: {e}\033[0m")
        raise

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("\033[32mONNX model verification passed\033[0m")

        # Print model info
        print(f"\033[34mModel info:\033[0m")
        print(f"  - Opset version: {config['opset_version']}")
        print(f"  - Input shape: [{batch_size}, 1, {input_height}, {input_width}]")
        print(f"  - Dynamic batch: {config['dynamic_axes'] is not None}")
        print(f"  - Outputs: {output_names}")

    except ImportError:
        print("\033[33mWarning: onnx package not installed, skipping verification\033[0m")
    except Exception as e:
        print(f"\033[33mWarning: ONNX verification failed: {e}\033[0m")

    # Platform-specific post-processing hints
    print(f"\n\033[34mDeployment hints for {platform}:\033[0m")
    if platform == "jetson":
        print("  - Use trtexec to convert to TensorRT engine:")
        print(f"    trtexec --onnx={output_path} --saveEngine=model.engine --fp16")
        print("  - Or use TensorRT Python API for more control")
    elif platform == "nuc":
        print("  - For OpenVINO, use Model Optimizer:")
        print(f"    mo --input_model {output_path} --output_dir openvino_model/")
        print("  - Or use ONNX Runtime directly for inference")
    else:
        print("  - Compatible with most ONNX runtimes")
        print("  - Consider platform-specific optimizations for production")


def main():
    parser = argparse.ArgumentParser(
        description="Export VAE depth model to ONNX format for robot deployment"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_save/vae_pretrain_new.pth",
        help="Path to model weights (.pth file)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for ONNX model (auto-generated if not specified)",
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=["jetson", "nuc", "generic"],
        default="generic",
        help="Target deployment platform",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="VAE latent dimension",
    )
    parser.add_argument(
        "--input_height",
        type=int,
        default=40,
        help="Input depth map height",
    )
    parser.add_argument(
        "--input_width",
        type=int,
        default=64,
        help="Input depth map width",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for export",
    )
    parser.add_argument(
        "--include_latent",
        action="store_true",
        help="Include latent outputs (mu, logvar) in export",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Export only encoder with mu output for robot deployment",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export with FP16 precision (recommended for Jetson)",
    )

    args = parser.parse_args()

    # Auto-generate output path if not specified
    if args.output_path is None:
        model_name = Path(args.model_path).stem
        config = get_platform_config(args.platform)
        suffix = config["suffix"]
        deploy_suffix = "_deploy" if args.deploy else ""
        fp16_suffix = "_fp16" if args.fp16 else ""
        args.output_path = f"output/{model_name}{suffix}{deploy_suffix}{fp16_suffix}.onnx"

    export_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        platform=args.platform,
        latent_dim=args.latent_dim,
        input_height=args.input_height,
        input_width=args.input_width,
        batch_size=args.batch_size,
        include_latent=args.include_latent,
        deploy=args.deploy,
        fp16=args.fp16,
    )


if __name__ == "__main__":
    main()
