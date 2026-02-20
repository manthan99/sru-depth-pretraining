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


def _onnx_export(model: nn.Module, dummy_input: torch.Tensor, output_path: str, **kwargs) -> None:
    """Call torch.onnx.export with the TorchScript-based exporter.

    PyTorch 2.6+ changed the default to dynamo=True (which requires onnxscript).
    Passing dynamo=False forces the stable TorchScript exporter with no extra
    dependencies.  For PyTorch < 2.1 the dynamo kwarg did not exist yet, so we
    fall back gracefully.
    """
    try:
        torch.onnx.export(model, dummy_input, output_path, dynamo=False, **kwargs)
    except TypeError:
        # PyTorch < 2.1: dynamo parameter not yet available
        torch.onnx.export(model, dummy_input, output_path, **kwargs)


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


class DepthPreprocess(nn.Module):
    """Preprocessing module for raw depth input (e.g., ZED camera).

    Performs the following steps:
    1. Replaces NaN and Inf values with 0.0
    2. Zeros out pixels outside [min_depth, max_depth] (invalid range -> 0.0)
    3. Resizes to the encoder input resolution using nearest-neighbor interpolation
    """

    def __init__(
        self,
        min_depth: float,
        max_depth: float,
        output_height: int = 40,
        output_width: int = 64,
    ):
        """Initialize depth preprocessor.

        Args:
            min_depth: Minimum valid depth (e.g. 0.25 m for ZED X)
            max_depth: Maximum valid depth (e.g. 10.0 m for ZED X)
            output_height: Encoder input height (default: 40)
            output_width: Encoder input width (default: 64)
        """
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.output_height = output_height
        self.output_width = output_width

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """Preprocess raw depth input.

        Args:
            depth: Raw depth tensor [B, 1, H, W] in metres

        Returns:
            Preprocessed depth tensor [B, 1, output_height, output_width]
        """
        # 1. Zero out pixels outside valid range.
        # IEEE 754 comparisons with NaN always return False, so NaN/Inf pixels
        # automatically get valid=False without a separate nan_to_num call.
        # torch.where is used instead of multiplication to avoid NaN*0=NaN.
        valid = (depth >= self.min_depth) & (depth <= self.max_depth)
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        # 2. Resize to encoder input resolution with bilinear interpolation
        # (matches training pipeline in RandomCropDownsample)
        depth = torch.nn.functional.interpolate(
            depth,
            size=(self.output_height, self.output_width),
            mode="bilinear",
            align_corners=False,
        )
        return depth


class VAEDeployWithPreprocess(nn.Module):
    """End-to-end model: raw depth preprocessing + VAE encoder (mu output).

    Accepts raw camera depth (any H x W), preprocesses it, and returns the
    latent mean (mu) from the VAE encoder.  Intended for robot deployment
    where the camera output is fed directly to this module.
    """

    def __init__(self, preprocess: DepthPreprocess, vae_net: VAENet):
        """Initialize combined model.

        Args:
            preprocess: DepthPreprocess module configured for the camera
            vae_net: Loaded VAENet model
        """
        super().__init__()
        self.preprocess = preprocess
        self.encoder_deploy = VAEEncoderDeploy(vae_net)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            depth: Raw depth tensor [B, 1, H, W] in metres

        Returns:
            mu tensor [B, latent_dim, H', W']
        """
        depth = self.preprocess(depth)
        return self.encoder_deploy(depth)


class VAEFullWithPreprocess(nn.Module):
    """End-to-end model: raw depth preprocessing + full VAE (depth reconstruction output).

    Accepts raw camera depth (any H x W), preprocesses it, and returns the
    reconstructed depth from the full VAE (encoder + decoder).
    """

    def __init__(self, preprocess: DepthPreprocess, vae_net: VAENet):
        """Initialize combined model.

        Args:
            preprocess: DepthPreprocess module configured for the camera
            vae_net: Loaded VAENet model
        """
        super().__init__()
        self.preprocess = preprocess
        self.vae_inference = VAENetInference(vae_net)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            depth: Raw depth tensor [B, 1, H, W] in metres

        Returns:
            Reconstructed depth tensor [B, 1, encoder_H, encoder_W]
        """
        depth = self.preprocess(depth)
        return self.vae_inference(depth)


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
    static: bool = False,
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
        static: If True, disable all dynamic axes (required for TensorRT conversion)
    """
    config = get_platform_config(platform)
    if static:
        config["dynamic_axes"] = None
        print("\033[33mStatic mode: dynamic axes disabled (all shapes fixed)\033[0m")
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
        _onnx_export(
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


def export_onnx_with_preprocess(
    model_path: str,
    output_path: str,
    platform: str = "generic",
    latent_dim: int = 64,
    min_depth: float = 0.25,
    max_depth: float = 10.0,
    encoder_height: int = 40,
    encoder_width: int = 64,
    raw_height: int = 600,
    raw_width: int = 960,
    batch_size: int = 1,
    deploy: bool = False,
    fp16: bool = False,
    static: bool = False,
) -> None:
    """Export a combined preprocessing + VAE model to ONNX.

    The exported model accepts raw camera depth (e.g. ZED X SVGA 960x600) and
    internally handles NaN/Inf replacement, range masking, resizing, and
    VAE encoding/decoding.

    Preprocessing steps baked into the graph:
      1. NaN / ±Inf  -> 0.0
      2. depth < min_depth or depth > max_depth -> 0.0
      3. Nearest-neighbor resize to (encoder_height, encoder_width)

    Args:
        model_path: Path to the saved model weights (.pth file)
        output_path: Path to save the ONNX model (.onnx file)
        platform: Target platform ('jetson', 'nuc', or 'generic')
        latent_dim: Latent dimension of the VAE model
        min_depth: Minimum valid depth in metres (values below -> 0.0)
        max_depth: Maximum valid depth in metres (values above -> 0.0)
        encoder_height: Encoder input height after resize (default: 40)
        encoder_width: Encoder input width after resize (default: 64)
        raw_height: Raw camera input height (default: 600 for ZED X SVGA)
        raw_width: Raw camera input width (default: 960 for ZED X SVGA)
        batch_size: Batch size for export (fixed for Jetson TensorRT)
        deploy: If True, export encoder only (output: mu). If False, export
                full VAE (output: reconstructed depth).
        fp16: Whether to export with FP16 weights (for Jetson)
        static: If True, disable all dynamic axes (required for TensorRT conversion)
    """
    config = get_platform_config(platform)
    if static:
        config["dynamic_axes"] = None
        print("\033[33mStatic mode: dynamic axes disabled (all shapes fixed)\033[0m")
    print(f"\033[34mExporting with preprocessing for: {config['description']}\033[0m")
    print(f"\033[34mPreprocess: min_depth={min_depth}, max_depth={max_depth}, "
          f"resize -> ({encoder_height}, {encoder_width})\033[0m")

    # Load VAE model
    vae_model = VAENet(latent_dim)
    try:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        vae_model.load_state_dict(state_dict, strict=True)
        print(f"\033[32mLoaded model weights from {model_path}\033[0m")
    except Exception as e:
        print(f"\033[31mFailed to load model weights: {e}\033[0m")
        raise

    # Build preprocessing module
    preprocess = DepthPreprocess(
        min_depth=min_depth,
        max_depth=max_depth,
        output_height=encoder_height,
        output_width=encoder_width,
    )

    # Select model variant and output names
    if deploy:
        model = VAEDeployWithPreprocess(preprocess, vae_model)
        output_names = ["mu"]
        print("\033[34mDeploy mode: preprocess + encoder only (output: mu)\033[0m")
    else:
        model = VAEFullWithPreprocess(preprocess, vae_model)
        output_names = ["depth"]
        print("\033[34mFull VAE mode: preprocess + full VAE (output: depth)\033[0m")

    model.eval()

    if fp16:
        model = model.half()
        dummy_input = torch.randn(batch_size, 1, raw_height, raw_width, dtype=torch.float16)
        print("\033[33mExporting with FP16 precision\033[0m")
    else:
        dummy_input = torch.randn(batch_size, 1, raw_height, raw_width)

    # Dynamic axes: batch size only (output spatial dims are fixed by encoder/decoder)
    dynamic_axes = config["dynamic_axes"]
    if dynamic_axes is not None:
        dynamic_axes = {"input": {0: "batch_size"}, output_names[0]: {0: "batch_size"}}

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _onnx_export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=config["opset_version"],
            do_constant_folding=config["do_constant_folding"],
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        print(f"\033[32mONNX model (with preprocess) exported to {output_path}\033[0m")
    except Exception as e:
        print(f"\033[31mFailed to export ONNX model: {e}\033[0m")
        raise

    # Verify exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("\033[32mONNX model verification passed\033[0m")
        print(f"\033[34mModel info:\033[0m")
        print(f"  - Opset version: {config['opset_version']}")
        print(f"  - Raw input shape: [{batch_size}, 1, {raw_height}, {raw_width}]")
        print(f"  - Encoder input shape (after resize): [{batch_size}, 1, {encoder_height}, {encoder_width}]")
        print(f"  - Dynamic batch: {dynamic_axes is not None}")
        print(f"  - Outputs: {output_names}")
    except ImportError:
        print("\033[33mWarning: onnx package not installed, skipping verification\033[0m")
    except Exception as e:
        print(f"\033[33mWarning: ONNX verification failed: {e}\033[0m")

    print(f"\n\033[34mDeployment hints for {platform}:\033[0m")
    if platform == "jetson":
        print("  - Use trtexec to convert to TensorRT engine:")
        print(f"    trtexec --onnx={output_path} --saveEngine=model.engine --fp16")
    elif platform == "nuc":
        print("  - For OpenVINO, use Model Optimizer:")
        print(f"    mo --input_model {output_path} --output_dir openvino_model/")
        print("  - Or use ONNX Runtime directly for inference")
    else:
        print("  - Compatible with most ONNX runtimes")


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
    parser.add_argument(
        "--static",
        action="store_true",
        help=(
            "Disable dynamic axes so all input/output shapes are fixed. "
            "Required for TensorRT conversion (overrides platform defaults)."
        ),
    )
    parser.add_argument(
        "--with_preprocess",
        action="store_true",
        help=(
            "Export a model that includes depth preprocessing (NaN/Inf handling, "
            "range masking, resize) fused into the ONNX graph. "
            "Takes raw camera input instead of pre-processed depth. "
            "Combine with --deploy for encoder-only output (mu), or omit for full VAE (depth)."
        ),
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.25,
        help="Minimum valid depth in metres for preprocessing (default: 0.25 m, ZED X)",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=10.0,
        help="Maximum valid depth in metres for preprocessing (default: 10.0 m, ZED X)",
    )
    parser.add_argument(
        "--raw_height",
        type=int,
        default=600,
        help="Raw camera input height for --with_preprocess (default: 600, ZED X SVGA)",
    )
    parser.add_argument(
        "--raw_width",
        type=int,
        default=960,
        help="Raw camera input width for --with_preprocess (default: 960, ZED X SVGA)",
    )

    args = parser.parse_args()

    # Auto-generate output path if not specified
    if args.output_path is None:
        model_name = Path(args.model_path).stem
        config = get_platform_config(args.platform)
        suffix = config["suffix"]
        deploy_suffix = "_deploy" if args.deploy else ""
        fp16_suffix = "_fp16" if args.fp16 else ""
        preprocess_suffix = "_preprocess" if args.with_preprocess else ""
        static_suffix = "_static" if args.static else ""
        args.output_path = (
            f"output/{model_name}{suffix}{deploy_suffix}"
            f"{preprocess_suffix}{static_suffix}{fp16_suffix}.onnx"
        )

    if args.with_preprocess:
        export_onnx_with_preprocess(
            model_path=args.model_path,
            output_path=args.output_path,
            platform=args.platform,
            latent_dim=args.latent_dim,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            encoder_height=args.input_height,
            encoder_width=args.input_width,
            raw_height=args.raw_height,
            raw_width=args.raw_width,
            batch_size=args.batch_size,
            deploy=args.deploy,
            fp16=args.fp16,
            static=args.static,
        )
    else:
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
            static=args.static,
        )


if __name__ == "__main__":
    main()
