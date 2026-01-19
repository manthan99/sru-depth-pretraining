# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

"""
TorchScript Export Script for VAE Depth Estimation Model

Usage:
    python convert_jit.py --model_path model_save/vae_pretrain.pth
    python convert_jit.py --model_path model_save/vae_pretrain.pth --deploy
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from network import VAENet


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


def compile_vae_model(
    model_path: str,
    output_path: str,
    latent_dim: int = 64,
    deploy: bool = False,
) -> None:
    """Compile a VAE model to TorchScript format for deployment.

    Args:
        model_path: Path to the saved model weights (.pth file)
        output_path: Path to save the compiled model (.pt file)
        latent_dim: Latent dimension of the VAE model (default: 64)
        deploy: If True, export only encoder with mu output for robot deployment
    """
    # Load the VAE model
    vae_model = VAENet(latent_dim)

    # Load the model weights
    try:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        vae_model.load_state_dict(state_dict, strict=True)
        print(f"\033[32mLoaded model weights from {model_path}\033[0m")
    except Exception as e:
        print(f"\033[31mFailed to load model weights: {e}\033[0m")
        raise

    # Select model variant
    if deploy:
        model = VAEEncoderDeploy(vae_model)
        print("\033[34mDeploy mode: exporting encoder only (output: mu)\033[0m")
    else:
        model = vae_model
        print("\033[34mFull model mode: exporting complete VAE\033[0m")

    model.eval()

    # Compile the model to TorchScript
    try:
        compiled_model = torch.jit.script(model)
        print("\033[32mModel compiled successfully\033[0m")
    except Exception as e:
        print(f"\033[31mFailed to compile model: {e}\033[0m")
        raise

    # Save the compiled model
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled_model.save(output_path)
    print(f"\033[32mCompiled model saved to {output_path}\033[0m")


def main():
    parser = argparse.ArgumentParser(
        description="Export VAE depth model to TorchScript format"
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
        help="Output path for JIT model (auto-generated if not specified)",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="VAE latent dimension",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Export only encoder with mu output for robot deployment",
    )

    args = parser.parse_args()

    # Auto-generate output path if not specified
    if args.output_path is None:
        model_name = Path(args.model_path).stem
        deploy_suffix = "_deploy" if args.deploy else ""
        args.output_path = f"output/{model_name}{deploy_suffix}_jit.pt"

    compile_vae_model(
        model_path=args.model_path,
        output_path=args.output_path,
        latent_dim=args.latent_dim,
        deploy=args.deploy,
    )


if __name__ == "__main__":
    main()