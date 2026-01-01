# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

import torch
from pathlib import Path
from network import VAENet


def compile_vae_model(model_path: str, output_path: str, latent_dim: int = 64) -> None:
    """Compile a VAE model to TorchScript format for deployment.

    Args:
        model_path: Path to the saved model weights (.pth file)
        output_path: Path to save the compiled model (.pt file)
        latent_dim: Latent dimension of the VAE model (default: 64)
    """
    # Load the VAE model
    vae_model = VAENet(latent_dim)
    vae_model.eval()

    # Load the model weights
    try:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        vae_model.load_state_dict(state_dict, strict=True)
        print(f"\033[32mLoaded model weights from {model_path}\033[0m")
    except Exception as e:
        print(f"\033[31mFailed to load model weights: {e}\033[0m")
        raise

    # Compile the model to TorchScript
    try:
        compiled_model = torch.jit.script(vae_model)
        print("\033[32mModel compiled successfully\033[0m")
    except Exception as e:
        print(f"\033[31mFailed to compile model: {e}\033[0m")
        raise

    # Save the compiled model
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled_model.save(output_path)
    print(f"\033[32mCompiled model saved to {output_path}\033[0m")


if __name__ == "__main__":
    model_name = "vae_pretrain_new"
    model_path = f"model_save/{model_name}.pth"
    compiled_model_path = f"output/{model_name}_jit.pt"

    compile_vae_model(model_path, compiled_model_path, latent_dim=64)