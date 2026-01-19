# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

"""
Test script to verify correctness of exported models (JIT and ONNX).

Compares outputs from exported models against the original PyTorch model
to ensure numerical consistency within acceptable tolerances.

Usage:
    python test_export.py --model_path model_save/vae_pretrain_new.pth
    python test_export.py --model_path model_save/vae_pretrain_new.pth --skip_onnx
    python test_export.py --model_path model_save/vae_pretrain_new.pth --skip_jit
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

from network import VAENet


def load_pytorch_model(model_path: str, latent_dim: int = 64) -> VAENet:
    """Load the original PyTorch model."""
    model = VAENet(latent_dim)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def test_jit_export(
    model_path: str,
    latent_dim: int = 64,
    input_shape: tuple = (1, 1, 40, 64),
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> bool:
    """Test JIT export correctness.

    Args:
        model_path: Path to model weights
        latent_dim: VAE latent dimension
        input_shape: Input tensor shape (B, C, H, W)
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if test passes, False otherwise
    """
    print("\n" + "=" * 60)
    print("Testing JIT Export")
    print("=" * 60)

    # Load original model
    print("Loading PyTorch model...")
    pytorch_model = load_pytorch_model(model_path, latent_dim)

    # Create test input with fixed seed for reproducibility
    torch.manual_seed(42)
    test_input = torch.randn(*input_shape)

    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
        pytorch_depth = pytorch_output["depth"]
        pytorch_mu, pytorch_logvar = pytorch_output["mu"], pytorch_output["logvar"]

    # Export to JIT
    print("Exporting to TorchScript...")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        jit_path = f.name

    try:
        jit_model = torch.jit.script(pytorch_model)
        jit_model.save(jit_path)

        # Load JIT model
        print("Loading JIT model...")
        loaded_jit_model = torch.jit.load(jit_path)
        loaded_jit_model.eval()

        # Get JIT output
        with torch.no_grad():
            jit_output = loaded_jit_model(test_input)
            jit_depth = jit_output["depth"]
            jit_mu, jit_logvar = jit_output["mu"], jit_output["logvar"]

        # Compare outputs
        print("\nComparing outputs...")

        depth_match = torch.allclose(pytorch_depth, jit_depth, rtol=rtol, atol=atol)
        mu_match = torch.allclose(pytorch_mu, jit_mu, rtol=rtol, atol=atol)
        logvar_match = torch.allclose(pytorch_logvar, jit_logvar, rtol=rtol, atol=atol)

        depth_max_diff = (pytorch_depth - jit_depth).abs().max().item()
        mu_max_diff = (pytorch_mu - jit_mu).abs().max().item()
        logvar_max_diff = (pytorch_logvar - jit_logvar).abs().max().item()

        print(f"  Depth:  {'PASS' if depth_match else 'FAIL'} (max diff: {depth_max_diff:.2e})")
        print(f"  Mu:     {'PASS' if mu_match else 'FAIL'} (max diff: {mu_max_diff:.2e})")
        print(f"  Logvar: {'PASS' if logvar_match else 'FAIL'} (max diff: {logvar_max_diff:.2e})")

        all_pass = depth_match and mu_match and logvar_match

        if all_pass:
            print("\n\033[32m✓ JIT Export Test PASSED\033[0m")
        else:
            print("\n\033[31m✗ JIT Export Test FAILED\033[0m")

        return all_pass

    except Exception as e:
        print(f"\n\033[31m✗ JIT Export Test FAILED with error: {e}\033[0m")
        return False

    finally:
        # Cleanup
        Path(jit_path).unlink(missing_ok=True)


def test_onnx_export(
    model_path: str,
    latent_dim: int = 64,
    input_shape: tuple = (1, 1, 40, 64),
    rtol: float = 1e-4,
    atol: float = 1e-5,
    platforms: list = None,
) -> bool:
    """Test ONNX export correctness.

    Args:
        model_path: Path to model weights
        latent_dim: VAE latent dimension
        input_shape: Input tensor shape (B, C, H, W)
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        platforms: List of platforms to test

    Returns:
        True if all tests pass, False otherwise
    """
    print("\n" + "=" * 60)
    print("Testing ONNX Export")
    print("=" * 60)

    try:
        import onnx
        import onnxruntime as ort
    except ImportError as e:
        print(f"\033[33mSkipping ONNX test: {e}\033[0m")
        print("Install with: pip install onnx onnxruntime")
        return True  # Don't fail if ONNX not installed

    if platforms is None:
        platforms = ["generic", "jetson", "nuc"]

    # Load original model
    print("Loading PyTorch model...")
    pytorch_model = load_pytorch_model(model_path, latent_dim)

    # Create test input with fixed seed
    torch.manual_seed(42)
    test_input = torch.randn(*input_shape)

    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
        pytorch_depth = pytorch_output["depth"].numpy()

    all_pass = True

    for platform in platforms:
        print(f"\n--- Testing platform: {platform} ---")

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        try:
            # Export to ONNX
            print(f"Exporting to ONNX ({platform})...")
            from convert_onnx import export_onnx

            export_onnx(
                model_path=model_path,
                output_path=onnx_path,
                platform=platform,
                latent_dim=latent_dim,
                input_height=input_shape[2],
                input_width=input_shape[3],
                batch_size=input_shape[0],
                include_latent=False,
                fp16=False,
            )

            # Load and verify ONNX model
            print("Loading ONNX model...")
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # Run inference with ONNX Runtime
            print("Running ONNX Runtime inference...")
            ort_session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )

            ort_inputs = {"input": test_input.numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)
            onnx_depth = ort_outputs[0]

            # Compare outputs
            depth_match = np.allclose(pytorch_depth, onnx_depth, rtol=rtol, atol=atol)
            depth_max_diff = np.abs(pytorch_depth - onnx_depth).max()

            print(f"  Depth: {'PASS' if depth_match else 'FAIL'} (max diff: {depth_max_diff:.2e})")

            if depth_match:
                print(f"\033[32m✓ ONNX Export Test ({platform}) PASSED\033[0m")
            else:
                print(f"\033[31m✗ ONNX Export Test ({platform}) FAILED\033[0m")
                all_pass = False

        except Exception as e:
            print(f"\033[31m✗ ONNX Export Test ({platform}) FAILED with error: {e}\033[0m")
            all_pass = False

        finally:
            Path(onnx_path).unlink(missing_ok=True)

    # Test with latent outputs
    print(f"\n--- Testing ONNX with latent outputs ---")

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name

    try:
        from convert_onnx import export_onnx

        export_onnx(
            model_path=model_path,
            output_path=onnx_path,
            platform="generic",
            latent_dim=latent_dim,
            input_height=input_shape[2],
            input_width=input_shape[3],
            batch_size=input_shape[0],
            include_latent=True,
            fp16=False,
        )

        # Get PyTorch latent outputs
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)
            pytorch_mu = pytorch_output["mu"].numpy()
            pytorch_logvar = pytorch_output["logvar"].numpy()

        # Run ONNX inference
        ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_inputs = {"input": test_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)

        onnx_depth = ort_outputs[0]
        onnx_mu = ort_outputs[1]
        onnx_logvar = ort_outputs[2]

        # Compare all outputs
        depth_match = np.allclose(pytorch_depth, onnx_depth, rtol=rtol, atol=atol)
        mu_match = np.allclose(pytorch_mu, onnx_mu, rtol=rtol, atol=atol)
        logvar_match = np.allclose(pytorch_logvar, onnx_logvar, rtol=rtol, atol=atol)

        print(f"  Depth:  {'PASS' if depth_match else 'FAIL'} (max diff: {np.abs(pytorch_depth - onnx_depth).max():.2e})")
        print(f"  Mu:     {'PASS' if mu_match else 'FAIL'} (max diff: {np.abs(pytorch_mu - onnx_mu).max():.2e})")
        print(f"  Logvar: {'PASS' if logvar_match else 'FAIL'} (max diff: {np.abs(pytorch_logvar - onnx_logvar).max():.2e})")

        latent_pass = depth_match and mu_match and logvar_match

        if latent_pass:
            print(f"\033[32m✓ ONNX Export with Latent Test PASSED\033[0m")
        else:
            print(f"\033[31m✗ ONNX Export with Latent Test FAILED\033[0m")
            all_pass = False

    except Exception as e:
        print(f"\033[31m✗ ONNX Export with Latent Test FAILED with error: {e}\033[0m")
        all_pass = False

    finally:
        Path(onnx_path).unlink(missing_ok=True)

    return all_pass


def test_batch_consistency(
    model_path: str,
    latent_dim: int = 64,
    input_hw: tuple = (40, 64),
    batch_sizes: list = None,
) -> bool:
    """Test that different batch sizes produce consistent per-sample outputs.

    Args:
        model_path: Path to model weights
        latent_dim: VAE latent dimension
        input_hw: Input height and width
        batch_sizes: List of batch sizes to test

    Returns:
        True if test passes, False otherwise
    """
    print("\n" + "=" * 60)
    print("Testing Batch Consistency")
    print("=" * 60)

    if batch_sizes is None:
        batch_sizes = [1, 2, 4]

    # Load model
    print("Loading PyTorch model...")
    model = load_pytorch_model(model_path, latent_dim)

    # Create single test input
    torch.manual_seed(42)
    single_input = torch.randn(1, 1, *input_hw)

    # Get reference output
    with torch.no_grad():
        ref_output = model(single_input)
        ref_depth = ref_output["depth"]

    all_pass = True

    for batch_size in batch_sizes:
        if batch_size == 1:
            continue

        # Create batched input by repeating single input
        batched_input = single_input.repeat(batch_size, 1, 1, 1)

        with torch.no_grad():
            batched_output = model(batched_input)
            batched_depth = batched_output["depth"]

        # Check each sample matches reference
        matches = []
        for i in range(batch_size):
            match = torch.allclose(ref_depth, batched_depth[i : i + 1], rtol=1e-5, atol=1e-6)
            matches.append(match)

        batch_pass = all(matches)
        print(f"  Batch size {batch_size}: {'PASS' if batch_pass else 'FAIL'}")

        if not batch_pass:
            all_pass = False

    if all_pass:
        print("\n\033[32m✓ Batch Consistency Test PASSED\033[0m")
    else:
        print("\n\033[31m✗ Batch Consistency Test FAILED\033[0m")

    return all_pass


def test_determinism(
    model_path: str,
    latent_dim: int = 64,
    input_shape: tuple = (1, 1, 40, 64),
    num_runs: int = 5,
) -> bool:
    """Test that model produces deterministic outputs in eval mode.

    Args:
        model_path: Path to model weights
        latent_dim: VAE latent dimension
        input_shape: Input tensor shape
        num_runs: Number of runs to compare

    Returns:
        True if test passes, False otherwise
    """
    print("\n" + "=" * 60)
    print("Testing Determinism (Eval Mode)")
    print("=" * 60)

    # Load model
    model = load_pytorch_model(model_path, latent_dim)

    # Create test input
    torch.manual_seed(42)
    test_input = torch.randn(*input_shape)

    outputs = []
    for i in range(num_runs):
        with torch.no_grad():
            output = model(test_input)
            outputs.append(output["depth"].clone())

    # Compare all outputs to first
    all_match = True
    for i in range(1, num_runs):
        match = torch.allclose(outputs[0], outputs[i], rtol=0, atol=0)
        if not match:
            all_match = False
            diff = (outputs[0] - outputs[i]).abs().max().item()
            print(f"  Run {i} differs from run 0 by {diff:.2e}")

    if all_match:
        print(f"  All {num_runs} runs produced identical outputs")
        print("\n\033[32m✓ Determinism Test PASSED\033[0m")
    else:
        print("\n\033[31m✗ Determinism Test FAILED\033[0m")
        print("  Note: VAE sampling is stochastic in train mode, ensure model.eval() is called")

    return all_match


def test_jit_deploy(
    model_path: str,
    latent_dim: int = 64,
    input_shape: tuple = (1, 1, 40, 64),
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> bool:
    """Test JIT deploy mode (encoder only, mu output).

    Args:
        model_path: Path to model weights
        latent_dim: VAE latent dimension
        input_shape: Input tensor shape (B, C, H, W)
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if test passes, False otherwise
    """
    print("\n" + "=" * 60)
    print("Testing JIT Deploy Mode (encoder only)")
    print("=" * 60)

    from convert_jit import VAEEncoderDeploy

    # Load original model
    print("Loading PyTorch model...")
    pytorch_model = load_pytorch_model(model_path, latent_dim)

    # Create deploy wrapper
    deploy_model = VAEEncoderDeploy(pytorch_model)
    deploy_model.eval()

    # Create test input
    torch.manual_seed(42)
    test_input = torch.randn(*input_shape)

    # Get reference mu from full model
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
        pytorch_mu = pytorch_output["mu"]

    # Get mu from deploy wrapper
    with torch.no_grad():
        deploy_mu = deploy_model(test_input)

    # Export to JIT
    print("Exporting deploy model to TorchScript...")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        jit_path = f.name

    try:
        jit_model = torch.jit.script(deploy_model)
        jit_model.save(jit_path)

        # Load JIT model
        print("Loading JIT deploy model...")
        loaded_jit_model = torch.jit.load(jit_path)
        loaded_jit_model.eval()

        # Get JIT output
        with torch.no_grad():
            jit_mu = loaded_jit_model(test_input)

        # Compare outputs
        print("\nComparing outputs...")

        wrapper_match = torch.allclose(pytorch_mu, deploy_mu, rtol=rtol, atol=atol)
        jit_match = torch.allclose(pytorch_mu, jit_mu, rtol=rtol, atol=atol)

        wrapper_diff = (pytorch_mu - deploy_mu).abs().max().item()
        jit_diff = (pytorch_mu - jit_mu).abs().max().item()

        print(f"  Wrapper vs PyTorch: {'PASS' if wrapper_match else 'FAIL'} (max diff: {wrapper_diff:.2e})")
        print(f"  JIT vs PyTorch:     {'PASS' if jit_match else 'FAIL'} (max diff: {jit_diff:.2e})")

        all_pass = wrapper_match and jit_match

        if all_pass:
            print("\n\033[32m✓ JIT Deploy Test PASSED\033[0m")
        else:
            print("\n\033[31m✗ JIT Deploy Test FAILED\033[0m")

        return all_pass

    except Exception as e:
        print(f"\n\033[31m✗ JIT Deploy Test FAILED with error: {e}\033[0m")
        return False

    finally:
        Path(jit_path).unlink(missing_ok=True)


def test_onnx_deploy(
    model_path: str,
    latent_dim: int = 64,
    input_shape: tuple = (1, 1, 40, 64),
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """Test ONNX deploy mode (encoder only, mu output).

    Args:
        model_path: Path to model weights
        latent_dim: VAE latent dimension
        input_shape: Input tensor shape (B, C, H, W)
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if test passes, False otherwise
    """
    print("\n" + "=" * 60)
    print("Testing ONNX Deploy Mode (encoder only)")
    print("=" * 60)

    try:
        import onnxruntime as ort
    except ImportError as e:
        print(f"\033[33mSkipping ONNX deploy test: {e}\033[0m")
        return True

    from convert_onnx import export_onnx

    # Load original model
    print("Loading PyTorch model...")
    pytorch_model = load_pytorch_model(model_path, latent_dim)

    # Create test input
    torch.manual_seed(42)
    test_input = torch.randn(*input_shape)

    # Get reference mu from full model
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
        pytorch_mu = pytorch_output["mu"].numpy()

    all_pass = True

    for platform in ["generic", "jetson", "nuc"]:
        print(f"\n--- Testing deploy mode for platform: {platform} ---")

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        try:
            # Export with deploy mode
            export_onnx(
                model_path=model_path,
                output_path=onnx_path,
                platform=platform,
                latent_dim=latent_dim,
                input_height=input_shape[2],
                input_width=input_shape[3],
                batch_size=input_shape[0],
                deploy=True,
                fp16=False,
            )

            # Load and run ONNX model
            ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            ort_inputs = {"input": test_input.numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)
            onnx_mu = ort_outputs[0]

            # Compare
            mu_match = np.allclose(pytorch_mu, onnx_mu, rtol=rtol, atol=atol)
            mu_diff = np.abs(pytorch_mu - onnx_mu).max()

            print(f"  Mu: {'PASS' if mu_match else 'FAIL'} (max diff: {mu_diff:.2e})")

            if mu_match:
                print(f"\033[32m✓ ONNX Deploy Test ({platform}) PASSED\033[0m")
            else:
                print(f"\033[31m✗ ONNX Deploy Test ({platform}) FAILED\033[0m")
                all_pass = False

        except Exception as e:
            print(f"\033[31m✗ ONNX Deploy Test ({platform}) FAILED with error: {e}\033[0m")
            all_pass = False

        finally:
            Path(onnx_path).unlink(missing_ok=True)

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Test exported model correctness")
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_save/vae_pretrain_new.pth",
        help="Path to model weights (.pth file)",
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
        "--skip_jit",
        action="store_true",
        help="Skip JIT export test",
    )
    parser.add_argument(
        "--skip_onnx",
        action="store_true",
        help="Skip ONNX export test",
    )
    parser.add_argument(
        "--skip_deploy",
        action="store_true",
        help="Skip deploy mode tests",
    )

    args = parser.parse_args()

    input_shape = (1, 1, args.input_height, args.input_width)

    print("\n" + "#" * 60)
    print("# Model Export Verification Tests")
    print("#" * 60)
    print(f"Model: {args.model_path}")
    print(f"Input shape: {input_shape}")
    print(f"Latent dim: {args.latent_dim}")

    results = {}

    # Run tests
    results["determinism"] = test_determinism(
        args.model_path, args.latent_dim, input_shape
    )

    results["batch_consistency"] = test_batch_consistency(
        args.model_path, args.latent_dim, (args.input_height, args.input_width)
    )

    if not args.skip_jit:
        results["jit"] = test_jit_export(
            args.model_path, args.latent_dim, input_shape
        )

    if not args.skip_onnx:
        results["onnx"] = test_onnx_export(
            args.model_path, args.latent_dim, input_shape
        )

    if not args.skip_deploy:
        if not args.skip_jit:
            results["jit_deploy"] = test_jit_deploy(
                args.model_path, args.latent_dim, input_shape
            )
        if not args.skip_onnx:
            results["onnx_deploy"] = test_onnx_deploy(
                args.model_path, args.latent_dim, input_shape
            )

    # Summary
    print("\n" + "#" * 60)
    print("# Test Summary")
    print("#" * 60)

    all_pass = True
    for test_name, passed in results.items():
        status = "\033[32mPASS\033[0m" if passed else "\033[31mFAIL\033[0m"
        print(f"  {test_name}: {status}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("\033[32m" + "=" * 60 + "\033[0m")
        print("\033[32m  All tests PASSED!\033[0m")
        print("\033[32m" + "=" * 60 + "\033[0m")
        return 0
    else:
        print("\033[31m" + "=" * 60 + "\033[0m")
        print("\033[31m  Some tests FAILED!\033[0m")
        print("\033[31m" + "=" * 60 + "\033[0m")
        return 1


if __name__ == "__main__":
    exit(main())
