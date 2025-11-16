"""
Test Chessformer on Modal with real preprocessed data.

Usage:
    modal run test_chessformer_modal.py
"""
import modal

app = modal.App("test-chessformer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy", "chess")
    .add_local_dir("src/models", remote_path="/root/models")
)

data_volume = modal.Volume.from_name("chess-training-data", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": data_volume},
    gpu="T4",  # Small GPU for testing
    timeout=600,
)
def test_chessformer():
    """Test Chessformer model with real data"""
    import torch
    import sys
    import numpy as np
    from pathlib import Path

    sys.path.insert(0, "/root")

    from models.chessformer import LeelaZeroTransformer

    print("=" * 60)
    print("🧪 TESTING CHESSFORMER MODEL ON MODAL")
    print("=" * 60)

    # Load real preprocessed data
    data_dir = Path("/data/lc0_processed")
    npz_files = list(data_dir.glob("*.npz"))

    if not npz_files:
        print("❌ No preprocessed data found!")
        print(f"Contents of {data_dir}:")
        for item in data_dir.iterdir():
            print(f"  - {item}")
        return {"error": "No data found"}

    print(f"\n📂 Found {len(npz_files)} preprocessed files")
    print(f"Using: {npz_files[0].name}")

    # Load data
    data = np.load(npz_files[0])
    inputs = torch.from_numpy(data['inputs'][:8])  # 8 positions

    print(f"✅ Loaded data")
    print(f"   Shape: {inputs.shape}")
    print(f"   Expected: [8, 112, 8, 8]")

    # Test configurations
    configs = [
        {"name": "small", "filters": 128, "depth": 6, "heads": 8, "dropout": 0.15},
        {"name": "medium", "filters": 192, "depth": 8, "heads": 12, "dropout": 0.12},
    ]

    results = {}

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Testing {cfg['name']} config:")
        print(f"  Filters: {cfg['filters']}")
        print(f"  Depth: {cfg['depth']}")
        print(f"  Heads: {cfg['heads']}")
        print(f"  Dropout: {cfg['dropout']}")

        try:
            # Create model
            print(f"\n🏗️  Creating model...")
            model = LeelaZeroTransformer(
                num_filters=cfg['filters'],
                num_residual_blocks=cfg['depth'],
                heads=cfg['heads'],
                dropout=cfg['dropout'],
            ).cuda()

            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ Model created")
            print(f"   Parameters: {total_params:,}")
            print(f"   Size: ~{total_params * 4 / 1e6:.1f} MB")

            # Forward pass
            print(f"\n🚀 Running forward pass...")
            model.eval()
            with torch.no_grad():
                output = model(inputs.cuda())

            print(f"✅ Forward pass successful")
            print(f"   Policy: {output.policy.shape} (expected [8, 1858])")
            print(f"   Value: {output.value.shape} (expected [8, 3])")
            print(f"   Moves left: {output.moves_left.shape} (expected [8, 1])")

            # Check shapes
            shapes_ok = (
                output.policy.shape == (8, 1858) and
                output.value.shape == (8, 3) and
                output.moves_left.shape == (8, 1)
            )

            if not shapes_ok:
                print(f"❌ Shape mismatch!")
                results[cfg['name']] = {"success": False, "error": "Shape mismatch"}
                continue

            # Check for NaN/Inf
            has_nan = (
                torch.isnan(output.policy).any() or
                torch.isnan(output.value).any() or
                torch.isnan(output.moves_left).any()
            )
            has_inf = (
                torch.isinf(output.policy).any() or
                torch.isinf(output.value).any() or
                torch.isinf(output.moves_left).any()
            )

            if has_nan or has_inf:
                print(f"❌ NaN/Inf detected!")
                results[cfg['name']] = {"success": False, "error": "NaN/Inf"}
                continue

            print(f"✅ No NaN/Inf")

            # Check value ranges
            policy_range = (output.policy.min().item(), output.policy.max().item())
            value_range = (output.value.min().item(), output.value.max().item())
            ml_range = (output.moves_left.min().item(), output.moves_left.max().item())

            print(f"\n📊 Output ranges:")
            print(f"   Policy: [{policy_range[0]:.3f}, {policy_range[1]:.3f}]")
            print(f"   Value: [{value_range[0]:.3f}, {value_range[1]:.3f}]")
            print(f"   Moves left: [{ml_range[0]:.3f}, {ml_range[1]:.3f}]")

            # Benchmark inference speed
            print(f"\n⏱️  Benchmarking inference...")
            import time
            torch.cuda.synchronize()
            start = time.time()

            for _ in range(100):
                with torch.no_grad():
                    _ = model(inputs.cuda())

            torch.cuda.synchronize()
            elapsed = time.time() - start
            avg_time = elapsed / 100 * 1000  # ms

            print(f"✅ Inference: {avg_time:.1f} ms/batch (8 positions)")
            print(f"   Per position: {avg_time/8:.1f} ms")

            results[cfg['name']] = {
                "success": True,
                "parameters": total_params,
                "inference_ms": avg_time,
                "inference_per_pos_ms": avg_time / 8,
            }

            print(f"\n✅ {cfg['name'].upper()} CONFIG PASSED!")

        except Exception as e:
            print(f"\n❌ {cfg['name'].upper()} CONFIG FAILED!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results[cfg['name']] = {"success": False, "error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")

    for name, result in results.items():
        if result.get("success"):
            print(f"\n✅ {name.upper()}: PASSED")
            print(f"   Parameters: {result['parameters']:,}")
            print(f"   Inference: {result['inference_per_pos_ms']:.1f} ms/position")
        else:
            print(f"\n❌ {name.upper()}: FAILED")
            print(f"   Error: {result.get('error', 'Unknown')}")

    all_passed = all(r.get("success", False) for r in results.values())

    if all_passed:
        print(f"\n{'='*60}")
        print(f"🎉 ALL TESTS PASSED - CHESSFORMER READY FOR TRAINING!")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"❌ SOME TESTS FAILED - FIX BEFORE TRAINING")
        print(f"{'='*60}")

    return results


@app.local_entrypoint()
def main():
    """Run Chessformer tests"""
    print("🚀 Launching Chessformer tests on Modal...")
    result = test_chessformer.remote()

    print(f"\n📋 Final results:")
    for name, res in result.items():
        status = "✅ PASS" if res.get("success") else "❌ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r.get("success", False) for r in result.values())
    return 0 if all_passed else 1
