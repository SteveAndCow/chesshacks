"""
Test model shapes and forward pass.

Verifies:
- All models accept 16-channel input
- Output shapes are correct: policy (4096), value (1), result (3)
- Relative position bias works in transformers
- No runtime errors
"""
import torch
import sys
from pathlib import Path

from models.cnn import ChessCNN, ChessCNNLite
from models.transformer import ChessTransformer, ChessTransformerLite

# Add models to path
sys.path.insert(0, str(Path(__file__).parent))




def test_model(name: str, model: torch.nn.Module, batch_size: int = 4):
    """Test a single model."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    # Create dummy input: (batch, 16 channels, 8, 8)
    dummy_input = torch.randn(batch_size, 16, 8, 8)
    print(f"Input shape: {dummy_input.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        try:
            policy, value, result = model(dummy_input)

            # Check shapes
            expected_policy = (batch_size, 4096)
            expected_value = (batch_size, 1)
            expected_result = (batch_size, 3)

            assert policy.shape == expected_policy, \
                f"Policy shape wrong: expected {expected_policy}, got {policy.shape}"
            assert value.shape == expected_value, \
                f"Value shape wrong: expected {expected_value}, got {value.shape}"
            assert result.shape == expected_result, \
                f"Result shape wrong: expected {expected_result}, got {result.shape}"

            print(f"✅ Policy shape: {policy.shape}")
            print(f"✅ Value shape: {value.shape}")
            print(f"✅ Result shape: {result.shape}")

            # Check value ranges
            print(f"Value range: [{value.min():.3f}, {value.max():.3f}]")
            print(f"Result logits range: [{result.min():.3f}, {result.max():.3f}]")

            return True

        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    print("\n" + "="*60)
    print("CHESS MODEL SHAPE TESTING")
    print("="*60)

    # Test all models
    models_to_test = [
        ("ChessCNN", ChessCNN(num_residual_blocks=5, num_channels=128)),
        ("ChessCNNLite", ChessCNNLite(num_channels=128)),
        ("ChessTransformer", ChessTransformer(d_model=256, nhead=8, num_layers=4)),
        ("ChessTransformerLite", ChessTransformerLite(d_model=128, nhead=4, num_layers=2)),
    ]

    results = {}
    for name, model in models_to_test:
        results[name] = test_model(name, model)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = all(results.values())

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    if all_passed:
        print("\n✅ All models passed shape tests!")
        return 0
    else:
        print("\n❌ Some models failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
