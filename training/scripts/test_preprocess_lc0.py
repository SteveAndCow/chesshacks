"""
Test the LC0 preprocessing pipeline locally.

Usage:
    python training/scripts/test_preprocess_lc0.py
"""
import sys
from pathlib import Path
import tempfile

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocess_pgn_to_lc0 import process_pgn_file
import numpy as np


def create_sample_pgn(pgn_path: Path):
    """Create a sample PGN file for testing."""
    pgn_content = """[Event "Test Game 1"]
[Site "Test"]
[Date "2024.01.01"]
[Round "1"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]
[WhiteElo "2500"]
[BlackElo "2400"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Na5 10. Bc2 c5 1-0

[Event "Test Game 2"]
[Site "Test"]
[Date "2024.01.01"]
[Round "2"]
[White "Player C"]
[Black "Player D"]
[Result "0-1"]
[WhiteElo "2300"]
[BlackElo "2350"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6 7. Bh4 b6 8. cxd5 exd5 9. Bd3 Bb7 10. O-O Nbd7 0-1

[Event "Test Game 3 - Low ELO"]
[Site "Test"]
[Date "2024.01.01"]
[Round "3"]
[White "Beginner"]
[Black "Novice"]
[Result "1/2-1/2"]
[WhiteElo "1200"]
[BlackElo "1300"]

1. e4 e5 2. Nf3 Nc6 1/2-1/2
"""
    with open(pgn_path, 'w') as f:
        f.write(pgn_content)
    print(f"Created sample PGN: {pgn_path}")


def test_preprocessing():
    """Test the preprocessing pipeline."""
    print("="*60)
    print("TESTING LC0 PREPROCESSING PIPELINE")
    print("="*60)

    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pgn_path = tmpdir / "test.pgn"
        output_path = tmpdir / "test_output.npz"

        # Create sample PGN
        create_sample_pgn(pgn_path)

        # Process it
        print("\nProcessing PGN file...")
        try:
            process_pgn_file(pgn_path, output_path, min_elo=2000)
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Load and verify output
        print("\nVerifying output...")
        if not output_path.exists():
            print(f"❌ Output file not created: {output_path}")
            return False

        data = np.load(output_path)

        print(f"\n📊 Output Statistics:")
        print(f"  Inputs shape: {data['inputs'].shape}")
        print(f"  Policies shape: {data['policies'].shape}")
        print(f"  Values shape: {data['values'].shape}")
        print(f"  Moves left shape: {data['moves_left'].shape}")

        # Expected: 2 games with ~20 positions total (filtered by ELO)
        num_positions = len(data['inputs'])
        print(f"\n  Total positions: {num_positions}")

        # Verify shapes
        inputs = data['inputs']
        policies = data['policies']
        values = data['values']
        moves_left = data['moves_left']

        errors = []

        if inputs.shape[1:] != (112, 8, 8):
            errors.append(f"Inputs shape wrong: {inputs.shape} (expected (N, 112, 8, 8))")

        if policies.shape[1] != 1858:
            errors.append(f"Policies shape wrong: {policies.shape} (expected (N, 1858))")

        if values.shape[1] != 3:
            errors.append(f"Values shape wrong: {values.shape} (expected (N, 3))")

        # Check data validity
        if not np.all((inputs >= 0) & (inputs <= 1)):
            # Some channels might be normalized differently
            print("⚠️  Warning: Some input values outside [0, 1] range")

        if not np.all((policies >= 0) & (policies <= 1)):
            errors.append("Policy values not in [0, 1] range")

        if not np.all(np.isclose(policies.sum(axis=1), 1.0)):
            errors.append("Policy doesn't sum to 1 (should be one-hot)")

        if not np.all((values >= 0) & (values <= 1)):
            errors.append("Value values not in [0, 1] range")

        if not np.all(np.isclose(values.sum(axis=1), 1.0)):
            errors.append("Values don't sum to 1 (should be WDL distribution)")

        # Print first position details
        print(f"\n🔍 First Position Details:")
        print(f"  Input channels sum: {inputs[0].sum():.1f}")
        print(f"  Policy target (move index): {np.argmax(policies[0])}")
        print(f"  Value target (WDL): {values[0]}")
        print(f"  Moves left: {moves_left[0]:.0f}")

        # Check for errors
        if errors:
            print(f"\n❌ VALIDATION ERRORS:")
            for error in errors:
                print(f"  - {error}")
            return False

        print("\n✅ ALL TESTS PASSED!")
        print(f"\nPreprocessing pipeline is working correctly.")
        print(f"Ready to process full dataset on Modal.")

        return True


if __name__ == "__main__":
    success = test_preprocessing()
    sys.exit(0 if success else 1)
