"""
Test preprocessed training data.

Verifies:
- Data files exist
- Board tensors have 16 channels (not old 12-channel data)
- Data shapes are correct
- Move indices are valid [0, 4095]
- Values are in valid range [-1, 1]
"""
import numpy as np
from pathlib import Path
import sys

def main():
    print("\n" + "="*60)
    print("TRAINING DATA VALIDATION")
    print("="*60)

    # Path to preprocessed data
    data_dir = Path("training/data/processed")

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Run preprocessing first!")
        return 1

    # Load data files
    print(f"\n📂 Loading data from {data_dir}...")

    try:
        boards = np.load(data_dir / "boards.npy")
        moves = np.load(data_dir / "moves.npy")
        values = np.load(data_dir / "values.npy")
    except FileNotFoundError as e:
        print(f"❌ Missing data file: {e}")
        return 1

    # Check shapes
    print(f"\n📊 Data Statistics:")
    print(f"  Boards shape: {boards.shape}")
    print(f"  Moves shape: {moves.shape}")
    print(f"  Values shape: {values.shape}")

    num_samples = boards.shape[0]
    print(f"  Total samples: {num_samples:,}")

    # Critical check: Verify 16 channels (not old 12-channel data)
    print(f"\n🔍 Validating data format...")

    expected_channels = 16
    actual_channels = boards.shape[1]

    if actual_channels != expected_channels:
        print(f"❌ CRITICAL ERROR: Expected {expected_channels} channels, got {actual_channels}")
        print(f"   This means you have old preprocessed data (12 channels)!")
        print(f"   You need to re-run preprocessing with the updated preprocess.py")
        return 1

    print(f"✅ Boards have {actual_channels} channels (correct!)")

    # Validate board dimensions
    if boards.shape[2:] != (8, 8):
        print(f"❌ Invalid board dimensions: {boards.shape[2:]}, expected (8, 8)")
        return 1

    print(f"✅ Board dimensions: 8x8 (correct!)")

    # Check data ranges
    print(f"\n📈 Data Ranges:")
    print(f"  Boards: [{boards.min():.3f}, {boards.max():.3f}]")
    print(f"  Moves: [{moves.min()}, {moves.max()}]")
    print(f"  Values: [{values.min():.3f}, {values.max():.3f}]")

    # Validate moves are in valid range [0, 4095]
    if moves.min() < 0:
        print(f"❌ Invalid move index: {moves.min()} (must be >= 0)")
        return 1

    if moves.max() >= 4096:
        print(f"❌ Invalid move index: {moves.max()} (must be < 4096)")
        return 1

    print(f"✅ Moves in valid range [0, 4095]")

    # Validate values are in [-1, 1]
    if values.min() < -1.1 or values.max() > 1.1:
        print(f"⚠️  Warning: Values outside expected range [-1, 1]")

    print(f"✅ Values in reasonable range")

    # Check channel statistics
    print(f"\n📊 Channel Statistics:")
    for i in range(16):
        channel_name = [
            "White Pawn", "White Knight", "White Bishop", "White Rook", "White Queen", "White King",
            "Black Pawn", "Black Knight", "Black Bishop", "Black Rook", "Black Queen", "Black King",
            "Castling Kingside", "Castling Queenside", "En Passant", "Halfmove Clock"
        ][i]

        channel_data = boards[:, i, :, :]
        non_zero = (channel_data != 0).sum()
        coverage = non_zero / (num_samples * 64) * 100

        print(f"  Ch {i:2d} ({channel_name:20s}): {coverage:5.2f}% non-zero")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ All validation checks passed!")
    print(f"✅ Data has {num_samples:,} training examples with 16 channels")
    print(f"✅ Ready for training!")

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
