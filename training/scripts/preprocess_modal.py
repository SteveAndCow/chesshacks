"""
Preprocess chess data on Modal with high RAM and parallelization.

Handles large PGN files efficiently with streaming and batch processing.
"""
import modal
from pathlib import Path

app = modal.App("chesshacks-preprocessing")

# High-RAM image for preprocessing
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "python-chess",
        "numpy",
        "tqdm",
    )
    .add_local_file(
        local_path="training/scripts/preprocess.py",
        remote_path="/root/preprocess.py"
    )
)

# Modal Volume for data storage
data_volume = modal.Volume.from_name("chess-training-data", create_if_missing=True)


@app.function(
    image=image,
    memory=65536,  # 64GB RAM - plenty for large files
    cpu=8.0,  # 8 CPUs for parallel processing
    timeout=3600 * 2,  # 2 hours max
    volumes={"/data": data_volume},
)
def preprocess_large_pgn():
    """
    Preprocess large PGN file with streaming and batching.

    Processes positions in batches to avoid memory issues.
    """
    import chess.pgn
    import numpy as np
    from tqdm import tqdm
    import sys

    # Add preprocess module to path
    sys.path.insert(0, "/root")
    from preprocess import board_to_tensor, move_to_index

    print("="*60)
    print("PREPROCESSING ON MODAL (64GB RAM, 8 CPUs)")
    print("="*60)

    # Input/output paths
    pgn_path = "/data/raw/filtered_games.pgn"
    output_dir = Path("/data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nInput: {pgn_path}")
    print(f"Output: {output_dir}")

    # Check if input exists
    if not Path(pgn_path).exists():
        print(f"❌ PGN file not found at {pgn_path}")
        print("Upload with: modal volume put chess-training-data data/raw/filtered_games.pgn /raw/filtered_games.pgn")
        return {"error": "PGN file not found"}

    # Get file size
    file_size = Path(pgn_path).stat().st_size / (1024**3)  # GB
    print(f"File size: {file_size:.2f} GB")

    # Process in streaming mode with batching
    print("\n📊 Processing positions...")

    batch_size = 10000  # Process 10k positions at a time
    boards_batch = []
    moves_batch = []
    values_batch = []

    total_positions = 0
    games_processed = 0

    with open(pgn_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            games_processed += 1
            if games_processed % 100 == 0:
                print(f"Games: {games_processed:,} | Positions: {total_positions:,}")

            # Get game result
            result = game.headers.get("Result", "*")
            if result == "1-0":
                value = 1.0
            elif result == "0-1":
                value = -1.0
            elif result == "1/2-1/2":
                value = 0.0
            else:
                continue  # Skip games without result

            # Process each position in the game
            board = game.board()
            for move in game.mainline_moves():
                # Convert board to tensor
                board_tensor = board_to_tensor(board)
                move_idx = move_to_index(move)

                boards_batch.append(board_tensor)
                moves_batch.append(move_idx)
                values_batch.append(value)

                total_positions += 1

                # Save batch when full
                if len(boards_batch) >= batch_size:
                    save_batch(
                        output_dir,
                        boards_batch,
                        moves_batch,
                        values_batch,
                        total_positions
                    )
                    boards_batch = []
                    moves_batch = []
                    values_batch = []

                board.push(move)

    # Save remaining positions
    if boards_batch:
        save_batch(
            output_dir,
            boards_batch,
            moves_batch,
            values_batch,
            total_positions
        )

    print(f"\n✅ Processing complete!")
    print(f"Total games: {games_processed:,}")
    print(f"Total positions: {total_positions:,}")

    # Concatenate all batch files
    print("\n📦 Combining batches...")
    combine_batches(output_dir)

    # Commit changes to volume
    data_volume.commit()

    print("\n✅ Data ready for training!")

    return {
        "total_games": games_processed,
        "total_positions": total_positions,
        "output_dir": str(output_dir),
    }


def save_batch(output_dir, boards, moves, values, position_count):
    """Save a batch of positions to disk."""
    import numpy as np

    batch_dir = output_dir / "batches"
    batch_dir.mkdir(exist_ok=True)

    batch_num = position_count // len(boards)

    np.save(batch_dir / f"boards_{batch_num}.npy", np.array(boards, dtype=np.float32))
    np.save(batch_dir / f"moves_{batch_num}.npy", np.array(moves, dtype=np.int64))
    np.save(batch_dir / f"values_{batch_num}.npy", np.array(values, dtype=np.float32))


def combine_batches(output_dir):
    """Combine all batch files into final arrays."""
    import numpy as np
    from glob import glob

    batch_dir = output_dir / "batches"

    # Load and concatenate all batches
    print("Loading board batches...")
    board_files = sorted(glob(str(batch_dir / "boards_*.npy")))
    boards = np.concatenate([np.load(f) for f in board_files])

    print("Loading move batches...")
    move_files = sorted(glob(str(batch_dir / "moves_*.npy")))
    moves = np.concatenate([np.load(f) for f in move_files])

    print("Loading value batches...")
    value_files = sorted(glob(str(batch_dir / "values_*.npy")))
    values = np.concatenate([np.load(f) for f in value_files])

    # Save final files
    print("Saving final files...")
    np.save(output_dir / "boards.npy", boards)
    np.save(output_dir / "moves.npy", moves)
    np.save(output_dir / "values.npy", values)

    print(f"Final shapes:")
    print(f"  Boards: {boards.shape}")
    print(f"  Moves: {moves.shape}")
    print(f"  Values: {values.shape}")

    # Clean up batch files
    print("Cleaning up batch files...")
    import shutil
    shutil.rmtree(batch_dir)


@app.local_entrypoint()
def main():
    """
    Run preprocessing on Modal.

    Usage:
        modal run scripts/preprocess_modal.py
    """
    print("🚀 Starting preprocessing on Modal (64GB RAM, 8 CPUs)...")
    print("This will take 30-60 minutes for 7.7M positions\n")

    result = preprocess_large_pgn.remote()

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Results: {result}")
    print("\nData is ready for training on Modal Volume!")
