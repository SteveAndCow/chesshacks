"""
Parallel preprocessing on Modal for LC0 112-channel format.

This script processes PGN files in parallel across multiple Modal workers,
converting them to 112-channel .npz format for training.

Usage:
    # Upload PGN files first
    modal volume put chess-training-data data/pgn /pgn

    # Run preprocessing
    modal run training/scripts/preprocess_modal_lc0.py --min-elo 2000

    # Check results
    modal volume ls chess-training-data /lc0_processed
"""
import modal
from pathlib import Path

# Define Modal app
app = modal.App("chesshacks-preprocessing-lc0")

# Container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "chess",
        "numpy",
        "tqdm",
    )
    .add_local_dir(
        local_path="training/scripts/models",
        remote_path="/root/models"
    )
)

# Modal Volume for data storage
data_volume = modal.Volume.from_name("chess-training-data", create_if_missing=True)


@app.function(
    image=image,
    cpu=4,  # More CPU for faster preprocessing
    timeout=14400,  # 4 hours max per file (enough for large PGN files)
    volumes={"/data": data_volume},
)
def preprocess_pgn_file(
    pgn_filename: str,
    min_elo: int = 2000,
    positions_per_file: int = 50000
):
    """
    Preprocess a single PGN file to 112-channel format.

    Args:
        pgn_filename: Name of PGN file in /data/pgn/
        min_elo: Minimum ELO rating to include
        positions_per_file: Split into chunks of this size

    Returns:
        Dictionary with statistics
    """
    import chess
    import chess.pgn
    import numpy as np
    import sys
    from pathlib import Path
    import traceback

    # Wrap everything in try/except to catch errors
    try:
        def board_to_112_channels(board: chess.Board, history: list) -> np.ndarray:
            """Convert chess board + history to 112-channel LC0 format."""
            channels = []

            # 104 channels: 8 board positions × 13 planes each
            positions = history[-7:] + [board]

            # Pad with empty boards if we don't have 8 positions yet
            while len(positions) < 8:
                positions.insert(0, chess.Board(None))  # Empty board

            for pos in positions:
                # 12 piece planes (6 types × 2 colors)
                for piece_type in chess.PIECE_TYPES:
                    # Our pieces
                    plane = np.zeros(64, dtype=np.float32)
                    if pos.piece_map():  # Not empty board
                        for square in pos.pieces(piece_type, board.turn):
                            plane[square] = 1.0
                    channels.append(plane)

                    # Opponent pieces
                    plane = np.zeros(64, dtype=np.float32)
                    if pos.piece_map():
                        for square in pos.pieces(piece_type, not board.turn):
                            plane[square] = 1.0
                    channels.append(plane)

                # Repetition counter
                repetitions = sum(1 for p in positions if p == pos)
                channels.append(np.full(64, repetitions, dtype=np.float32))

            # 5 unit planes (castling rights + side to move)
            channels.append(np.full(64, float(board.has_kingside_castling_rights(board.turn))))
            channels.append(np.full(64, float(board.has_queenside_castling_rights(board.turn))))
            channels.append(np.full(64, float(board.has_kingside_castling_rights(not board.turn))))
            channels.append(np.full(64, float(board.has_queenside_castling_rights(not board.turn))))
            channels.append(np.full(64, 1.0 if board.turn == chess.WHITE else 0.0))

            # 1 rule50 plane
            channels.append(np.full(64, board.halfmove_clock / 99.0, dtype=np.float32))

            # 2 constant planes
            channels.append(np.zeros(64, dtype=np.float32))
            channels.append(np.ones(64, dtype=np.float32))

            return np.stack(channels).reshape(112, 8, 8)

        def move_to_policy_index(move: chess.Move, turn: chess.Color) -> int:
            """
            Convert move to LC0 policy index (0-1857).
            Uses proper LC0 policy mapping with board flipping for black.

            Args:
                move: The chess move
                turn: Current side to move (WHITE or BLACK)
            """
            # Import policy_index from models directory (copied to /root/models in container)
            import sys
            if '/root/models' not in sys.path:
                sys.path.insert(0, '/root/models')
            from policy_index import policy_index

            # LC0 always encodes from the current player's perspective
            # For black, we need to flip the board (mirror vertically and horizontally)
            if turn == chess.BLACK:
                # Flip the move coordinates
                from_square = chess.square_mirror(move.from_square)
                to_square = chess.square_mirror(move.to_square)
                flipped_move = chess.Move(from_square, to_square, move.promotion)
                move_str = flipped_move.uci()
            else:
                move_str = move.uci()

            # Handle promotions (LC0 only encodes queen, rook, bishop - knight is encoded as normal move)
            if move.promotion:
                if move.promotion == chess.KNIGHT:
                    # Knight promotions use base move notation without suffix
                    move_str = move_str[:4]
                else:
                    # Queen, Rook, Bishop promotions include suffix
                    promo_map = {
                        chess.QUEEN: 'q',
                        chess.ROOK: 'r',
                        chess.BISHOP: 'b'
                    }
                    move_str = move_str[:4] + promo_map[move.promotion]

            # Find the index in policy_index
            try:
                return policy_index.index(move_str)
            except ValueError:
                # Fallback for unexpected moves (shouldn't happen with legal moves)
                print(f"Warning: Move {move_str} (original: {move.uci()}, turn: {'BLACK' if turn == chess.BLACK else 'WHITE'}) not found in LC0 policy index")
                return 0

        def game_result_to_wdl(result: str, pov_color: chess.Color) -> np.ndarray:
            """Convert game result to WDL from POV of pov_color."""
            if result == "1-0":
                return np.array([0.0, 0.0, 1.0] if pov_color == chess.WHITE else [1.0, 0.0, 0.0], dtype=np.float32)
            elif result == "0-1":
                return np.array([0.0, 0.0, 1.0] if pov_color == chess.BLACK else [1.0, 0.0, 0.0], dtype=np.float32)
            else:
                return np.array([0.0, 1.0, 0.0], dtype=np.float32)

        print(f"Processing {pgn_filename}...")

        pgn_path = Path(f"/data/pgn/chunked/{pgn_filename}")

        # Check if file exists
        if not pgn_path.exists():
            print(f"❌ Error: File not found: {pgn_path}")
            print(f"Volume contents at /data/pgn/chunked/:")
            import os
            if Path("/data/pgn/chunked").exists():
                for item in os.listdir("/data/pgn/chunked")[:10]:
                    print(f"  - {item}")
            return {"error": f"File not found: {pgn_path}"}

        # Check file size
        file_size = pgn_path.stat().st_size / 1e6
        print(f"File size: {file_size:.2f} MB")

        inputs_list = []
        policies_list = []
        values_list = []
        moves_left_list = []

        games_processed = 0
        games_skipped = 0
        games_filtered = 0
        chunk_num = 0
        total_games_read = 0

        print(f"Opening file and starting to read games...")

        with open(pgn_path) as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                total_games_read += 1

                # Print progress every 1000 games
                if total_games_read % 1000 == 0:
                    print(f"Read {total_games_read} games, processed {games_processed}, filtered {games_filtered}, skipped {games_skipped}")

                # Filter by ELO
                try:
                    white_elo = int(game.headers.get("WhiteElo", 0))
                    black_elo = int(game.headers.get("BlackElo", 0))
                except ValueError:
                    games_skipped += 1
                    continue

                if white_elo < min_elo or black_elo < min_elo:
                    games_filtered += 1
                    continue

                result = game.headers.get("Result", "*")
                if result not in ["1-0", "0-1", "1/2-1/2"]:
                    games_skipped += 1
                    continue

                # Extract positions
                board = game.board()
                history = []
                moves = list(game.mainline_moves())

                if len(moves) < 10:  # Skip very short games
                    games_skipped += 1
                    continue

                for i, move in enumerate(moves):
                    try:
                        # Build 112-channel input
                        inputs = board_to_112_channels(board, history)
                        inputs_list.append(inputs)

                        # Policy target (proper LC0 1858 encoding)
                        policy = np.zeros(1858, dtype=np.float32)
                        move_idx = move_to_policy_index(move, board.turn)
                        policy[move_idx] = 1.0
                        policies_list.append(policy)

                        # Value target
                        values = game_result_to_wdl(result, board.turn)
                        values_list.append(values)

                        # Moves left
                        moves_left = len(moves) - i - 1
                        moves_left_list.append(moves_left)

                        # Update history
                        history.append(board.copy())
                        if len(history) > 7:
                            history.pop(0)

                        # Make move
                        board.push(move)

                    except Exception as e:
                        print(f"Error processing position: {e}")
                        continue

                games_processed += 1

                # Save chunk if we have enough positions
                if len(inputs_list) >= positions_per_file:
                    pgn_name = Path(pgn_filename).stem
                    output_path = Path(f"/data/lc0_processed_lichess/{pgn_name}_chunk{chunk_num:04d}.npz")
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    np.savez_compressed(
                        output_path,
                        inputs=np.array(inputs_list, dtype=np.float32),
                        policies=np.array(policies_list, dtype=np.float32),
                        values=np.array(values_list, dtype=np.float32),
                        moves_left=np.array(moves_left_list, dtype=np.float32)
                    )

                    print(f"Saved chunk {chunk_num} with {len(inputs_list)} positions")
                    chunk_num += 1

                    # Reset lists
                    inputs_list = []
                    policies_list = []
                    values_list = []
                    moves_left_list = []

        # Save final chunk
        if inputs_list:
            pgn_name = Path(pgn_filename).stem
            output_path = Path(f"/data/lc0_processed/{pgn_name}_chunk{chunk_num:04d}.npz")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                output_path,
                inputs=np.array(inputs_list, dtype=np.float32),
                policies=np.array(policies_list, dtype=np.float32),
                values=np.array(values_list, dtype=np.float32),
                moves_left=np.array(moves_left_list, dtype=np.float32)
            )
            print(f"Saved final chunk {chunk_num} with {len(inputs_list)} positions")

        # Print final summary
        print(f"\n{'='*60}")
        print(f"Preprocessing Summary for {pgn_filename}")
        print(f"{'='*60}")
        print(f"Total games read: {total_games_read}")
        print(f"Games filtered (low ELO): {games_filtered}")
        print(f"Games skipped (other): {games_skipped}")
        print(f"Games processed: {games_processed}")
        print(f"Chunks created: {chunk_num + (1 if inputs_list else 0)}")
        print(f"Total positions: {len(inputs_list) if inputs_list else 0}")
        print(f"{'='*60}\n")

        # Commit changes to volume
        data_volume.commit()
        print("Volume committed successfully")

        return {
            "pgn_file": pgn_filename,
            "total_games_read": total_games_read,
            "games_processed": games_processed,
            "games_filtered": games_filtered,
            "games_skipped": games_skipped,
            "chunks_created": chunk_num + (1 if inputs_list else 0),
            "total_positions": len(inputs_list) if inputs_list else 0
        }

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ ERROR PROCESSING {pgn_filename}")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        return {
            "pgn_file": pgn_filename,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.local_entrypoint()
def main(
    min_elo: int = 2000,
    positions_per_file: int = 50000,
    max_chunks: int = None,
    start_chunk: int = None,
    end_chunk: int = None,
):
    """
    Preprocess PGN files in parallel on Modal.

    Args:
        min_elo: Minimum ELO rating to include
        positions_per_file: Positions per output file
        max_chunks: Maximum number of PGN chunks to process (None = all)
        start_chunk: Start processing from this chunk number (e.g., 100 for chunk_0100.pgn)
        end_chunk: Stop processing at this chunk number (exclusive, e.g., 150 for chunk_0149.pgn)
    """
    import time

    print("="*60)
    print("PARALLEL PGN PREPROCESSING ON MODAL")
    print("="*60)
    print(f"Min ELO: {min_elo}")
    print(f"Positions per output file: {positions_per_file:,}")

    if start_chunk is not None or end_chunk is not None:
        print(f"Chunk range: {start_chunk or 0} to {end_chunk or 'end'}")
    else:
        print(f"Max chunks to process: {max_chunks if max_chunks else 'ALL'}")

    # List PGN files in volume
    print("\nListing PGN files in volume...")
    pgn_files = list_pgn_files.remote()

    if not pgn_files:
        print("❌ No PGN files found in /data/pgn/chunked/")
        print("\nUpload files with:")
        print("  modal volume put chess-training-data training/data/chunked/*.pgn /pgn/chunked/")
        return

    print(f"✅ Found {len(pgn_files)} total PGN files")

    # Sort files to ensure consistent ordering
    pgn_files = sorted(pgn_files)

    # Filter by chunk range if specified
    if start_chunk is not None or end_chunk is not None:
        filtered_files = []
        for f in pgn_files:
            # Extract chunk number from filename (e.g., filtered_games_chunk0100.pgn -> 100)
            import re
            match = re.search(r'chunk(\d+)', f)
            if match:
                chunk_num = int(match.group(1))
                if (start_chunk is None or chunk_num >= start_chunk) and \
                   (end_chunk is None or chunk_num < end_chunk):
                    filtered_files.append(f)
        pgn_files = filtered_files
        print(f"📋 Processing chunks {start_chunk or 0} to {end_chunk or 'end'}: {len(pgn_files)} files")
    # Otherwise use max_chunks if specified
    elif max_chunks and max_chunks < len(pgn_files):
        print(f"⚡ Processing first {max_chunks} chunks (out of {len(pgn_files)})")
        pgn_files = pgn_files[:max_chunks]
    else:
        print(f"📋 Processing all {len(pgn_files)} chunks")

    for i, f in enumerate(pgn_files[:10]):
        print(f"  {i+1}. {f}")
    if len(pgn_files) > 10:
        print(f"  ... and {len(pgn_files) - 10} more")

    # Process files in parallel
    print("\n🚀 Starting parallel preprocessing...")
    print(f"   Modal will spawn up to {len(pgn_files)} workers")
    print(f"   (actual concurrency limited by your account tier)")
    start_time = time.time()

    # Use spawn for proper parallel execution with multiple arguments
    futures = [preprocess_pgn_file.spawn(f, min_elo, positions_per_file) for f in pgn_files]
    results = [future.get() for future in futures]

    elapsed = time.time() - start_time

    # Aggregate results
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)

    total_games = sum(r.get("games_processed", 0) for r in results if r)
    total_skipped = sum(r.get("games_skipped", 0) for r in results if r)
    total_chunks = sum(r.get("chunks_created", 0) for r in results if r)
    total_positions = sum(r.get("total_positions", 0) for r in results if r)

    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Games processed: {total_games:,}")
    print(f"Games skipped: {total_skipped:,}")
    print(f"Output chunks: {total_chunks:,}")
    print(f"Total positions: {total_positions:,}")
    if elapsed > 0:
        print(f"Throughput: {total_positions / elapsed:.0f} positions/sec")

    print("\n✅ Data ready for training!")
    print(f"Location: /data/lc0_processed/")
    print("\nNext step: Train with:")
    print("  modal run training/scripts/train_modal_lc0_fixed.py")


@app.function(
    image=image,
    volumes={"/data": data_volume},
)
def list_pgn_files():
    """Helper function to list PGN files in volume."""
    from pathlib import Path
    pgn_dir = Path("/data/pgn/chunked")  # Updated to chunked subdirectory
    if not pgn_dir.exists():
        return []
    return [f.name for f in pgn_dir.glob("*.pgn")]


@app.local_entrypoint()
def list_files():
    """List PGN files in the volume."""
    print("Listing PGN files in /data/pgn/...")
    files = list_pgn_files.remote()
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  - {f}")
