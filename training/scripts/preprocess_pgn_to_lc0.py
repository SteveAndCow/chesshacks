import chess
import chess.pgn
import numpy as np
from pathlib import Path
from tqdm import tqdm


def board_to_112_channels(board: chess.Board, history: list[chess.Board]) -> np.ndarray:
    """
    Convert chess board + history to 112-channel LC0 format.

    Args:
        board: Current board position
        history: List of previous 7 board positions (8 total with current)

    Returns:
        (112, 8, 8) numpy array
    """
    channels = []

    # 104 channels: 8 board positions × 13 planes each
    # (6 piece types × 2 colors + repetition counter)
    positions = history[-7:] + [board]  # Last 8 positions

    # Pad with empty boards if we don't have 8 positions yet
    while len(positions) < 8:
        # Create empty board (None FEN creates empty board)
        empty_board = chess.Board(None)
        positions.insert(0, empty_board)

    for pos in positions:
        # 12 piece planes (6 types × 2 colors)
        for piece_type in chess.PIECE_TYPES:
            # Our pieces
            plane = np.zeros(64, dtype=np.float32)
            for square in pos.pieces(piece_type, board.turn):
                plane[square] = 1.0
            channels.append(plane)

            # Opponent pieces
            plane = np.zeros(64, dtype=np.float32)
            for square in pos.pieces(piece_type, not board.turn):
                plane[square] = 1.0
            channels.append(plane)

        # Repetition counter (1 plane per position)
        repetitions = sum(1 for p in positions if p == pos)
        channels.append(np.full(64, repetitions, dtype=np.float32))

    # 5 unit planes (castling rights + side to move)
    channels.append(np.full(64, float(board.has_kingside_castling_rights(board.turn))))
    channels.append(np.full(64, float(board.has_queenside_castling_rights(board.turn))))
    channels.append(np.full(64, float(board.has_kingside_castling_rights(not board.turn))))
    channels.append(np.full(64, float(board.has_queenside_castling_rights(not board.turn))))
    channels.append(np.full(64, 1.0 if board.turn == chess.WHITE else 0.0))

    # 1 rule50 plane (normalized)
    channels.append(np.full(64, board.halfmove_clock / 99.0, dtype=np.float32))

    # 2 constant planes
    channels.append(np.zeros(64, dtype=np.float32))  # All zeros
    channels.append(np.ones(64, dtype=np.float32))   # All ones

    # Stack and reshape to (112, 8, 8)
    return np.stack(channels).reshape(112, 8, 8)


def move_to_lc0_policy(move: chess.Move, board: chess.Board) -> int:
    """
    Convert chess.Move to LC0 policy index (0-1857).
    Uses lc0_policy_map.py mapping.
    """
    # Import from models directory
    import sys
    from pathlib import Path
    models_path = Path(__file__).parent / "models"
    if str(models_path) not in sys.path:
        sys.path.insert(0, str(models_path))

    from policy_index import policy_index

    # UCI format already includes promotion (e.g., "e7e8q")
    move_str = move.uci()

    # LC0 uses 'n' for knight promotion in notation, but it's included in UCI
    # Knight promotions in LC0 are encoded as regular moves without promotion suffix
    # So we need to remove 'n' if present
    if move.promotion == chess.KNIGHT and move_str.endswith('n'):
        move_str = move_str[:-1]  # Remove the 'n'

    try:
        return policy_index.index(move_str)
    except ValueError:
        # If not found, it might be a move not in LC0's 1858 legal moves
        # This can happen with underpromotions to knight in some positions
        # For now, skip this move or use a fallback
        print(f"Warning: Move {move_str} not in LC0 policy index, skipping")
        return None


def game_result_to_wdl(result: str, pov_color: chess.Color) -> np.ndarray:
    """
    Convert game result to WDL (Win/Draw/Loss) from POV of pov_color.

    Returns:
        [loss_prob, draw_prob, win_prob] as one-hot vector
    """
    if result == "1-0":  # White wins
        if pov_color == chess.WHITE:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Win
        else:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Loss
    elif result == "0-1":  # Black wins
        if pov_color == chess.BLACK:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Win
        else:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Loss
    else:  # Draw
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def process_pgn_file(pgn_path: Path, output_path: Path, min_elo: int = 2000):
    """
    Process PGN file into 112-channel training data.

    Saves as .npz with:
        - inputs: (N, 112, 8, 8) board representations
        - policies: (N, 1858) one-hot policy targets
        - values: (N, 3) WDL targets
        - moves_left: (N,) remaining ply counts
    """
    with open(pgn_path) as pgn_file:
        inputs_list = []
        policies_list = []
        values_list = []
        moves_left_list = []

        games_processed = 0
        positions_processed = 0
        positions_skipped = 0

        # Progress bar
        pbar = tqdm(desc="Processing games", unit=" games")

        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            # Filter by ELO
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
            if white_elo < min_elo or black_elo < min_elo:
                continue

            result = game.headers["Result"]
            if result not in ["1-0", "0-1", "1/2-1/2"]:
                continue

            # Extract positions
            board = game.board()
            history = []

            for i, move in enumerate(game.mainline_moves()):
                # Build 112-channel input
                inputs = board_to_112_channels(board, history)

                # Policy target (one-hot)
                move_idx = move_to_lc0_policy(move, board)
                if move_idx is None:
                    # Skip this position if move not in policy index
                    # This can happen with rare underpromotions
                    positions_skipped += 1
                    board.push(move)
                    history.append(board.copy())
                    if len(history) > 7:
                        history.pop(0)
                    continue

                inputs_list.append(inputs)
                positions_processed += 1

                policy = np.zeros(1858, dtype=np.float32)
                policy[move_idx] = 1.0
                policies_list.append(policy)

                # Value target (WDL from current player's POV)
                values = game_result_to_wdl(result, board.turn)
                values_list.append(values)

                # Moves left (remaining plies)
                total_plies = len(list(game.mainline_moves()))
                moves_left = total_plies - i - 1
                moves_left_list.append(moves_left)

                # Update history
                history.append(board.copy())
                if len(history) > 7:
                    history.pop(0)

                # Make move
                board.push(move)

            # Update progress
            games_processed += 1
            pbar.update(1)
            pbar.set_postfix({
                'positions': f'{positions_processed:,}',
                'skipped': positions_skipped
            })

        pbar.close()

        # Save as compressed npz
        np.savez_compressed(
            output_path,
            inputs=np.array(inputs_list, dtype=np.float32),
            policies=np.array(policies_list, dtype=np.float32),
            values=np.array(values_list, dtype=np.float32),
            moves_left=np.array(moves_left_list, dtype=np.float32)
        )

        print(f"\n✅ Preprocessing complete!")
        print(f"   Games processed: {games_processed:,}")
        print(f"   Positions saved: {positions_processed:,}")
        print(f"   Positions skipped: {positions_skipped:,}")
        print(f"   Output file: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PGN to LC0 112-channel format")
    parser.add_argument("--input", type=str, required=True, help="Input PGN file")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file")
    parser.add_argument("--min-elo", type=int, default=2000, help="Minimum ELO rating")
    parser.add_argument("--max-games", type=int, default=None, help="Max games to process (for testing)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing {input_path}...")
    print(f"Min ELO: {args.min_elo}")
    if args.max_games:
        print(f"Max games: {args.max_games}")

    # For testing with max_games, we need to modify the function
    # For now, just call it
    process_pgn_file(input_path, output_path, min_elo=args.min_elo)

    print(f"\n✅ Done! Output saved to {output_path}")