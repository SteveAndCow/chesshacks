"""
Preprocess chess games into training data for neural network.

Input: PGN files with chess games OR FEN positions
Output: Tensors of (board_state, move, result) for training

Supports multiple input formats:
- PGN: Full game with move sequence
- FEN: Single position snapshots
- Outputs unified bitboard tensor representation (12, 8, 8)
"""
import chess
import chess.pgn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import io
import warnings
import sys

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert chess board to tensor representation.

    Enhanced version with game state information:
    - Channels 0-11: Piece positions (12 planes for 6 types × 2 colors)
    - Channels 12-13: Castling rights (kingside/queenside for both colors)
    - Channel 14: En passant file indicator
    - Channel 15: Halfmove clock (normalized to [0, 1])

    Total: 16 channels instead of 12
    """
    tensor = np.zeros((16, 8, 8), dtype=np.float32)

    piece_idx = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }

    # Fill piece positions (channels 0-11)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            idx = piece_idx[(piece.piece_type, piece.color)]
            rank = square // 8
            file = square % 8
            tensor[idx, rank, file] = 1.0

    # Channel 12: Castling rights (kingside)
    # Set to 1.0 across all squares if castling is available
    if board.has_kingside_castling_rights(chess.WHITE) or board.has_kingside_castling_rights(chess.BLACK):
        tensor[12, :, :] = 1.0

    # Channel 13: Castling rights (queenside)
    if board.has_queenside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.BLACK):
        tensor[13, :, :] = 1.0

    # Channel 14: En passant file indicator
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        tensor[14, :, ep_file] = 1.0

    # Channel 15: Halfmove clock (normalized to [0, 1])
    # Halfmove clock counts moves since last pawn move or capture (max 100 for draw)
    tensor[15, :, :] = board.halfmove_clock / 100.0

    return tensor

def move_to_index(move: chess.Move) -> int:
    """
    Convert move to an index for policy output.

    Total possible moves in chess: ~4672 (not all legal at once)
    Simplified: from_square (64) * to_square (64) = 4096 base moves
    Plus promotion moves

    For hackathon speed, use simple encoding:
    index = from_square * 64 + to_square
    """
    return move.from_square * 64 + move.to_square

def result_to_value(result: str, turn: chess.Color) -> float:
    """
    Convert game result to value for training.
    1.0 = win, 0.0 = draw, -1.0 = loss
    """
    if result == "1-0":  # White wins
        return 1.0 if turn == chess.WHITE else -1.0
    elif result == "0-1":  # Black wins
        return -1.0 if turn == chess.WHITE else 1.0
    else:  # Draw
        return 0.0

def fen_to_tensor(fen: str) -> np.ndarray:
    """
    Convert FEN string directly to tensor representation.
    This is a convenience wrapper around board_to_tensor.

    Args:
        fen: FEN string representing a chess position

    Returns:
        numpy array of shape (12, 8, 8)
    """
    board = chess.Board(fen)
    return board_to_tensor(board)

def process_fen_file(fen_path: str, output_dir: str,
                     evaluations_provided: bool = False,
                     max_positions: int = None):
    """
    Process a file containing FEN positions into training examples.

    Expected format per line:
    - Without evaluations: "fen_string"
    - With evaluations: "fen_string evaluation"

    Args:
        fen_path: Path to file with FEN positions (one per line)
        output_dir: Directory to save processed data
        evaluations_provided: If True, expects each line to have "FEN eval"
        max_positions: Maximum number of positions to process
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    boards = []
    values = []

    with open(fen_path) as f:
        for idx, line in enumerate(tqdm(f, desc="Processing FEN positions")):
            if max_positions and idx >= max_positions:
                break

            line = line.strip()
            if not line:
                continue

            try:
                if evaluations_provided:
                    # Format: "FEN evaluation"
                    parts = line.split(maxsplit=6)  # FEN has 6 space-separated parts
                    fen = " ".join(parts[:6])
                    evaluation = float(parts[6]) if len(parts) > 6 else 0.0
                else:
                    # Just FEN, no evaluation
                    fen = line
                    evaluation = 0.0

                # Convert FEN to tensor
                board_tensor = fen_to_tensor(fen)
                boards.append(board_tensor)
                values.append(evaluation)

            except Exception as e:
                print(f"Error processing line {idx}: {e}")
                continue

    # Convert to numpy arrays and save
    boards_array = np.array(boards, dtype=np.float32)
    values_array = np.array(values, dtype=np.float32)

    print(f"\nSaving {len(boards)} training examples...")
    np.save(output_dir / "boards.npy", boards_array)
    np.save(output_dir / "values.npy", values_array)

    print(f"Done! Saved to {output_dir}")
    print(f"Total positions: {len(boards)}")
    print(f"Board shape: {boards_array.shape}")
    print(f"Values shape: {values_array.shape}")

def process_pgn_file(pgn_path: str, output_dir: str, max_games: int = None):
    """
    Process a PGN file into training examples.

    For each position in each game, create:
    - Input: Board state tensor
    - Target policy: Move played (as index)
    - Target value: Game result
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    boards = []
    moves = []
    values = []

    with open(pgn_path) as pgn_file:
        game_count = 0
        skipped_games = 0

        while True:
            # Suppress python-chess warnings about illegal moves
            # These are printed to stderr by the library during parsing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Redirect stderr temporarily to suppress illegal SAN warnings
                original_stderr = sys.stderr
                sys.stderr = io.StringIO()

                try:
                    game = chess.pgn.read_game(pgn_file)
                finally:
                    sys.stderr = original_stderr
            
            if game is None:
                break

            if max_games and game_count >= max_games:
                break

            result = game.headers.get("Result", "*")
            if result == "*":  # Skip unfinished games
                skipped_games += 1
                continue

            board = game.board()

            # Try to iterate through moves, skip if parsing errors
            try:
                # Iterate through all moves in the game
                for move in game.mainline_moves():
                    # Store this position and the move played
                    board_tensor = board_to_tensor(board)
                    move_idx = move_to_index(move)
                    value = result_to_value(result, board.turn)

                    boards.append(board_tensor)
                    moves.append(move_idx)
                    values.append(value)

                    board.push(move)
            except (ValueError, AssertionError):
                # Skip games with illegal moves or parsing errors
                skipped_games += 1
                continue

            game_count += 1
            if game_count % 1000 == 0:
                print(f"Processed {game_count} games, {len(boards)} positions "
                      f"(skipped {skipped_games} invalid games)")

    # Convert to numpy arrays and save
    boards_array = np.array(boards, dtype=np.float32)
    moves_array = np.array(moves, dtype=np.int64)
    values_array = np.array(values, dtype=np.float32)

    print(f"\nSaving {len(boards)} training examples...")
    np.save(output_dir / "boards.npy", boards_array)
    np.save(output_dir / "moves.npy", moves_array)
    np.save(output_dir / "values.npy", values_array)

    print(f"Done! Saved to {output_dir}")
    print(f"Total positions: {len(boards)}")
    print(f"Total games processed: {game_count}")
    print(f"Games skipped (invalid/unfinished): {skipped_games}")
    print(f"Board shape: {boards_array.shape}")
    print(f"Moves shape: {moves_array.shape}")
    print(f"Values shape: {values_array.shape}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess chess data (PGN or FEN) into neural network training format"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input file (PGN or FEN)")
    parser.add_argument("--format", type=str, choices=["pgn", "fen"], default="pgn",
                        help="Input format: 'pgn' for games or 'fen' for positions")
    parser.add_argument("--output", type=str, default="training/data/processed",
                        help="Output directory")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Max games to process (for PGN)")
    parser.add_argument("--max-positions", type=int, default=None,
                        help="Max positions to process (for FEN)")
    parser.add_argument("--with-evaluations", action="store_true",
                        help="FEN file includes evaluations (format: 'FEN eval')")

    args = parser.parse_args()

    if args.format == "pgn":
        print(f"Processing PGN file: {args.input}")
        process_pgn_file(args.input, args.output, args.max_games)
    elif args.format == "fen":
        print(f"Processing FEN file: {args.input}")
        process_fen_file(args.input, args.output,
                         args.with_evaluations, args.max_positions)
    else:
        raise ValueError(f"Unsupported format: {args.format}")
