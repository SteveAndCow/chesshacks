"""
Preprocess chess games into training data for neural network.

Input: PGN files with chess games
Output: Tensors of (board_state, move, result) for training
"""
import chess
import chess.pgn
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import io

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert chess board to tensor representation.

    Multiple options:
    1. Bitboard representation (8x8x12 for each piece type/color)
    2. Simple piece encoding (8x8 with piece values)
    3. With history planes (8x8x(12*history + extras))

    This uses option 1: 12 planes for piece types
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

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

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            idx = piece_idx[(piece.piece_type, piece.color)]
            rank = square // 8
            file = square % 8
            tensor[idx, rank, file] = 1.0

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

        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            if max_games and game_count >= max_games:
                break

            result = game.headers.get("Result", "*")
            if result == "*":  # Skip unfinished games
                continue

            board = game.board()

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

            game_count += 1
            if game_count % 1000 == 0:
                print(f"Processed {game_count} games, {len(boards)} positions")

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
    print(f"Board shape: {boards_array.shape}")
    print(f"Moves shape: {moves_array.shape}")
    print(f"Values shape: {values_array.shape}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=str, required=True, help="Path to PGN file")
    parser.add_argument("--output", type=str, default="training/data/processed", help="Output directory")
    parser.add_argument("--max-games", type=int, default=None, help="Max games to process")

    args = parser.parse_args()

    process_pgn_file(args.pgn, args.output, args.max_games)
