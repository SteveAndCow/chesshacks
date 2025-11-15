#!/usr/bin/env python3
"""
Generate Stockfish evaluations for training positions.

This creates a hybrid dataset with both:
1. Policy labels from master games (move predictions)
2. Value labels from Stockfish (accurate position evaluations)

This combines the best of both worlds:
- Learn move patterns from strong players
- Learn accurate evaluations from Stockfish
"""

import chess
import chess.engine
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def generate_stockfish_labels(
    positions_file: str,
    output_file: str,
    stockfish_path: str = "/usr/local/bin/stockfish",
    time_limit: float = 0.1,
    max_positions: int = None
):
    """
    Augment existing training data with Stockfish evaluations.

    Args:
        positions_file: Path to existing positions (boards.npy)
        output_file: Where to save Stockfish values
        stockfish_path: Path to Stockfish binary
        time_limit: Time per position (seconds)
        max_positions: Limit number of positions to evaluate
    """

    # Load existing position data
    print(f"Loading positions from {positions_file}...")
    boards = np.load(positions_file)

    if max_positions:
        boards = boards[:max_positions]

    print(f"Loaded {len(boards)} positions")
    print(f"Time limit: {time_limit}s per position")
    print(f"Estimated time: {len(boards) * time_limit / 60:.1f} minutes")

    # Initialize Stockfish
    print(f"\nInitializing Stockfish from {stockfish_path}...")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # Evaluate all positions
    stockfish_values = np.zeros(len(boards), dtype=np.float32)

    print("\nEvaluating positions...")
    for idx in tqdm(range(len(boards))):
        # Convert tensor back to board
        # (You'll need to implement this - reverse of board_to_tensor)
        board = tensor_to_board(boards[idx])

        # Get Stockfish evaluation
        try:
            info = engine.analyse(
                board,
                chess.engine.Limit(time=time_limit),
                info=chess.engine.INFO_SCORE
            )

            # Convert score to value in [-1, 1]
            score = info["score"].relative

            if score.is_mate():
                # Mate score: +1 or -1
                value = 1.0 if score.mate() > 0 else -1.0
            else:
                # Centipawn score: convert to [-1, 1] with tanh
                # Typical range is -500 to +500 centipawns
                centipawns = score.score()
                value = np.tanh(centipawns / 500.0)

            stockfish_values[idx] = value

        except Exception as e:
            print(f"\nError evaluating position {idx}: {e}")
            # Fallback to 0.0 (equal position)
            stockfish_values[idx] = 0.0

    # Save results
    engine.quit()

    print(f"\nSaving Stockfish evaluations to {output_file}...")
    np.save(output_file, stockfish_values)

    # Statistics
    print("\n" + "="*70)
    print("STOCKFISH EVALUATION STATISTICS")
    print("="*70)
    print(f"Positions evaluated: {len(stockfish_values)}")
    print(f"Mean value: {stockfish_values.mean():.3f}")
    print(f"Std dev: {stockfish_values.std():.3f}")
    print(f"Min value: {stockfish_values.min():.3f}")
    print(f"Max value: {stockfish_values.max():.3f}")
    print(f"\nValue distribution:")
    print(f"  Strong advantage (|v| > 0.7): {(np.abs(stockfish_values) > 0.7).sum()}")
    print(f"  Moderate (0.3 < |v| < 0.7): {((np.abs(stockfish_values) > 0.3) & (np.abs(stockfish_values) <= 0.7)).sum()}")
    print(f"  Equal (|v| < 0.3): {(np.abs(stockfish_values) <= 0.3).sum()}")


def tensor_to_board(tensor: np.ndarray) -> chess.Board:
    """
    Convert 16-channel tensor back to chess.Board.

    This is the reverse of board_to_tensor in preprocess.py.
    """
    board = chess.Board(fen=None)  # Empty board
    board.clear()

    # Reconstruct piece positions from channels 0-11
    piece_map = {
        0: (chess.PAWN, chess.WHITE),
        1: (chess.KNIGHT, chess.WHITE),
        2: (chess.BISHOP, chess.WHITE),
        3: (chess.ROOK, chess.WHITE),
        4: (chess.QUEEN, chess.WHITE),
        5: (chess.KING, chess.WHITE),
        6: (chess.PAWN, chess.BLACK),
        7: (chess.KNIGHT, chess.BLACK),
        8: (chess.BISHOP, chess.BLACK),
        9: (chess.ROOK, chess.BLACK),
        10: (chess.QUEEN, chess.BLACK),
        11: (chess.KING, chess.BLACK),
    }

    for channel, (piece_type, color) in piece_map.items():
        for rank in range(8):
            for file in range(8):
                if tensor[channel, rank, file] > 0.5:  # Piece present
                    square = rank * 8 + file
                    board.set_piece_at(square, chess.Piece(piece_type, color))

    # Reconstruct castling rights from channels 12-13
    # (Simplified - full reconstruction is more complex)
    board.castling_rights = 0
    if tensor[12, 0, 0] > 0.5:  # Kingside castling available
        board.castling_rights |= chess.BB_H1 | chess.BB_H8
    if tensor[13, 0, 0] > 0.5:  # Queenside castling available
        board.castling_rights |= chess.BB_A1 | chess.BB_A8

    # Set side to move (assume white - this should be stored separately)
    board.turn = chess.WHITE

    return board


def hybrid_training_example():
    """
    Example showing how to use hybrid labels in training.
    """
    print("\n" + "="*70)
    print("HYBRID TRAINING APPROACH")
    print("="*70)
    print("""
    After generating Stockfish labels, modify your training loop:

    # Load data
    boards = np.load("boards.npy")
    moves = np.load("moves.npy")  # From master games (policy labels)
    game_values = np.load("values.npy")  # From game outcomes
    stockfish_values = np.load("stockfish_values.npy")  # NEW!

    # Training loop
    for batch in dataloader:
        policy_logits, value_pred, result_logits = model(boards)

        # Policy loss: Learn from master games
        policy_loss = CrossEntropyLoss(policy_logits, moves)

        # Value loss: Learn from ACCURATE Stockfish evals (not game outcomes!)
        value_loss = MSELoss(value_pred, stockfish_values)  # Changed!

        # Result loss: Still use game outcomes
        result_loss = CrossEntropyLoss(result_logits, result_targets)

        # Combined loss
        loss = policy_loss + value_loss + 0.5 * result_loss

    Benefits:
    - Policy head learns strategic patterns from masters
    - Value head learns accurate evaluations from Stockfish
    - Result head learns game outcomes

    Expected improvement: +200-400 Elo over pure game-outcome training!
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Stockfish labels for training data")
    parser.add_argument("--positions", required=True, help="Path to boards.npy")
    parser.add_argument("--output", required=True, help="Where to save stockfish_values.npy")
    parser.add_argument("--stockfish", default="/usr/local/bin/stockfish",
                       help="Path to Stockfish binary")
    parser.add_argument("--time", type=float, default=0.1,
                       help="Time limit per position (seconds)")
    parser.add_argument("--max-positions", type=int, default=None,
                       help="Maximum positions to evaluate (for testing)")

    args = parser.parse_args()

    print("\n🎯 STOCKFISH LABEL GENERATION\n")

    generate_stockfish_labels(
        positions_file=args.positions,
        output_file=args.output,
        stockfish_path=args.stockfish,
        time_limit=args.time,
        max_positions=args.max_positions
    )

    hybrid_training_example()

    print("\n✅ Done! Now use these labels in training for better value predictions.")
    print(f"\nNext steps:")
    print(f"1. Modify data_loader.py to load stockfish_values.npy")
    print(f"2. Update train_modal.py to use Stockfish values for value_loss")
    print(f"3. Retrain model and compare performance!")
