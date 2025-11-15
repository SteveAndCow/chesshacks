#!/usr/bin/env python3
"""
Real-time comparison of your model's moves vs. Stockfish.

This creates an interactive interface to:
1. Play through games and see model vs. Stockfish recommendations
2. Identify positions where model fails badly
3. Understand model strengths/weaknesses

Usage:
    python compare_moves_live.py \
        --model checkpoints/transformer_tiny.pt \
        --stockfish /usr/local/bin/stockfish
"""

import chess
import chess.engine
import chess.svg
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_vs_stockfish import ModelEvaluator


class LiveMoveComparison:
    """Interactive move comparison tool."""

    def __init__(self, evaluator: ModelEvaluator):
        self.evaluator = evaluator
        self.board = chess.Board()

    def display_position(self):
        """Display current position."""
        print(f"\n{'='*70}")
        print(f"Position: {self.board.fen()}")
        print(f"Turn: {'White' if self.board.turn else 'Black'}")
        print(f"Move: {self.board.fullmove_number}")
        print(f"{'='*70}")
        print(self.board)
        print(f"{'='*70}")

    def compare_moves(self, top_n: int = 5):
        """Compare model and Stockfish top moves."""
        # Get model predictions
        pred = self.evaluator.predict(self.board)

        # Get legal moves sorted by model probability
        legal_moves = list(self.board.legal_moves)
        move_probs = []
        for move in legal_moves:
            from preprocess import move_to_index
            move_idx = move_to_index(move)
            prob = pred["policy"][move_idx]
            move_probs.append((move, prob))

        move_probs.sort(key=lambda x: x[1], reverse=True)

        # Get Stockfish analysis
        sf_analysis = self.evaluator.get_stockfish_analysis(self.board)

        # Display comparison
        print(f"\n📊 MODEL vs. STOCKFISH COMPARISON")
        print(f"{'-'*70}")

        print(f"\n🤖 MODEL PREDICTIONS:")
        print(f"  Position eval: {pred['value']:+.3f} (tanh scale)")
        print(f"  Result probs: Loss={pred['result'][0]:.1%}, "
              f"Draw={pred['result'][1]:.1%}, Win={pred['result'][2]:.1%}")
        print(f"\n  Top {top_n} moves:")
        for i, (move, prob) in enumerate(move_probs[:top_n], 1):
            san = self.board.san(move)
            marker = "✓" if move == sf_analysis["best_move"] else " "
            print(f"    {i}. {san:8} (prob={prob:.2%}) {marker}")

        print(f"\n🐟 STOCKFISH ANALYSIS:")
        print(f"  Position eval: {sf_analysis['value']:+.3f} (tanh scale)")
        if sf_analysis['score_cp'] is not None:
            print(f"  Raw score: {sf_analysis['score_cp']:+d} centipawns")
        print(f"  Best move: {self.board.san(sf_analysis['best_move'])}")

        # Agreement check
        model_best = move_probs[0][0]
        if model_best == sf_analysis["best_move"]:
            print(f"\n✅ AGREEMENT: Model and Stockfish agree!")
        else:
            print(f"\n⚠️  DISAGREEMENT:")
            print(f"    Model prefers: {self.board.san(model_best)} "
                  f"(prob={move_probs[0][1]:.1%})")
            print(f"    Stockfish prefers: {self.board.san(sf_analysis['best_move'])}")

            # Find Stockfish move in model's rankings
            sf_rank = None
            for i, (move, prob) in enumerate(move_probs, 1):
                if move == sf_analysis["best_move"]:
                    sf_rank = i
                    sf_prob = prob
                    break

            if sf_rank:
                print(f"    Stockfish move ranked #{sf_rank} by model (prob={sf_prob:.1%})")
            else:
                print(f"    Stockfish move not in top moves!")

        # Evaluation agreement
        eval_diff = abs(pred['value'] - sf_analysis['value'])
        if eval_diff < 0.2:
            print(f"  Evaluation: Close agreement (diff={eval_diff:.3f})")
        else:
            print(f"  Evaluation: Large disagreement (diff={eval_diff:.3f})")

    def interactive_session(self):
        """Run interactive session."""
        print("\n🎮 INTERACTIVE MOVE COMPARISON")
        print("Commands:")
        print("  <move>  - Make a move (e.g., 'e4', 'Nf3', 'O-O')")
        print("  'compare' - Show model vs. Stockfish comparison")
        print("  'reset'   - Reset to starting position")
        print("  'fen <fen>' - Load position from FEN")
        print("  'quit'    - Exit")

        while True:
            self.display_position()
            self.compare_moves(top_n=5)

            try:
                command = input("\n>>> ").strip()

                if command == "quit":
                    break
                elif command == "reset":
                    self.board = chess.Board()
                elif command.startswith("fen "):
                    fen = command[4:]
                    self.board = chess.Board(fen)
                elif command == "compare":
                    # Already shown above
                    continue
                else:
                    # Try to parse as move
                    try:
                        move = self.board.parse_san(command)
                        self.board.push(move)
                    except ValueError:
                        print(f"❌ Invalid move: {command}")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def analyze_interesting_positions(self):
        """Show some interesting test positions."""
        interesting_fens = [
            # Tactical position
            ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
             "Italian Game - Classical"),

            # Endgame
            ("8/8/4k3/8/8/3K4/6R1/8 w - - 0 1",
             "Rook endgame - Basic technique"),

            # Tactics - Fork
            ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
             "Knight fork opportunity"),

            # Sacrifice
            ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
             "After short castle - tactical alert"),
        ]

        print("\n🎯 ANALYZING INTERESTING POSITIONS\n")

        for fen, description in interesting_fens:
            print(f"\n{'='*70}")
            print(f"Position: {description}")
            print(f"{'='*70}")

            self.board = chess.Board(fen)
            print(self.board)

            self.compare_moves(top_n=3)

            input("\nPress Enter for next position...")


def main():
    parser = argparse.ArgumentParser(description="Live move comparison tool")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--stockfish", default="/usr/local/bin/stockfish",
                       help="Path to Stockfish binary")
    parser.add_argument("--mode", choices=["interactive", "test"],
                       default="interactive",
                       help="Mode: interactive session or predefined tests")
    parser.add_argument("--depth", type=int, default=15,
                       help="Stockfish search depth")

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("LIVE MOVE COMPARISON TOOL")
    print(f"{'='*70}")

    evaluator = ModelEvaluator(
        model_path=args.model,
        stockfish_path=args.stockfish,
        stockfish_depth=args.depth,
    )

    comparator = LiveMoveComparison(evaluator)

    if args.mode == "interactive":
        comparator.interactive_session()
    else:
        comparator.analyze_interesting_positions()


if __name__ == "__main__":
    main()
