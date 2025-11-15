#!/usr/bin/env python3
"""
Evaluate your model's performance using Stockfish as ground truth.

This script measures:
1. Move prediction accuracy vs. Stockfish best moves
2. Position evaluation correlation with Stockfish
3. Tactical puzzle solving ability
4. Position understanding in different game phases

Usage:
    python evaluate_vs_stockfish.py \
        --model checkpoints/transformer_tiny.pt \
        --stockfish /usr/local/bin/stockfish \
        --test-positions test_positions.pgn
"""

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import sys
from pathlib import Path
from tqdm import tqdm
import argparse
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from models.model_factory import create_model_from_config
from preprocess import board_to_tensor, move_to_index


class ModelEvaluator:
    """Evaluate chess model against Stockfish."""

    def __init__(
        self,
        model_path: str,
        stockfish_path: str = "/usr/local/bin/stockfish",
        stockfish_depth: int = 15,
        stockfish_time: float = 1.0,
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model checkpoint
            stockfish_path: Path to Stockfish binary
            stockfish_depth: Search depth for Stockfish
            stockfish_time: Time limit per position (seconds)
        """
        self.model_path = model_path
        self.stockfish_depth = stockfish_depth
        self.stockfish_time = stockfish_time

        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model = self._load_model(checkpoint)
        self.model.eval()

        # Initialize Stockfish
        print(f"Initializing Stockfish (depth={stockfish_depth}, time={stockfish_time}s)...")
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def _load_model(self, checkpoint):
        """Reconstruct model from checkpoint."""
        # This assumes you saved model_type and model_params
        model_type = checkpoint.get("model_type", "transformer_lite")
        model_params = checkpoint.get("model_params", {})

        # Create model using factory
        config = {
            "model": {
                "type": model_type,
                "params": model_params
            }
        }

        from model_factory import create_model_from_config
        model = create_model_from_config(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    def predict(self, board: chess.Board):
        """Get model predictions for a position."""
        tensor = board_to_tensor(board)
        tensor = torch.from_numpy(tensor).unsqueeze(0).float()  # Add batch dim

        with torch.no_grad():
            policy_logits, value_pred, result_logits = self.model(tensor)

        # Convert to numpy
        policy_probs = torch.softmax(policy_logits[0], dim=0).numpy()
        value = value_pred[0, 0].item()
        result_probs = torch.softmax(result_logits[0], dim=0).numpy()

        return {
            "policy": policy_probs,
            "value": value,
            "result": result_probs,  # [loss, draw, win]
        }

    def get_stockfish_analysis(self, board: chess.Board):
        """Get Stockfish's best move and evaluation."""
        info = self.engine.analyse(
            board,
            chess.engine.Limit(depth=self.stockfish_depth, time=self.stockfish_time)
        )

        best_move = info["pv"][0] if "pv" in info else None
        score = info["score"].relative

        # Convert score to value in [-1, 1]
        if score.is_mate():
            value = 1.0 if score.mate() > 0 else -1.0
        else:
            centipawns = score.score()
            value = np.tanh(centipawns / 500.0)

        return {
            "best_move": best_move,
            "value": value,
            "score_cp": score.score() if not score.is_mate() else None,
        }

    def test_move_accuracy(self, positions: list[chess.Board], top_n: int = 3):
        """
        Test if model's top moves match Stockfish's best move.

        Args:
            positions: List of chess positions
            top_n: Check if Stockfish move is in top-N predictions

        Returns:
            Dictionary with accuracy metrics
        """
        print(f"\n{'='*70}")
        print("MOVE PREDICTION ACCURACY TEST")
        print(f"{'='*70}")

        top1_correct = 0
        top3_correct = 0
        top5_correct = 0

        for board in tqdm(positions, desc="Testing move accuracy"):
            # Get model's top moves
            pred = self.predict(board)
            legal_moves = list(board.legal_moves)

            # Get probabilities for legal moves only
            move_probs = []
            for move in legal_moves:
                move_idx = move_to_index(move)
                prob = pred["policy"][move_idx]
                move_probs.append((move, prob))

            # Sort by probability
            move_probs.sort(key=lambda x: x[1], reverse=True)
            top_moves = [m for m, p in move_probs[:5]]

            # Get Stockfish's best move
            sf_analysis = self.get_stockfish_analysis(board)
            sf_best = sf_analysis["best_move"]

            # Check if SF move is in top-N
            if sf_best == top_moves[0]:
                top1_correct += 1
                top3_correct += 1
                top5_correct += 1
            elif sf_best in top_moves[:3]:
                top3_correct += 1
                top5_correct += 1
            elif sf_best in top_moves[:5]:
                top5_correct += 1

        total = len(positions)
        results = {
            "top1_accuracy": top1_correct / total * 100,
            "top3_accuracy": top3_correct / total * 100,
            "top5_accuracy": top5_correct / total * 100,
        }

        print(f"\nResults ({total} positions):")
        print(f"  Top-1 accuracy: {results['top1_accuracy']:.1f}%")
        print(f"  Top-3 accuracy: {results['top3_accuracy']:.1f}%")
        print(f"  Top-5 accuracy: {results['top5_accuracy']:.1f}%")

        return results

    def test_evaluation_correlation(self, positions: list[chess.Board]):
        """
        Test correlation between model and Stockfish evaluations.

        Args:
            positions: List of chess positions

        Returns:
            Dictionary with correlation metrics
        """
        print(f"\n{'='*70}")
        print("EVALUATION CORRELATION TEST")
        print(f"{'='*70}")

        model_values = []
        sf_values = []

        for board in tqdm(positions, desc="Testing evaluation correlation"):
            # Get model evaluation
            pred = self.predict(board)
            model_values.append(pred["value"])

            # Get Stockfish evaluation
            sf_analysis = self.get_stockfish_analysis(board)
            sf_values.append(sf_analysis["value"])

        model_values = np.array(model_values)
        sf_values = np.array(sf_values)

        # Calculate correlation
        pearson_corr = np.corrcoef(model_values, sf_values)[0, 1]
        spearman_corr, _ = spearmanr(model_values, sf_values)

        # Calculate MAE
        mae = np.abs(model_values - sf_values).mean()

        results = {
            "pearson": pearson_corr,
            "spearman": spearman_corr,
            "mae": mae,
            "model_values": model_values,
            "sf_values": sf_values,
        }

        print(f"\nResults ({len(positions)} positions):")
        print(f"  Pearson correlation: {pearson_corr:.3f}")
        print(f"  Spearman correlation: {spearman_corr:.3f}")
        print(f"  Mean absolute error: {mae:.3f}")

        return results

    def test_game_phase_understanding(self, positions_by_phase: dict):
        """
        Test model accuracy in different game phases.

        Args:
            positions_by_phase: Dict like {"opening": [...], "middlegame": [...], "endgame": [...]}

        Returns:
            Dictionary with per-phase metrics
        """
        print(f"\n{'='*70}")
        print("GAME PHASE UNDERSTANDING TEST")
        print(f"{'='*70}")

        results = {}

        for phase, positions in positions_by_phase.items():
            print(f"\nTesting {phase} ({len(positions)} positions)...")

            phase_results = self.test_move_accuracy(positions, top_n=3)
            phase_results["count"] = len(positions)
            results[phase] = phase_results

        print(f"\n{'='*70}")
        print("PHASE COMPARISON")
        print(f"{'='*70}")

        for phase, metrics in results.items():
            print(f"\n{phase.upper()} ({metrics['count']} positions):")
            print(f"  Top-1 accuracy: {metrics['top1_accuracy']:.1f}%")
            print(f"  Top-3 accuracy: {metrics['top3_accuracy']:.1f}%")

        return results

    def plot_evaluation_comparison(self, model_values, sf_values, output_path=None):
        """Plot model vs. Stockfish evaluations."""
        plt.figure(figsize=(10, 6))

        plt.scatter(sf_values, model_values, alpha=0.5, s=10)
        plt.plot([-1, 1], [-1, 1], 'r--', label='Perfect correlation')

        plt.xlabel("Stockfish Evaluation")
        plt.ylabel("Model Evaluation")
        plt.title("Model vs. Stockfish Position Evaluation")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to {output_path}")
        else:
            plt.show()

    def run_full_evaluation(self, test_positions_file: str, output_dir: str = "evaluation_results"):
        """
        Run comprehensive evaluation suite.

        Args:
            test_positions_file: PGN file with test positions
            output_dir: Where to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print(f"\n{'='*70}")
        print("COMPREHENSIVE MODEL EVALUATION")
        print(f"{'='*70}")

        # Load test positions
        print(f"\nLoading test positions from {test_positions_file}...")
        positions = self._load_positions_from_pgn(test_positions_file, max_positions=500)
        print(f"Loaded {len(positions)} test positions")

        # Split by game phase
        positions_by_phase = self._categorize_by_phase(positions)

        # Run tests
        move_results = self.test_move_accuracy(positions, top_n=5)
        eval_results = self.test_evaluation_correlation(positions)
        phase_results = self.test_game_phase_understanding(positions_by_phase)

        # Plot
        self.plot_evaluation_comparison(
            eval_results["model_values"],
            eval_results["sf_values"],
            output_path=output_dir / "eval_correlation.png"
        )

        # Save results
        results = {
            "move_accuracy": move_results,
            "evaluation_correlation": {
                "pearson": eval_results["pearson"],
                "spearman": eval_results["spearman"],
                "mae": eval_results["mae"],
            },
            "phase_results": phase_results,
        }

        import json
        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"\nModel: {self.model_path}")
        print(f"Test positions: {len(positions)}")
        print(f"\nMove Prediction:")
        print(f"  Top-1: {move_results['top1_accuracy']:.1f}%")
        print(f"  Top-3: {move_results['top3_accuracy']:.1f}%")
        print(f"\nEvaluation Quality:")
        print(f"  Correlation: {eval_results['pearson']:.3f}")
        print(f"  MAE: {eval_results['mae']:.3f}")
        print(f"\nResults saved to {output_dir}/")

        return results

    def _load_positions_from_pgn(self, pgn_file: str, max_positions: int = None):
        """Load positions from PGN file."""
        positions = []

        with open(pgn_file) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                board = game.board()
                for move in game.mainline_moves():
                    positions.append(board.copy())
                    board.push(move)

                    if max_positions and len(positions) >= max_positions:
                        return positions

        return positions

    def _categorize_by_phase(self, positions: list[chess.Board]):
        """Split positions by game phase."""
        by_phase = {
            "opening": [],
            "middlegame": [],
            "endgame": [],
        }

        for board in positions:
            # Count pieces (excluding kings)
            piece_count = len(board.piece_map()) - 2

            # Simple heuristic
            if board.fullmove_number <= 10:
                by_phase["opening"].append(board)
            elif piece_count <= 10:
                by_phase["endgame"].append(board)
            else:
                by_phase["middlegame"].append(board)

        return by_phase

    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'engine'):
            self.engine.quit()


def main():
    parser = argparse.ArgumentParser(description="Evaluate chess model vs. Stockfish")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--stockfish", default="/usr/local/bin/stockfish",
                       help="Path to Stockfish binary")
    parser.add_argument("--test-positions", required=True,
                       help="PGN file with test positions")
    parser.add_argument("--output-dir", default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--depth", type=int, default=15,
                       help="Stockfish search depth")
    parser.add_argument("--time", type=float, default=1.0,
                       help="Stockfish time limit per position")

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        model_path=args.model,
        stockfish_path=args.stockfish,
        stockfish_depth=args.depth,
        stockfish_time=args.time,
    )

    evaluator.run_full_evaluation(
        test_positions_file=args.test_positions,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
