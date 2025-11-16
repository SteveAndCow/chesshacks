#!/usr/bin/env python3
"""
Quick test script for LC0 model inference.

Tests:
1. Model loading from HuggingFace
2. Board representation conversion
3. Move prediction
4. Position evaluation
"""
import sys
sys.path.insert(0, 'src')

import chess
from src.models.lc0_inference import LC0ModelLoader

def test_lc0_model():
    """Test the LC0 model on a few positions."""

    print("="*60)
    print("LC0 MODEL LOCAL TEST")
    print("="*60)

    # Initialize model loader
    print("\n1️⃣  Initializing LC0 model loader...")
    model_loader = LC0ModelLoader(
        repo_id="steveandcow/chesshacks-lc0",
        model_file="best_lc0_model.pt",
        device="cpu"  # Change to "cuda" if you have GPU
    )

    # Load model
    print("\n2️⃣  Loading model from HuggingFace...")
    try:
        model_loader.load_model()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test positions
    test_positions = [
        ("Starting position", chess.Board()),
        ("Sicilian Defense", chess.Board("r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")),
        ("Endgame", chess.Board("8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 99 50")),
    ]

    print("\n3️⃣  Testing predictions on sample positions...")
    print("="*60)

    for name, board in test_positions:
        print(f"\n📋 {name}")
        print(f"FEN: {board.fen()}")
        print(board)
        print()

        try:
            # Get predictions
            move_probs, value = model_loader.predict(board)

            # Show top 5 moves
            top_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:5]

            print(f"Position Value: {value:+.3f} (from current player's perspective)")
            print(f"Win/Draw/Loss estimate: W:{(value+1)/2:.1%}, L:{(1-value)/2:.1%}")
            print("\nTop 5 moves:")
            for move, prob in top_moves:
                print(f"  {move.uci():6s} → {prob:6.2%}")

            # Get best move
            best_move, _ = model_loader.get_best_move(board)
            print(f"\n🎯 Best move: {best_move.uci()}")

        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()

        print("-"*60)

    print("\n✅ All tests completed!")
    print("="*60)


if __name__ == "__main__":
    test_lc0_model()
