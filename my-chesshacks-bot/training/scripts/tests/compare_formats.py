"""
Quick comparison script to show the difference between PGN and FEN training data.
"""
import chess
import chess.pgn
import io
from preprocess import board_to_tensor, fen_to_tensor, move_to_index, result_to_value


def demo_pgn_training_data():
    """Show what training data looks like from a PGN game."""
    print("=" * 70)
    print("PGN TRAINING DATA EXAMPLE")
    print("=" * 70)

    # Sample game: Italian Game
    pgn_text = """
[Event "Example Game"]
[White "Strong Player"]
[Black "Strong Player"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 1-0
"""

    pgn = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn)
    board = game.board()
    result = game.headers["Result"]

    print(f"\nGame: {game.headers['White']} vs {game.headers['Black']}")
    print(f"Result: {result}")
    print(f"\nExtracting training examples from each position:\n")

    positions = []
    for idx, move in enumerate(game.mainline_moves(), 1):
        print(f"Position {idx}:")
        print(f"  FEN: {board.fen()}")
        print(f"  Move played: {board.san(move)}")
        print(f"  Move index: {move_to_index(move)}")
        print(f"  Value (from game result): {result_to_value(result, board.turn):.1f}")

        # What goes into training
        board_tensor = board_to_tensor(board)
        move_idx = move_to_index(move)
        value = result_to_value(result, board.turn)

        positions.append({
            'board': board_tensor,
            'move': move_idx,
            'value': value,
            'fen': board.fen(),
            'san': board.san(move)
        })

        board.push(move)

    print(f"\n📊 Training data extracted:")
    print(f"   - {len(positions)} positions")
    print(f"   - Each has: board_tensor (12,8,8) + move_index + value")
    print(f"   - Model learns: 'In position X, play move Y, expected outcome Z'")

    return positions


def demo_fen_training_data():
    """Show what training data looks like from FEN positions."""
    print("\n\n" + "=" * 70)
    print("FEN TRAINING DATA EXAMPLE")
    print("=" * 70)

    # Sample positions with evaluations (simulated Stockfish evals)
    fen_data = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0.0),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", 0.3),
        ("rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2", 0.4),
        ("rnbqkb1r/pppppppp/5n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2", 0.2),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4", 0.6),
    ]

    print(f"\nProcessing {len(fen_data)} FEN positions with evaluations:\n")

    positions = []
    for idx, (fen, evaluation) in enumerate(fen_data, 1):
        print(f"Position {idx}:")
        print(f"  FEN: {fen}")
        print(f"  Stockfish eval: {evaluation:+.1f} pawns")
        print(f"  Move played: ??? (not provided)")

        # What goes into training
        board_tensor = fen_to_tensor(fen)

        positions.append({
            'board': board_tensor,
            'value': evaluation,
            'fen': fen
        })

    print(f"\n📊 Training data extracted:")
    print(f"   - {len(positions)} positions")
    print(f"   - Each has: board_tensor (12,8,8) + evaluation")
    print(f"   - Model learns: 'Position X is worth Y pawns'")
    print(f"   - Note: No move information! Need search to play.")

    return positions


def compare_model_outputs():
    """Compare what models trained on each format can do."""
    print("\n\n" + "=" * 70)
    print("MODEL COMPARISON: What can each model do?")
    print("=" * 70)

    test_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"

    print(f"\nTest position: {test_fen}")
    print("\n" + "-" * 70)
    print("PGN-TRAINED MODEL:")
    print("-" * 70)
    print("✅ Can predict move: model.predict_move(position)")
    print("   → Output: 'Nc3' (learned from training games)")
    print("\n✅ Can evaluate position: model.evaluate(position)")
    print("   → Output: +0.5 (based on game outcomes)")
    print("\n📝 Inference: FAST (direct policy prediction)")
    print("⚡ Can play immediately without search")

    print("\n" + "-" * 70)
    print("FEN-TRAINED MODEL:")
    print("-" * 70)
    print("❌ Cannot predict move directly (no policy head trained)")
    print("\n✅ Can evaluate position: model.evaluate(position)")
    print("   → Output: +0.62 (based on Stockfish)")
    print("\n📝 Inference: SLOW (needs search over all legal moves)")
    print("🐌 Must evaluate all legal moves and pick best:")
    print("   - O-O: eval = +0.58")
    print("   - Nc3: eval = +0.62  ← pick this")
    print("   - d3: eval = +0.45")
    print("   - ... (check all ~30 moves)")

    print("\n" + "-" * 70)
    print("HYBRID-TRAINED MODEL:")
    print("-" * 70)
    print("✅ Can predict move: model.predict_move(position)")
    print("   → Output: 'Nc3' (from PGN training)")
    print("\n✅ Can evaluate accurately: model.evaluate(position)")
    print("   → Output: +0.62 (from FEN fine-tuning)")
    print("\n📝 Inference: FAST (direct policy) + ACCURATE (refined value)")
    print("⚡ Best of both worlds!")


def show_data_statistics():
    """Show typical dataset sizes and characteristics."""
    print("\n\n" + "=" * 70)
    print("TYPICAL DATASET STATISTICS")
    print("=" * 70)

    print("\n📦 PGN Dataset (e.g., Lichess 2600+ Elo):")
    print("   - Games: 100,000")
    print("   - Positions: ~4,000,000 (avg 40 moves/game)")
    print("   - Size on disk: ~500MB (compressed PGN)")
    print("   - After preprocessing: ~50GB (numpy arrays)")
    print("   - Preparation time: ~1 hour")
    print("   - Training targets: Move + Outcome")
    print("   - Position diversity: Medium (common openings over-represented)")

    print("\n📦 FEN Dataset (e.g., Stockfish evaluations):")
    print("   - Positions: 1,000,000 (sampled uniformly)")
    print("   - Size on disk: ~100MB (text file)")
    print("   - After preprocessing: ~12GB (numpy arrays)")
    print("   - Preparation time: ~48 hours (Stockfish analysis!)")
    print("   - Training targets: Evaluation only")
    print("   - Position diversity: High (can sample rare positions)")


if __name__ == "__main__":
    print("\n🎯 CHESS TRAINING DATA FORMAT COMPARISON\n")

    # Run demos
    pgn_data = demo_pgn_training_data()
    fen_data = demo_fen_training_data()
    compare_model_outputs()
    show_data_statistics()

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    print("""
For a hackathon chess bot, here's what I recommend:

1. PRIMARY: Train on PGN (high-Elo games)
   ✅ Fast data preparation (just download)
   ✅ Can play moves immediately (no search needed)
   ✅ Learns strategic patterns
   ⚠️ Value estimates may be noisy

2. OPTIONAL: Fine-tune on FEN (if you have time)
   ✅ Improves evaluation accuracy
   ⚠️ Requires expensive Stockfish analysis
   💡 Use pre-computed evaluation datasets if available

3. IMPLEMENTATION:
   - Use preprocess.py with --format pgn for main training
   - Filter for 2400+ Elo games (quality > quantity)
   - Target: ~500k-1M positions (100-200MB processed data)
   - Training time: ~2-4 hours on GPU

4. FOR YOUR TEAM:
   - You: Focus on PGN pipeline (policy + value)
   - Teammate: Explore FEN for evaluation refinement
   - Combine later if both work well
""")
    print("=" * 70)
