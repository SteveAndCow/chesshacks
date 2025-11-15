"""
Test FEN to bitboard conversion to verify correctness.
"""
import sys
from pathlib import Path

# Import from stockfishdataset
sys.path.insert(0, str(Path(__file__).parent))
from stockfishdataset import fen_to_bitboards, square_index


def print_bitboard(bitboard, piece_name):
    """Print bitboard in a human-readable 8x8 format."""
    print(f"\n{piece_name}:")
    for rank in range(7, -1, -1):  # rank 8 to 1
        row = []
        for file in range(8):  # a to h
            square = rank * 8 + file
            if bitboard & (1 << square):
                row.append('1')
            else:
                row.append('.')
        print(f"  {rank + 1} {''.join(row)}")
    print("    abcdefgh")


def test_starting_position():
    """Test with starting chess position."""
    print("="*60)
    print("TEST 1: Starting Position")
    print("="*60)

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    bitboards = fen_to_bitboards(fen)

    # Print all bitboards
    piece_names = {
        "P": "White Pawns", "N": "White Knights", "B": "White Bishops",
        "R": "White Rooks", "Q": "White Queen", "K": "White King",
        "p": "Black Pawns", "n": "Black Knights", "b": "Black Bishops",
        "r": "Black Rooks", "q": "Black Queen", "k": "Black King"
    }

    for piece, name in piece_names.items():
        print_bitboard(bitboards[piece], name)

    # Verify specific pieces
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    tests = [
        ("White King on e1", bitboards["K"], square_index(1, 4)),
        ("White Queen on d1", bitboards["Q"], square_index(1, 3)),
        ("Black King on e8", bitboards["k"], square_index(8, 4)),
        ("Black Queen on d8", bitboards["q"], square_index(8, 3)),
        ("White Pawn on e2", bitboards["P"], square_index(2, 4)),
        ("Black Pawn on e7", bitboards["p"], square_index(7, 4)),
    ]

    all_passed = True
    for test_name, bitboard, expected_square in tests:
        is_set = bool(bitboard & (1 << expected_square))
        status = "✅ PASS" if is_set else "❌ FAIL"
        print(f"{status} - {test_name} (square {expected_square})")
        if not is_set:
            all_passed = False

    return all_passed


def test_after_e4():
    """Test with position after 1.e4."""
    print("\n" + "="*60)
    print("TEST 2: After 1.e4")
    print("="*60)

    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    bitboards = fen_to_bitboards(fen)

    print_bitboard(bitboards["P"], "White Pawns (after e4)")

    # Verify white pawn is on e4, not e2
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    tests = [
        ("White Pawn on e4", bitboards["P"], square_index(4, 4), True),
        ("No White Pawn on e2", bitboards["P"], square_index(2, 4), False),
    ]

    all_passed = True
    for test_name, bitboard, square, should_be_set in tests:
        is_set = bool(bitboard & (1 << square))
        expected = "set" if should_be_set else "empty"
        actual = "set" if is_set else "empty"
        passed = (is_set == should_be_set)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name} (square {square}): expected {expected}, got {actual}")
        if not passed:
            all_passed = False

    return all_passed


def test_midgame_position():
    """Test with a midgame position."""
    print("\n" + "="*60)
    print("TEST 3: Midgame Position")
    print("="*60)

    # Position with pieces scattered
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1"
    bitboards = fen_to_bitboards(fen)

    print_bitboard(bitboards["K"], "White King")
    print_bitboard(bitboards["k"], "Black King")
    print_bitboard(bitboards["n"], "Black Knights")

    # Verify some pieces
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    tests = [
        ("White King on e1", bitboards["K"], square_index(1, 4)),
        ("Black King on e8", bitboards["k"], square_index(8, 4)),
        ("Black Knight on c6", bitboards["n"], square_index(6, 2)),
        ("Black Knight on f6", bitboards["n"], square_index(6, 5)),
        ("White Bishop on c4", bitboards["B"], square_index(4, 2)),
        ("Black Bishop on c5", bitboards["b"], square_index(5, 2)),
    ]

    all_passed = True
    for test_name, bitboard, expected_square in tests:
        is_set = bool(bitboard & (1 << expected_square))
        status = "✅ PASS" if is_set else "❌ FAIL"
        print(f"{status} - {test_name} (square {expected_square})")
        if not is_set:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("\nTESTING FEN TO BITBOARD CONVERSION\n")

    test1_passed = test_starting_position()
    test2_passed = test_after_e4()
    test3_passed = test_midgame_position()

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Test 1 (Starting Position): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (After e4): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Test 3 (Midgame): {'✅ PASSED' if test3_passed else '❌ FAILED'}")

    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 All tests PASSED! FEN to bitboard conversion is working correctly.")
    else:
        print("\n❌ Some tests FAILED. Check the conversion logic.")
