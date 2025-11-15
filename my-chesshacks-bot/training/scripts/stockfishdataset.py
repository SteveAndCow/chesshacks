import math


def square_index(rank, file):
    return (rank - 1) * 8 + file


def fen_to_bitboards(fen):
    piece_order = ["P", "N", "B", "R", "Q", "K",
                   "p", "n", "b", "r", "q", "k"]
    bitboards = {p: 0 for p in piece_order}

    board = fen.split()[0]     # only the board portion
    ranks = board.split("/")

    for r, rank in enumerate(reversed(ranks), start=1):  # r=1 → rank 1
        file = 0
        for ch in rank:
            if ch.isdigit():
                file += int(ch)
            else:
                if ch in bitboards:
                    sq = square_index(r, file)
                    bitboards[ch] |= (1 << sq)
                file += 1

    return bitboards


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the dataset file")
    args = parser.parse_args()

    bitboards = []
    evaluations = []

    with open(args.file) as f:
        for line in f:
            parts = f.readline().split()
            bitboard = fen_to_bitboards(parts[0])
            value = parts[1]

            if value is "#":
                evaluation = math.inf
            else:
                evaluation = int(value)

            bitboards.append(bitboard)
            evaluations.append(evaluation)

fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
bitboards = fen_to_bitboards(fen)