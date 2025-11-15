import math
import os

from tqdm import tqdm
import torch
from torch.utils.data import Dataset


def square_index(rank, file):
    return (rank - 1) * 8 + file


def fen_to_bitboards(fen):


    # initialize 12 planes of 8×8 with zeros
    planes = torch.zeros((12, 8, 8), dtype=torch.uint8)

    pieces = fen.split(" ")
    board = pieces[0]
    to_move = pieces[1]

    multiplier = 1
    if to_move == "w":
        pieces = ["P", "N", "B", "R", "Q", "K",
                  "p", "n", "b", "r", "q", "k"]
    else:
        multiplier = -1
        pieces = ["p", "n", "b", "r", "q",
                  "k", "P", "N", "B", "R", "Q", "K"]

    ranks = board.split("/")

    # ranks: fen[0] is rank 8 → row 0
    for row, rank in enumerate(ranks):
        col = 0
        for ch in rank:
            if ch.isdigit():
                col += int(ch)
            else:
                if ch in pieces:
                    idx = pieces.index(ch)
                    planes[idx][row][col] = 1
                col += 1

    return planes, multiplier

class StockfishDataset(Dataset):
    def __init__(self, bitboards, evaluations):

        self.bitboards = bitboards
        self.evaluations = evaluations

    def __len__(self):
        return len(self.bitboards)

    def __getitem__(self, idx):
        return self.bitboards[idx], self.evaluations[idx]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")

    args = parser.parse_args()

    bitboards = []
    evaluations = []

    with open(args.input_file) as f:
        for line in tqdm(f):
            parts = f.readline().strip("\n").split(',')
            bitboard, to_move = fen_to_bitboards(parts[0])
            bitboard = bitboard.clone().detach()
            value = parts[1]

            if value[0] == "#":
                evaluation = int(value[1:])
            else:
                evaluation = int(value)

            evaluation = evaluation * to_move

            evaluation = max(min(evaluation, 2^14), -2^14)

            bitboards.append(bitboard)
            evaluations.append(evaluation)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    torch.save(bitboards, args.output_file + "bitboards.pt")
    torch.save(evaluations, args.output_file + "evaluations.pt")