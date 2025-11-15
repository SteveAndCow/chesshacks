# Chess Data Format Comparison

## Overview

This document explains the three chess representation formats used in our training pipeline and how they relate to each other.

## Format Hierarchy

```
PGN (Game) → Multiple FEN (Positions) → Multiple Bitboards (Neural Network Input)
```

## Detailed Breakdown

### 1. PGN (Portable Game Notation)

**Purpose**: Store complete games with metadata and move sequences

**Example**:
```
[Event "Rated Blitz game"]
[Site "https://lichess.org/abc123"]
[White "Magnus"]
[Black "Hikaru"]
[Result "1-0"]
[WhiteElo "2800"]
[BlackElo "2750"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0
```

**Key Features**:
- Human-readable
- Contains full game history
- Includes metadata (players, ratings, event, etc.)
- Standard format for chess databases

**Storage**: `.pgn` text files

**Use Case**: Downloading high-Elo games from Lichess/Chess.com databases

---

### 2. FEN (Forsyth-Edwards Notation)

**Purpose**: Represent a single chess position

**Example**:
```
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
```

**Breakdown**:
```
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR  ← Board (rank 8 to 1)
b                                                 ← Active color (b/w)
KQkq                                              ← Castling rights
e3                                                ← En passant square
0                                                 ← Halfmove clock
1                                                 ← Fullmove number
```

**Key Features**:
- Compact single-position representation
- Contains game state information (castling, en passant)
- Can be extracted from any point in a PGN game
- Standard for position databases

**Storage**: `.txt` or `.fen` files (one position per line)

**Use Case**:
- Position-based datasets (e.g., Stockfish evaluations)
- Quick position lookup
- Testing specific positions

---

### 3. Bitboard Tensor

**Purpose**: Neural network input representation

**Shape**: `(12, 8, 8)` numpy array

**Structure**:
```
12 channels (planes), each 8×8:
  [0] White Pawns       [6] Black Pawns
  [1] White Knights     [7] Black Knights
  [2] White Bishops     [8] Black Bishops
  [3] White Rooks       [9] Black Rooks
  [4] White Queens     [10] Black Queens
  [5] White King       [11] Black King
```

**Example** (starting position, White Pawns plane):
```python
array([[0, 0, 0, 0, 0, 0, 0, 0],  # Rank 8
       [0, 0, 0, 0, 0, 0, 0, 0],  # Rank 7
       [0, 0, 0, 0, 0, 0, 0, 0],  # Rank 6
       [0, 0, 0, 0, 0, 0, 0, 0],  # Rank 5
       [0, 0, 0, 0, 0, 0, 0, 0],  # Rank 4
       [0, 0, 0, 0, 0, 0, 0, 0],  # Rank 3
       [1, 1, 1, 1, 1, 1, 1, 1],  # Rank 2 (White Pawns)
       [0, 0, 0, 0, 0, 0, 0, 0]]) # Rank 1
```

**Key Features**:
- Direct CNN input format
- Efficient GPU computation
- Clear spatial relationships
- Float32 for neural network compatibility

**Storage**: `.npy` files (NumPy arrays)

**Use Case**: Direct training input for neural networks

---

## Conversion Pipeline

### PGN → Bitboard Tensors
```bash
python preprocess.py \
  --input games.pgn \
  --format pgn \
  --output data/processed \
  --max-games 10000
```

**What happens**:
1. Parse PGN file
2. For each game, iterate through moves
3. At each position, extract:
   - Board state → Bitboard tensor (12, 8, 8)
   - Move played → Move index
   - Game result → Value (-1, 0, 1)
4. Save as NumPy arrays

**Output files**:
- `boards.npy`: Shape `(N, 12, 8, 8)`
- `moves.npy`: Shape `(N,)` - move indices
- `values.npy`: Shape `(N,)` - game outcomes

---

### FEN → Bitboard Tensors
```bash
python preprocess.py \
  --input positions.fen \
  --format fen \
  --output data/processed \
  --with-evaluations \
  --max-positions 50000
```

**What happens**:
1. Parse FEN file (one position per line)
2. For each FEN:
   - Parse position → Bitboard tensor (12, 8, 8)
   - Extract evaluation (if provided)
3. Save as NumPy arrays

**Output files**:
- `boards.npy`: Shape `(N, 12, 8, 8)`
- `values.npy`: Shape `(N,)` - evaluations

---

## Alternative Bitboard Representation

Your teammate's `stockfishdataset.py` uses **bitwise integers**:

```python
bitboards = {
    "P": 0b0000000000000000000000000000000000000000000000001111111100000000,  # White Pawns
    "N": 0b0000000000000000000000000000000000000000000000000000000001000010,  # White Knights
    # ... etc
}
```

**Comparison**:

| Feature | 3D Tensor (preprocess.py) | Bitwise (stockfishdataset.py) |
|---------|---------------------------|-------------------------------|
| Shape | `(12, 8, 8)` numpy array | 12 separate 64-bit integers |
| Memory | ~3KB per position | ~96 bytes per position |
| CNN-ready | ✅ Yes | ❌ Needs conversion |
| Speed | Fast NumPy ops | Fast bitwise ops |
| Use case | Direct training | Efficient storage/logic |

**Conversion between them**:
```python
# Bitwise → Tensor (what we should add)
def bitboards_to_tensor(bitboards: dict) -> np.ndarray:
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_to_idx = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}

    for piece, bitboard in bitboards.items():
        idx = piece_to_idx[piece]
        for square in range(64):
            if bitboard & (1 << square):
                rank = square // 8
                file = square % 8
                tensor[idx, rank, file] = 1.0

    return tensor
```

---

## Usage Examples

### Processing High-Elo Games from Lichess

```bash
# Step 1: Download PGN
# (already done in download_games.py)

# Step 2: Convert to training data
python preprocess.py \
  --input ../data/lichess_db_standard_rated_2024-01.pgn \
  --format pgn \
  --output ../data/processed/lichess_2024_01 \
  --max-games 100000
```

### Processing Stockfish Evaluated Positions

```bash
# If you have a FEN file with evaluations:
# Format: "fen_string evaluation"

python preprocess.py \
  --input stockfish_positions.txt \
  --format fen \
  --with-evaluations \
  --output ../data/processed/stockfish_evals
```

### Quick Testing on Single Position

```python
from preprocess import fen_to_tensor

# Test a specific position
fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
tensor = fen_to_tensor(fen)
print(tensor.shape)  # (12, 8, 8)
```

---

## Summary

**PGN** → **FEN** → **Bitboard** represents the data pipeline from raw games to neural network input:

1. **PGN**: Download from databases (Lichess, Chess.com)
2. **FEN**: Optional intermediate format for position-based datasets
3. **Bitboard**: Final neural network training format

All three formats represent the same information at different levels of abstraction, optimized for different use cases. The `preprocess.py` script now handles both PGN and FEN inputs and outputs unified bitboard tensors ready for training.
