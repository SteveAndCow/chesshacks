# Quick Start Guide: Understanding & Using Our Preprocessing Pipeline

## TL;DR - The Key Difference

**PGN vs FEN vs Bitboard**:
```
PGN (Game Format)          FEN (Position Format)        Bitboard (NN Input)
     ↓                            ↓                            ↓
"1.e4 e5 2.Nf3..."         "rnbqkbnr/pppp..."         [12, 8, 8] tensor
(Full game)                (Single position)           (Neural network ready)
```

**Training difference**:
- **PGN Training**: Model learns "what move to play" + "who's winning" → Can play directly
- **FEN Training**: Model learns "how good is this position" → Needs search to play
- **Your current pipeline**: PGN → Multiple positions → Bitboards → Train move + value

---

## Setup (One-time)

```bash
# Make sure you're in the right directory
cd /Users/stephencao/projects/chesshacks/my-chesshacks-bot

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import chess, numpy, torch; print('✅ All dependencies installed')"
```

---

## Understanding the Formats

### Run the comparison demo:
```bash
cd training/scripts
python compare_formats.py
```

This will show you:
1. What training data looks like from PGN files
2. What training data looks like from FEN files
3. What each trained model can do
4. Practical recommendations

---

## Processing Your Data

### Option 1: Process PGN files (Recommended for hackathon)

```bash
# Process high-Elo games from Lichess
python training/scripts/preprocess.py \
  --input path/to/lichess_games.pgn \
  --format pgn \
  --output training/data/processed/pgn_data \
  --max-games 10000

# Output:
# - boards.npy: (N, 12, 8, 8) - board positions
# - moves.npy: (N,) - moves played
# - values.npy: (N,) - game outcomes
```

**What you get**: ~40 positions per game = 400k training examples from 10k games

---

### Option 2: Process FEN files (For evaluation refinement)

```bash
# Process FEN positions with Stockfish evaluations
python training/scripts/preprocess.py \
  --input path/to/stockfish_positions.txt \
  --format fen \
  --output training/data/processed/fen_data \
  --with-evaluations \
  --max-positions 50000

# Output:
# - boards.npy: (N, 12, 8, 8) - board positions
# - values.npy: (N,) - stockfish evaluations
```

**What you get**: Direct position evaluations, no move information

---

## Team Workflow

### Your approach (PGN-based):
```bash
# 1. Download high-Elo games (you have this)
python training/scripts/download_games.py --min-elo 2400

# 2. Preprocess to training format
python training/scripts/preprocess.py \
  --input ../data/downloaded_games.pgn \
  --format pgn \
  --output ../data/processed/main_training

# 3. Train neural network (use data_loader.py)
python training/scripts/train_modal.py \
  --data ../data/processed/main_training
```

### Teammate's approach (FEN-based):
```bash
# They're working with FEN → bitboard conversion
# Their stockfishdataset.py converts FEN to bitwise integers
# Our preprocess.py now supports this too!

python training/scripts/preprocess.py \
  --input stockfish_dataset.txt \
  --format fen \
  --with-evaluations \
  --output ../data/processed/fen_evals
```

### Unified format:
Both approaches now produce the **same output format**:
- `boards.npy`: (N, 12, 8, 8) numpy arrays ✅
- Ready for CNN training ✅

---

## Key Insights

### 1. All three formats represent the same thing:

```python
# These all describe the SAME chess position after 1.e4:

# PGN format:
"1. e4"

# FEN format:
"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

# Bitboard format (White Pawns plane):
[[0,0,0,0,0,0,0,0],  # rank 8
 [0,0,0,0,0,0,0,0],  # rank 7
 [0,0,0,0,0,0,0,0],  # rank 6
 [0,0,0,0,0,0,0,0],  # rank 5
 [0,0,0,0,1,0,0,0],  # rank 4 (pawn on e4!)
 [0,0,0,0,0,0,0,0],  # rank 3
 [1,1,1,1,0,1,1,1],  # rank 2 (e2 pawn moved to e4)
 [0,0,0,0,0,0,0,0]]  # rank 1
```

### 2. Bitboard has 12 planes:
Your `preprocess.py` already does this correctly:
```python
Plane 0:  White Pawns      Plane 6:  Black Pawns
Plane 1:  White Knights    Plane 7:  Black Knights
Plane 2:  White Bishops    Plane 8:  Black Bishops
Plane 3:  White Rooks      Plane 9:  Black Rooks
Plane 4:  White Queen      Plane 10: Black Queen
Plane 5:  White King       Plane 11: Black King
```

### 3. Training objectives differ:

**From PGN** (`preprocess.py:162-226`):
```python
for each position in game:
    X = board_state  # Input
    y_policy = move_played  # Target 1: What move did strong player make?
    y_value = game_outcome  # Target 2: Who won?
```

**From FEN** (`preprocess.py:98-160`):
```python
for each position:
    X = board_state  # Input
    y_value = stockfish_eval  # Target: How good is position?
    # Note: No y_policy!
```

---

## Testing Your Setup

```bash
# Test 1: Verify preprocessing works
cd training/scripts
python -c "
from preprocess import fen_to_tensor
import numpy as np

fen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1'
tensor = fen_to_tensor(fen)
print(f'✅ Tensor shape: {tensor.shape}')
print(f'✅ Expected: (12, 8, 8)')
assert tensor.shape == (12, 8, 8), 'Shape mismatch!'
print('✅ FEN conversion working!')
"

# Test 2: Run comparison demo
python compare_formats.py

# Test 3: Process a small sample
# (Use a small PGN file for testing)
python preprocess.py \
  --input test_sample.pgn \
  --format pgn \
  --output /tmp/test_output \
  --max-games 10
```

---

## Next Steps

1. **Verify your data pipeline**:
   ```bash
   python compare_formats.py  # Understand the differences
   ```

2. **Process your high-Elo games**:
   ```bash
   python preprocess.py \
     --input ../data/lichess_2600plus.pgn \
     --format pgn \
     --output ../data/processed/training_set \
     --max-games 50000
   ```

3. **Train your model** (existing train_modal.py):
   - Load `boards.npy` as X
   - Load `moves.npy` as y_policy
   - Load `values.npy` as y_value
   - Train CNN with two heads

4. **Optional: Add FEN fine-tuning**:
   - First train on PGN (step 3)
   - Then fine-tune value head on FEN data
   - Freeze policy head, only update value head

---

## Troubleshooting

**Q: Which format should I use?**
A: For hackathon, use PGN. It's faster and gives you both policy and value.

**Q: My teammate is using bitwise integers, should I switch?**
A: No! Your (12,8,8) tensor format is better for CNNs. Both are logically equivalent.

**Q: Where do I get high-Elo games?**
A: Your `download_games.py` script already does this! Use Lichess database.

**Q: How do I combine PGN and FEN data?**
A: Train on PGN first, then fine-tune on FEN (see docs/PGN_VS_FEN_TRAINING.md)

**Q: What's the difference in model behavior?**
A:
- PGN-trained: Can predict moves directly (fast inference)
- FEN-trained: Must search over moves (slow inference)
- Use PGN for hackathon!

---

## Summary

You now have a **unified preprocessing pipeline** that handles:
- ✅ PGN files (full games) → policy + value training
- ✅ FEN files (positions) → value-only training
- ✅ Both output the same format: (12, 8, 8) bitboard tensors
- ✅ Compatible with your existing CNN architecture

**Recommended for hackathon**: Stick with PGN training on 2400+ Elo games!
