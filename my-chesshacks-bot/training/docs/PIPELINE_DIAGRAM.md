# Complete Data Pipeline Diagram

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW CHESS DATA                              │
│                                                                 │
│  ┌──────────────┐              ┌──────────────┐                │
│  │   PGN Files  │              │  FEN Files   │                │
│  │              │              │              │                │
│  │ Full games   │              │ Positions    │                │
│  │ with moves   │              │ with evals   │                │
│  └──────┬───────┘              └───────┬──────┘                │
│         │                              │                       │
└─────────┼──────────────────────────────┼───────────────────────┘
          │                              │
          │                              │
          ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PREPROCESSING LAYER                            │
│                   (preprocess.py)                               │
│                                                                 │
│  ┌──────────────────────┐    ┌────────────────────────┐        │
│  │  process_pgn_file()  │    │ process_fen_file()     │        │
│  │                      │    │                        │        │
│  │  • Parse games       │    │ • Parse positions      │        │
│  │  • Extract positions │    │ • Extract evaluations  │        │
│  │  • Get moves played  │    │ • (No moves)           │        │
│  │  • Get game outcomes │    │                        │        │
│  └──────────┬───────────┘    └───────────┬────────────┘        │
│             │                            │                     │
│             └────────────┬───────────────┘                     │
│                          │                                     │
│                          ▼                                     │
│             ┌────────────────────────┐                         │
│             │  board_to_tensor()     │                         │
│             │                        │                         │
│             │  Convert to (12,8,8)   │                         │
│             │  bitboard tensor       │                         │
│             └────────────┬───────────┘                         │
└──────────────────────────┼─────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED OUTPUT FORMAT                        │
│                                                                 │
│  ┌────────────────┐     ┌────────────────┐  ┌────────────────┐│
│  │  boards.npy    │     │  moves.npy     │  │  values.npy    ││
│  │                │     │  (PGN only)    │  │                ││
│  │  (N, 12, 8, 8) │     │  (N,)          │  │  (N,)          ││
│  │                │     │                │  │                ││
│  │  Board states  │     │  Move indices  │  │  Outcomes/Evals││
│  └────────┬───────┘     └────────┬───────┘  └────────┬───────┘│
└───────────┼──────────────────────┼──────────────────┼─────────┘
            │                      │                  │
            └──────────────────────┴──────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NEURAL NETWORK                             │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                  Input Layer                           │    │
│  │              (12, 8, 8) tensor                         │    │
│  └─────────────────────┬──────────────────────────────────┘    │
│                        │                                       │
│                        ▼                                       │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              CNN Feature Extractor                     │    │
│  │   Conv2D → ReLU → Conv2D → ReLU → ...                 │    │
│  └─────────────────────┬──────────────────────────────────┘    │
│                        │                                       │
│                        ▼                                       │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                  Two Heads                             │    │
│  │                                                        │    │
│  │  ┌──────────────────────┐  ┌──────────────────────┐  │    │
│  │  │   Policy Head        │  │   Value Head         │  │    │
│  │  │   (4096 outputs)     │  │   (1 output)         │  │    │
│  │  │                      │  │                      │  │    │
│  │  │   "What move?"       │  │   "How good?"        │  │    │
│  │  │   Nc3, O-O, d4...    │  │   +0.65 pawns        │  │    │
│  │  └──────────────────────┘  └──────────────────────┘  │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Format Comparison

### PGN Data Flow
```
Input: lichess_games.pgn
├─ Game 1: "1.e4 e5 2.Nf3 Nc6 3.Bb5" [Result "1-0"]
│  ├─ Position 1 (starting)     → board_tensor_1, move="e4",   value=+1.0
│  ├─ Position 2 (after 1.e4)   → board_tensor_2, move="e5",   value=-1.0
│  ├─ Position 3 (after 1...e5) → board_tensor_3, move="Nf3",  value=+1.0
│  └─ ... (40 positions total)
├─ Game 2: ...
└─ Game N: ...

Output:
  boards.npy:  [board_tensor_1, board_tensor_2, ...] shape (N, 12, 8, 8)
  moves.npy:   [move_idx_1, move_idx_2, ...]         shape (N,)
  values.npy:  [+1.0, -1.0, +1.0, ...]                shape (N,)
```

### FEN Data Flow
```
Input: stockfish_positions.txt
├─ "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1" 0.30
│  └─ Position 1 → board_tensor_1, value=+0.30
├─ "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4" 0.65
│  └─ Position 2 → board_tensor_2, value=+0.65
└─ ... (N positions)

Output:
  boards.npy:  [board_tensor_1, board_tensor_2, ...] shape (N, 12, 8, 8)
  values.npy:  [0.30, 0.65, ...]                     shape (N,)
  (no moves.npy!)
```

---

## Bitboard Tensor Structure

```
Shape: (12, 8, 8)
│
├─ Plane 0: White Pawns     ┌─ Rank 8 (top, black's back rank)
│    ┌─────────────────────┐│
│    │ 0 0 0 0 0 0 0 0 │ 8││
│    │ 0 0 0 0 0 0 0 0 │ 7││
│    │ 0 0 0 0 0 0 0 0 │ 6││
│    │ 0 0 0 0 0 0 0 0 │ 5││
│    │ 0 0 0 0 0 0 0 0 │ 4││
│    │ 0 0 0 0 0 0 0 0 │ 3││
│    │ 1 1 1 1 1 1 1 1 │ 2││ ← White pawns on rank 2
│    │ 0 0 0 0 0 0 0 0 │ 1││
│    └─────────────────────┘│
│      a b c d e f g h      │
│                           └─ Rank 1 (bottom, white's back rank)
├─ Plane 1: White Knights
│    ┌─────────────────────┐
│    │ 0 0 0 0 0 0 0 0 │  │
│    │ 0 0 0 0 0 0 0 0 │  │
│    │ 0 0 0 0 0 0 0 0 │  │
│    │ 0 0 0 0 0 0 0 0 │  │
│    │ 0 0 0 0 0 0 0 0 │  │
│    │ 0 0 0 0 0 0 0 0 │  │
│    │ 0 0 0 0 0 0 0 0 │  │
│    │ 0 1 0 0 0 0 1 0 │  │ ← Knights on b1, g1
│    └─────────────────────┘
│
├─ Plane 2-5: White B, R, Q, K
├─ Plane 6-11: Black pieces (same structure)
│
└─ Each plane is 8×8 binary matrix
   1 = piece present, 0 = empty
```

---

## Training Loss Functions

### PGN Training (Two objectives)
```python
# Input
X = board_tensor  # (12, 8, 8)

# Targets
y_policy = move_index  # int in [0, 4095]
y_value = outcome      # float in {-1, 0, 1}

# Forward pass
policy_logits, value_pred = model(X)

# Losses
policy_loss = CrossEntropy(policy_logits, y_policy)
value_loss = MSE(value_pred, y_value)

total_loss = policy_loss + λ * value_loss
```

### FEN Training (One objective)
```python
# Input
X = board_tensor  # (12, 8, 8)

# Target
y_value = evaluation  # float (e.g., +0.65 pawns)

# Forward pass
value_pred = model.value_head(model.features(X))

# Loss
value_loss = MSE(value_pred, y_value)

total_loss = value_loss
```

---

## Inference Comparison

### PGN-Trained Model
```python
# Direct move prediction (FAST)
position = get_current_position()
tensor = board_to_tensor(position)

policy, value = model(tensor)
best_move = argmax(policy)  # O(1) - instant!

play(best_move)
```

### FEN-Trained Model
```python
# Must search over moves (SLOW)
position = get_current_position()
legal_moves = get_legal_moves(position)  # ~30 moves

best_eval = -inf
best_move = None

for move in legal_moves:  # O(N) - must check all moves!
    new_position = position.make_move(move)
    tensor = board_to_tensor(new_position)
    eval = model.value_head(model.features(tensor))

    if eval > best_eval:
        best_eval = eval
        best_move = move

play(best_move)
```

**Speed difference**: PGN model is ~30x faster at move selection!

---

## Recommended Pipeline for ChessHacks

```
┌─────────────────────────────────────────────────────────────┐
│                    WEEK 1: PGN Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Download Games                                          │
│     python download_games.py --min-elo 2400                 │
│                                                             │
│  2. Preprocess                                              │
│     python preprocess.py \                                  │
│       --input games.pgn \                                   │
│       --format pgn \                                        │
│       --output data/train \                                 │
│       --max-games 50000                                     │
│                                                             │
│  3. Train CNN                                               │
│     python train_modal.py --data data/train                 │
│                                                             │
│  4. Test Bot                                                │
│     python play.py --model trained_model.pt                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│             WEEK 2: Optional FEN Fine-tuning                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Get FEN evaluations                                     │
│     (Use pre-computed dataset or run Stockfish)             │
│                                                             │
│  2. Preprocess FEN                                          │
│     python preprocess.py \                                  │
│       --input evals.fen \                                   │
│       --format fen \                                        │
│       --with-evaluations \                                  │
│       --output data/finetune                                │
│                                                             │
│  3. Fine-tune value head                                    │
│     python train_modal.py \                                 │
│       --data data/finetune \                                │
│       --checkpoint trained_model.pt \                       │
│       --freeze-policy                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **All formats represent the same chess information**
   - PGN = sequence of positions
   - FEN = single position
   - Bitboard = neural network representation

2. **PGN gives you both policy and value**
   - Can play moves directly (fast)
   - Learn from strong players

3. **FEN gives you precise evaluations**
   - Value only (no moves)
   - Need search to play (slow)

4. **Your preprocessing is now unified**
   - `preprocess.py` handles both
   - Same output format: (12, 8, 8) tensors
   - Compatible with CNN training

5. **For hackathon: Use PGN**
   - Faster to prepare
   - Can play immediately
   - Good enough for competition

6. **Teammate's bitwise approach is equivalent**
   - Just a different representation
   - Both convert to same logical structure
   - Yours is more CNN-friendly
