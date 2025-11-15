# PGN vs FEN Training: Key Differences

## Data Characteristics

### PGN Training (Game Sequences)

**What you get**:
- Complete game trajectories from start to finish
- Move sequences with temporal relationships
- Game outcomes (1-0, 0-1, 1/2-1/2)
- Real human/engine gameplay

**Training targets**:
1. **Policy**: What move was played at this position
2. **Value**: Who won the game (from this position's perspective)

**Example data point**:
```python
Position: r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4
Move played: Nc3 (human/engine decision)
Game result: 1-0 (White won)
```

---

### FEN Training (Isolated Positions)

**What you get**:
- Independent positions (often from random points in games)
- Static evaluations (usually from Stockfish)
- No move context
- No temporal relationships

**Training targets**:
1. **Value only**: Engine's evaluation of the position (in centipawns or win probability)
2. **Policy**: Not available (unless you also include best move)

**Example data point**:
```python
Position: r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4
Evaluation: +0.65 (White slightly better)
Move: ??? (not provided)
```

---

## Training Objectives

### PGN: Imitation Learning + Outcome Prediction

**What the model learns**:
- "In this position, a 2800-rated player played Nc3"
- "Games where this position occurred, White won 65% of the time"
- Move patterns and typical continuations
- Strategic plans from game trajectories

**Loss functions**:
```python
# Policy loss: Cross-entropy
policy_loss = -log(P(move_played | position))

# Value loss: MSE
value_loss = (predicted_value - game_outcome)²

# Combined
total_loss = policy_loss + λ * value_loss
```

**Strengths**:
✅ Learns human/strong engine strategies
✅ Captures move sequences and plans
✅ Can learn both "what" and "why"
✅ Natural data from real games

**Weaknesses**:
❌ Biased by player styles (if using human games)
❌ Game outcome may not reflect position quality (one bad move can lose a good position)
❌ Temporal correlation (overfitting to game continuations)
❌ Imbalanced data (more common openings over-represented)

---

### FEN: Position Evaluation Learning

**What the model learns**:
- "This position is worth +0.65 pawns for White"
- Static evaluation without move context
- Position assessment from strongest engine

**Loss functions**:
```python
# Value loss only (typically)
value_loss = (predicted_eval - stockfish_eval)²
```

**Strengths**:
✅ Ground truth from strongest engines (Stockfish depth 20+)
✅ Unbiased evaluations
✅ Can sample diverse, rare positions
✅ Independent samples (no overfitting to game sequences)
✅ Precise positional understanding

**Weaknesses**:
❌ No move policy (can't suggest moves without additional data)
❌ Static evaluation may miss tactical dynamics
❌ Requires expensive engine analysis to generate
❌ Doesn't capture long-term strategic plans

---

## Practical Training Differences

### Model Architecture Implications

| Aspect | PGN Training | FEN Training |
|--------|--------------|--------------|
| **Output heads** | Policy (4096-dim) + Value (1-dim) | Value (1-dim) only* |
| **Training signal** | Supervised moves + outcomes | Supervised evaluations |
| **Data efficiency** | Lower (need full games) | Higher (any position works) |
| **Diversity** | Lower (common openings) | Higher (can sample uniformly) |

*Unless FEN file includes best moves

---

### Behavioral Differences

**PGN-trained model**:
```python
# Can do this:
position = get_position()
move = model.predict_move(position)  # Returns "Nc3"
eval = model.evaluate(position)       # Returns +0.65

# Behavior:
# - Makes moves similar to training games
# - May play popular lines even if suboptimal
# - Understands positional themes from game context
```

**FEN-trained model**:
```python
# Can only do this:
position = get_position()
eval = model.evaluate(position)  # Returns +0.65

# For move selection, need minimax search:
best_move = None
best_eval = -inf
for move in legal_moves:
    new_position = position.make_move(move)
    eval = model.evaluate(new_position)
    if eval > best_eval:
        best_eval = eval
        best_move = move

# Behavior:
# - More objective evaluation (not biased by training games)
# - Requires search to play (no direct policy)
# - Better at recognizing unusual positions
```

---

## Hybrid Approach (Best of Both Worlds)

### Combining PGN + FEN

**Strategy**:
1. Train on PGN for policy + rough value
2. Fine-tune on FEN for precise evaluations
3. Use both datasets simultaneously with different loss weights

**Implementation**:
```python
# Dataset 1: PGN games
for position, move, outcome in pgn_data:
    policy_loss = cross_entropy(model.policy(position), move)
    value_loss = mse(model.value(position), outcome)
    loss1 = policy_loss + value_loss

# Dataset 2: FEN positions with evals
for position, stockfish_eval in fen_data:
    value_loss = mse(model.value(position), stockfish_eval)
    loss2 = value_loss

# Combined training
total_loss = α * loss1 + β * loss2
```

**Why this works**:
- PGN provides move patterns (policy)
- FEN provides accurate evaluations (value)
- Model learns both strategic play and precise assessment

---

## Data Quality Considerations

### PGN Data Quality

**High-quality sources**:
- Lichess database (2600+ Elo games)
- Chess.com Master games
- Engine self-play (Stockfish vs Stockfish)

**Low-quality sources**:
- Random Lichess games (1200 Elo)
- Bullet/Blitz games (many blunders)

**Impact on training**:
```python
# Training on 2800 Elo games:
Model learns: "In this position, best move is Nc3"

# Training on 1500 Elo games:
Model learns: "In this position, players randomly play any move"
```

---

### FEN Data Quality

**High-quality sources**:
- Stockfish depth 20+ evaluations
- Lichess cloud evaluations
- curated puzzle datasets

**Low-quality sources**:
- Depth 5 evaluations (inaccurate)
- Positions from low-Elo games (not diverse)

**Impact on training**:
```python
# Deep engine analysis:
Position eval: +0.65 (accurate)

# Shallow analysis:
Position eval: +1.2 (missed tactic, inaccurate)
```

---

## Our Use Case: ChessHacks Bot

### Recommended Approach

**Option 1: PGN-Primary (What you're currently doing)**
```bash
# Use high-Elo games for policy + value
python preprocess.py \
  --input lichess_2600plus.pgn \
  --format pgn \
  --output data/pgn_processed
```

**Pros**: Can play moves directly, learns strategic patterns
**Cons**: Value estimates may be noisy (game outcomes)

---

**Option 2: FEN-Primary (What teammate is exploring)**
```bash
# Use engine evaluations for precise value
python preprocess.py \
  --input stockfish_positions.fen \
  --format fen \
  --with-evaluations \
  --output data/fen_processed
```

**Pros**: Precise position evaluation
**Cons**: No policy head (need search to play)

---

**Option 3: Hybrid (Recommended for hackathon)**
```python
# 1. Train initial model on PGN (gets policy + rough value)
train_on_pgn(
    data="lichess_2600plus.pgn",
    epochs=10,
    batch_size=256
)

# 2. Fine-tune value head on FEN (improves evaluation)
fine_tune_on_fen(
    data="stockfish_evals.fen",
    epochs=5,
    freeze_policy=True  # Only update value head
)

# 3. Result: Model with strong policy + accurate value
```

**Why hybrid**:
- You need policy to play moves quickly (no time for deep search in hackathon)
- You want accurate evaluation for position understanding
- Best of both worlds

---

## Time/Compute Tradeoffs

| Approach | Data Prep Time | Training Time | Inference Speed | Move Quality |
|----------|----------------|---------------|-----------------|--------------|
| PGN only | Fast (download) | Medium | ⚡ Fast | Good (imitation) |
| FEN only | Slow (engine) | Medium | 🐌 Slow (needs search) | Best (objective) |
| Hybrid | Medium | Longer | ⚡ Fast | Best (both) |

**For hackathon**:
- PGN is probably your best bet (faster data, can play immediately)
- Add FEN fine-tuning if you have time
- Ensure you're using 2400+ Elo games (quality matters!)

---

## Summary

**Key Takeaway**:
- **PGN** = Learn how strong players play (imitation learning)
- **FEN** = Learn what positions are good (evaluation learning)
- **Hybrid** = Learn both how to play AND how to evaluate

For a hackathon chess bot where you need to **make moves quickly** without expensive search, **PGN training with high-Elo games** is likely your best approach. If you have time, fine-tune on FEN to improve evaluation accuracy.
