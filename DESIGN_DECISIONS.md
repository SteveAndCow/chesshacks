# Chess Engine Design Decisions & Optimizations

**Last Updated:** 2025-11-15
**Focus:** ELO gains + Efficiency improvements for ChessHacks competition

---

## 🎯 Constraints & Goals

### Hard Constraints
- **Game time:** 1 minute total per game (~2-3s per move average)
- **Build time:** 3 minutes (CPU + GPU)
- **Must use NN:** Critical component (can't just use search)
- **Legal moves only:** Illegal move = disqualification

### Optimization Targets
1. **Maximize ELO:** Playing strength is #1 priority
2. **Fast inference:** <100ms NN forward pass for real-time MCTS
3. **Fast build:** Model + dependencies must build in <3 min

---

## 🧠 Model Architecture Decisions

### 1. Input Representation (High Impact on ELO)

**Current Standard:** 12-13 channels (pieces only)

**Recommended:** 16-20 channels
- Channels 0-11: Piece positions (6 types × 2 colors)
- Channel 12-13: Castling rights (kingside/queenside)
- Channel 14: En passant file
- Channel 15: Halfmove clock (50-move rule)
- **Optional Channel 16:** Side to move (all 1s or 0s)
- **Optional Channel 17-18:** Repetition count (draw detection)
- **Optional Channel 19:** Is check (immediate tactical alert)

**Why it helps:**
- Castling rights: +30-50 ELO (prevents illegal castle attempts)
- En passant: +10-20 ELO (correct tactical calculation)
- Halfmove clock: +5-10 ELO (endgame draw awareness)
- Is check: +20-40 ELO (prioritizes king safety)

**Tradeoff:** More channels = slightly slower inference (~5-10% slower)
**Verdict:** ✅ Use at least 16 channels, consider 20 for max strength

---

### 2. Output Representation (Medium Impact)

**Option A: From-To Encoding (4096 outputs)**
- Simple: `move_idx = from_square * 64 + to_square`
- Handles underpromotions poorly (treats e7e8=Q same as e7e8=R)
- Used by Minimal LCZero baseline

**Option B: Full Move Encoding (1858 outputs)**
- 64×64 from-to queen moves (4096)
- 64×3 underpromotions per pawn (knight, bishop, rook)
- Reduces to ~1858 legal chess moves
- Used by AlphaZero

**Option C: Action Head Encoding (73×8×8 outputs)**
- 73 "action types" per square (8 queen directions, knight moves, promotions)
- More complex but learns move patterns better
- Used by some ChessFormer variants

**Recommendation:**
- **For Minimal LCZero:** Start with Option A (simple, fast training)
- **For ChessFormers:** Try Option B or C (better expressiveness)
- **Optimization:** Mask illegal moves before softmax (huge ELO gain!)

**ELO Impact:**
- Illegal move masking: +100-200 ELO
- Better encoding (A→B): +20-50 ELO

---

### 3. Model Size vs. Inference Speed

**Critical Tradeoff:** Bigger model = stronger but slower

**Build Time Constraint (3 min):**
- Loading 100MB model from HF: ~10-20s
- Loading 500MB model from HF: ~60-90s (risky!)

**Inference Time Target:** <100ms per forward pass
- With 100 MCTS simulations: 100 × 100ms = 10s per move (too slow!)
- With batched inference (batch=16): 16 × 100ms / 16 = ~200-300ms total (acceptable)

**Model Size Recommendations:**

| Model Type | Params | Inference (CPU) | Inference (GPU) | Build Time | ELO Estimate |
|------------|--------|-----------------|-----------------|------------|--------------|
| **Tiny** | 1-3M | 20-40ms | 5-10ms | <30s | 1600-1800 |
| **Small** | 5-10M | 50-100ms | 10-20ms | 30-60s | 1800-2000 |
| **Medium** | 15-25M | 150-300ms | 20-40ms | 60-120s | 2000-2200 |
| **Large** | 40-60M | 400-800ms | 50-100ms | 120-180s | 2200-2400 |

**Recommendation:**
- **Slot 1:** Small model (fast, reliable)
- **Slot 2:** Medium model (balanced)
- **Slot 3:** Aggressive experiment (could be Tiny with better training or Large if build works)

**Key Insight:** A well-trained Small model > poorly-trained Large model!

---

### 4. Multi-Task Learning (High Impact)

**Standard:** Policy + Value heads

**Enhanced:** Policy + Value + Result + Auxiliary heads

**Recommended Heads:**
1. **Policy head:** Move probabilities (required)
2. **Value head:** Position evaluation [-1, 1] (required)
3. **Result head:** Win/Draw/Loss classification (strongly recommended)
4. **Move-left head:** Predict moves until game end (optional, helps endgame)
5. **Opponent-strength head:** Predict opponent ELO (helps adapt style)

**Loss Function:**
```python
total_loss = 1.0 * policy_loss + \
             1.0 * value_loss + \
             0.5 * result_loss + \
             0.2 * moves_left_loss
```

**ELO Impact:**
- Result head: +100-150 ELO (better endgame evaluation)
- Moves-left head: +30-60 ELO (improved time management)

**Verdict:** ✅ At minimum use Policy + Value + Result

---

## 🌲 MCTS Implementation Decisions

### 5. Search Algorithm Parameters

**PUCT Formula:** Standard UCB with policy guidance
```python
Q(s,a) = value from rollouts
U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
score = Q(s,a) + U(s,a)
```

**Critical Parameters:**

| Parameter | Conservative | Balanced | Aggressive |
|-----------|--------------|----------|------------|
| `c_puct` | 1.0 | 1.5-2.5 | 4.0+ |
| Simulations | 50-100 | 200-400 | 800-1600 |
| Dirichlet alpha | 0.15 | 0.3 | 0.5 |
| Dirichlet epsilon | 0.0 | 0.25 | 0.4 |

**Recommendations:**
- **Early game (moves 1-15):** Lower c_puct (trust policy more), use Dirichlet noise (exploration)
- **Mid game (moves 16-40):** Higher c_puct (search more), less noise
- **Late game (moves 41+):** Lower simulations (faster moves), precise evaluation
- **Time trouble (<10s left):** Drop to 50-100 simulations, fast moves

**ELO Impact:**
- Good c_puct tuning: +50-100 ELO
- Adaptive simulations: +30-50 ELO
- Dirichlet exploration: +20-40 ELO (prevents opening book memorization)

---

### 6. Virtual Loss & Parallelization

**Problem:** Sequential MCTS is slow (100 sims × 50ms = 5s)

**Solution:** Parallel MCTS with virtual loss

**Implementation:**
```python
# When selecting path to explore
add_virtual_loss(path, virtual_loss=3)
# This prevents other threads from exploring same path
# After rollout completes
remove_virtual_loss(path, virtual_loss=3)
backup_value(path, true_value)
```

**Parallel Workers:**
- **CPU:** 4-8 threads (most servers have 4+ cores)
- **GPU:** Batch inference 8-16 positions at once

**Speedup:** 4-8x faster search (critical for time control!)

**Verdict:** ✅ Essential for competitive play

---

### 7. Transposition Table / Caching

**Idea:** Cache NN evaluations for positions we've seen before

**Implementation:**
```python
cache = {}  # Dict[board_hash, (policy, value)]

def nn_eval(board):
    board_hash = hash(board.fen())
    if board_hash in cache:
        return cache[board_hash]

    policy, value = model(board)
    cache[board_hash] = (policy, value)
    return policy, value
```

**Benefits:**
- Faster search (skip duplicate NN calls)
- Better tree reuse between moves

**Memory:** ~1KB per position, 10k positions = 10MB (acceptable)

**ELO Impact:** +20-50 ELO (more simulations in same time)

**Verdict:** ✅ Easy win, definitely implement

---

## ⚡ Inference Optimization

### 8. Model Quantization

**Problem:** FP32 models are slow and large

**Solutions:**

| Precision | Size | Speed | ELO Loss |
|-----------|------|-------|----------|
| FP32 | 100% | 1.0x | 0 |
| FP16 | 50% | 2.0x | -5 to -10 |
| INT8 | 25% | 3-4x | -10 to -30 |
| INT4 | 12.5% | 5-8x | -50 to -100 |

**Recommendation:**
- Train in FP32
- Deploy in FP16 (best tradeoff)
- Avoid INT8 unless desperate for speed

**How to implement:**
```python
# PyTorch
model = model.half()  # Convert to FP16
input_tensor = input_tensor.half()
```

**Verdict:** ✅ Use FP16 for 2x speedup with minimal ELO loss

---

### 9. ONNX Runtime (Advanced)

**Idea:** Export PyTorch model to ONNX for faster inference

**Benefits:**
- 1.5-2x faster than PyTorch on CPU
- Better operator fusion
- Works on deployment servers without PyTorch

**Tradeoff:**
- Extra export step
- Potential build time increase
- Risk of export bugs

**Implementation:**
```python
# Export to ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# Load with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
```

**Verdict:** ⚠️ Only if you have time, test thoroughly

---

### 10. Batch Inference in MCTS

**Problem:** MCTS explores 1 position at a time → GPU underutilized

**Solution:** Collect N leaf nodes, run batch inference

**Implementation:**
```python
# Virtual loss approach
leaf_nodes = []
for _ in range(batch_size):
    leaf = select_leaf_with_virtual_loss(root)
    leaf_nodes.append(leaf)

# Batch inference
boards = [leaf.board for leaf in leaf_nodes]
policies, values = model.batch_forward(boards)

# Backup
for leaf, policy, value in zip(leaf_nodes, policies, values):
    backup(leaf, policy, value)
```

**Speedup:** 5-10x on GPU (critical!)

**Verdict:** ✅ Essential if using GPU

---

## 📊 Training Data & Strategy

### 11. Data Quality > Quantity

**Bad Dataset:** 10M games from Lichess 1200-2000 ELO
**Good Dataset:** 1M games from Lichess 2200+ ELO

**Why?**
- Low-ELO games teach bad patterns
- Model learns to blunder like humans
- Hard to unlearn later

**Filtering Recommendations:**
- Min ELO: 2200+ (strong players)
- Time control: Classical/Rapid only (no bullet/blitz - too many blunders)
- Remove games with ??/? (blunder) annotations
- Balance openings (don't overtrain on e4)

**ELO Impact:** Clean 1M > Dirty 10M (+200-400 ELO difference!)

---

### 12. Data Augmentation

**Chess Symmetry:** Board can be mirrored horizontally

**Implementation:**
```python
def augment(board, move):
    if random.random() < 0.5:
        board = mirror_horizontal(board)
        move = mirror_horizontal(move)
    return board, move
```

**Benefit:** 2x effective dataset size

**ELO Impact:** +30-50 ELO

**Caveat:** Don't augment opening positions (breaks opening theory)

**Verdict:** ✅ Easy win, definitely use

---

### 13. Self-Play (Advanced)

**AlphaZero Strategy:** Train on own games

**Pros:**
- Learns to exploit its own weaknesses
- Continuous improvement loop
- No need for external data

**Cons:**
- Requires good starting point (cold start problem)
- Computationally expensive
- May not converge in 36 hours

**Recommendation for Hackathon:**
1. Start with supervised learning on Lichess data (0-12 hours)
2. If strong enough (>1800 ELO), switch to self-play (12-36 hours)
3. Mix self-play data with human data (70% self, 30% human)

**Verdict:** ⚠️ Advanced, only attempt if base model is working well

---

## ⏱️ Time Management

### 14. Dynamic Time Allocation

**Simple Strategy:** Equal time per move (bad!)
- 60s / 40 moves = 1.5s per move
- Wastes time in simple positions, rushes in complex ones

**Better Strategy:** Complexity-based allocation

**Complexity Indicators:**
1. **Number of legal moves:** More moves = more complexity
2. **Policy entropy:** Uniform policy = unclear best move
3. **Evaluation change:** Position getting worse = think harder
4. **Material on board:** More pieces = more complex

**Algorithm:**
```python
def allocate_time(board, policy, value, time_left, moves_played):
    base_time = time_left / (60 - moves_played)  # Assume 60 move game

    # Complexity multipliers
    num_moves = len(board.legal_moves)
    move_factor = num_moves / 35  # 35 = average

    entropy = -sum(p * log(p) for p in policy)
    entropy_factor = entropy / 3.0  # Normalized

    # Critical moments
    if is_tactical(board):  # Checks, captures, threats
        critical_factor = 2.0
    else:
        critical_factor = 1.0

    time_for_move = base_time * move_factor * entropy_factor * critical_factor
    return min(time_for_move, time_left * 0.2)  # Never use >20% on one move
```

**ELO Impact:** +50-100 ELO (huge!)

**Verdict:** ✅ Critical for competitive play

---

### 15. Fast Opening Moves

**Idea:** Don't waste time on opening theory

**Options:**

**A) Opening Book (Traditional):**
- Precompute strong openings
- Instant moves for first 8-10 moves
- Risk: Predictable, can be exploited

**B) Quick NN Inference (Recommended):**
- Run minimal MCTS (20-50 sims) in opening
- Trust NN policy more (low c_puct)
- Still adaptive to opponent

**C) Hybrid:**
- Use book for first 3-4 moves (e4, d4, Nf3, etc.)
- Switch to quick MCTS after

**Time Saved:** 5-10s (can spend on critical middlegame!)

**ELO Impact:** +20-40 ELO (time management)

**Verdict:** ✅ Use option B or C

---

### 16. Endgame Tablebases (Optional)

**Idea:** Use precomputed perfect play for ≤6 pieces

**Syzygy Tablebases:**
- 3-4-5 piece: ~1GB (reasonable)
- 6 piece: ~150GB (too large for 3 min build)

**Recommendation:**
- Include 3-4-5 piece tablebases (~1GB)
- Instant perfect play in endgame
- Never lose drawn endgame, always win winning endgame

**ELO Impact:** +30-80 ELO (critical for close games!)

**Build Time:** +20-40s (tablebase download/extract)

**Verdict:** ✅ Include if build time allows

---

## 🏗️ Architecture-Specific Decisions

### 17. Minimal LCZero Optimizations

**Baseline:** ResNet-style CNN

**Recommended Improvements:**

1. **Squeeze-and-Excitation blocks:** Channel attention (+20-40 ELO)
2. **Increase residual blocks:** 10-20 blocks if budget allows
3. **Policy head architecture:** Use 1x1 conv instead of FC (+10-20 ELO)
4. **Value head architecture:** Separate WDL (win/draw/loss) heads (+50-100 ELO)

**Example Config:**
```yaml
model:
  blocks: 15  # More is better, up to 20
  filters: 192  # 128/192/256
  se_ratio: 8  # Squeeze-excitation
  policy_head: "conv"  # Not "fc"
  value_head: "wdl"  # Not "single"
```

---

### 18. ChessFormers Optimizations

**Baseline:** Transformer with absolute position encoding

**Recommended Improvements:**

1. **Relative Position Bias:** Learn chess geometry (+100-200 ELO!)
   - Bishops → diagonals
   - Rooks → files/ranks
   - Knights → L-shapes

2. **Piece-Type Attention:** Separate attention heads per piece type
   - One head focuses on pawn structure
   - One head focuses on king safety
   - One head focuses on piece coordination

3. **Efficient Attention:** Use flash attention or similar
   - 2-3x faster training
   - Same accuracy

**Example Config:**
```yaml
model:
  layers: 6  # Transformer layers
  heads: 8   # Attention heads
  dim: 256   # Model dimension
  use_relative_bias: true  # Critical!
  flash_attention: true    # If available
```

**Verdict:** ✅ Relative position bias is must-have for transformers

---

## 🎮 Practical Recommendations

### Priority Optimizations (Must Have)

**Training:**
1. ✅ High-quality data (2200+ ELO, classical time control)
2. ✅ Data augmentation (horizontal flip)
3. ✅ Multi-task learning (policy + value + result heads)
4. ✅ Input representation (16+ channels)

**Model:**
5. ✅ Illegal move masking
6. ✅ Right size for inference speed (Small or Medium)
7. ✅ FP16 quantization

**Search:**
8. ✅ Parallel MCTS with virtual loss
9. ✅ Transposition table / caching
10. ✅ Dynamic time management

**Total Expected ELO Gain:** +500-800 ELO over naive baseline

---

### Advanced Optimizations (If Time Permits)

**Training:**
- Self-play reinforcement learning
- Curriculum learning (easy → hard positions)
- Ensemble training (multiple models)

**Model:**
- ONNX Runtime export
- Batch inference optimization
- Custom CUDA kernels (extreme!)

**Search:**
- Adaptive search depth
- Pondering (think during opponent's turn)
- Singular extension (analyze forced lines deeper)

**Other:**
- Endgame tablebases (3-5 piece)
- Opening book integration
- Position evaluation caching

**Total Additional Gain:** +200-400 ELO

---

## 📋 Implementation Checklist

### Minimal LCZero Track
- [ ] Use ≥16 channel input representation
- [ ] Add result head (WDL classification)
- [ ] Mask illegal moves in policy output
- [ ] Train on 2200+ ELO data only
- [ ] Implement parallel MCTS (4-8 workers)
- [ ] Add transposition table
- [ ] Dynamic time allocation
- [ ] FP16 model export
- [ ] Fast opening moves (minimal sims)

### ChessFormers Track
- [ ] Use ≥16 channel input representation
- [ ] Enable relative position bias
- [ ] Add result head (WDL classification)
- [ ] Mask illegal moves in policy output
- [ ] Train on 2200+ ELO data only
- [ ] Implement parallel MCTS (4-8 workers)
- [ ] Add transposition table
- [ ] Dynamic time allocation
- [ ] FP16 model export
- [ ] Fast opening moves (minimal sims)

---

## 🔬 Experimental Ideas

### 1. Hybrid Value Estimation
Combine NN value with quick shallow minimax:
```python
nn_value = model.value(board)
minimax_value = minimax(board, depth=2)
final_value = 0.7 * nn_value + 0.3 * minimax_value
```
**Why:** NN is strategic, minimax catches tactics
**ELO:** +30-80 ELO (unproven but promising)

### 2. Move Pruning in MCTS
Don't explore moves with very low policy:
```python
legal_moves = board.legal_moves
policy_threshold = 0.01
pruned_moves = [m for m in legal_moves if policy[m] > policy_threshold]
```
**Why:** Focus search on promising moves
**ELO:** +20-50 ELO, but risk of missing tactics

### 3. Position Fingerprinting
Learn to recognize position types:
```python
position_type = classifier(board)
# Types: "opening", "tactical", "endgame", "quiet"
# Adjust search parameters accordingly
```
**Why:** Different positions need different search strategies
**ELO:** +40-80 ELO if done well

---

## 📊 Expected Performance Table

| Configuration | ELO Estimate | Inference Time | Build Time |
|---------------|--------------|----------------|------------|
| **Naive Baseline** | 1400-1600 | 200ms | <1 min |
| + Priority optimizations | 1900-2200 | 80ms | <2 min |
| + Advanced optimizations | 2200-2500 | 60ms | <3 min |
| + Self-play (risky) | 2400-2800 | 60ms | <3 min |

**Target for Queen's Crown:** 2200+ ELO (achievable with priority optimizations!)

---

*This document should be updated as we implement and test these ideas.*
