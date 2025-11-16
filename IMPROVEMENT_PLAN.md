# ChessHacks Bot - Comprehensive Improvement Plan

**Date:** 2025-11-16
**Current Model:** LC0 128x6 (1.5M samples, 15 epochs, 256 filters)
**Current Issues:** Overfitting, timeout losses, limited training diversity

---

## 📋 Executive Summary

Based on analysis of the current codebase, we've identified critical improvements across three main areas:

1. **Training Pipeline Overfitting** - Model overfits on limited data
2. **Move Generation Speed** - Timeout losses due to slow inference
3. **Self-Play Implementation** - Missing capability for continuous improvement

Additionally, we address two strategic concerns:
- Training data quality (handling blunders)
- Dataset scaling for larger final models

---

## 🔴 CRITICAL ISSUE #1: LC0 Model Overfitting

### Root Cause Analysis

**Current Training Configuration (`train_modal_lc0_fixed.py`):**
- Dataset: 1.5M positions from 2200+ ELO games
- Epochs: 15
- Batch size: 256 (mentioned 512 in user requirements)
- Train/val split: 95%/5%
- **CRITICAL BUG:** Line 219-220 limits to 1000 batches per epoch
  - 1000 batches × 256 = 256k samples per epoch
  - Actual training: 15 epochs × 256k = seeing same 256k positions 15 times!
  - Total dataset: 1.5M positions, but only using 17% per epoch
- Weight decay: 0.0001 (very low)
- No dropout in model architecture
- No data augmentation
- CosineAnnealingLR scheduler (appropriate)

**Overfitting Indicators:**
1. **Limited data exposure:** Only 256k/1.5M positions used per epoch
2. **High epoch count:** 15 epochs on same subset = severe overfitting
3. **Minimal regularization:** No dropout, minimal weight decay
4. **No augmentation:** Missing easy 2x data boost from horizontal flip
5. **High train/val split:** 95% training leaves little validation data

### Specific Code Issues

**File:** `/home/user/chesshacks/training/scripts/train_modal_lc0_fixed.py`

**Issue 1 - Artificial epoch limit (Lines 218-220):**
```python
# Limit to 1000 batches per epoch for faster iteration during hackathon
if num_batches >= 1000:
    break
```
**Impact:** Model sees only 17% of data, then repeats 15 times → severe overfitting

**Issue 2 - No dropout (models/lccnn.py):**
```python
class LeelaZeroNet(pl.LightningModule):
    def __init__(...):
        # No dropout layers anywhere!
        self.residual_blocks = nn.Sequential(residual_blocks)
```
**Impact:** No stochastic regularization during training

**Issue 3 - Low weight decay (Line 157):**
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.0001  # Too low for preventing overfitting
)
```
**Impact:** Minimal L2 regularization

**Issue 4 - No data augmentation (data_loader_lc0.py):**
- No horizontal flipping (easy 2x data boost)
- No position rotation/mirroring
- No noise injection

---

### Solutions - Priority Order

#### 🔥 IMMEDIATE (Must fix before next training run)

**1. Remove artificial epoch limit**
```python
# training/scripts/train_modal_lc0_fixed.py:218-220
# DELETE these lines:
# if num_batches >= 1000:
#     break

# Replace with dynamic limit based on dataset size:
max_batches_per_epoch = min(10000, total_dataset_size // batch_size)
if num_batches >= max_batches_per_epoch:
    break
```
**Expected improvement:** +150-200 ELO from seeing full dataset

**2. Reduce epochs, increase data exposure**
```python
# train_modal_lc0_fixed.py
# Change from:
num_epochs: int = 10  # Default was 10, user mentioned 15

# To:
num_epochs: int = 5  # Fewer epochs, but see ALL data each time
```
**Expected improvement:** Better generalization, +50-100 ELO

**3. Add dropout to residual blocks**
```python
# training/scripts/models/pt_layers.py - ResidualBlock class
class ResidualBlock(nn.Module):
    def __init__(self, channels, se_ratio, dropout=0.1):
        super().__init__()
        # ... existing conv layers ...
        self.dropout = nn.Dropout2d(p=dropout)  # Add dropout

    def forward(self, inputs):
        out1 = self.conv1(inputs)
        out1 = F.relu(self.batch_norm(out1.float()))
        out1 = self.dropout(out1)  # Apply after activation
        out2 = self.conv2(out1)
        out2 = self.squeeze_excite(out2)
        return F.relu(inputs + out2)
```
**Expected improvement:** +30-60 ELO from better regularization

**4. Increase weight decay**
```python
# train_modal_lc0_fixed.py:157
weight_decay=0.0001  # Change to:
weight_decay=0.0005  # 5x stronger L2 regularization
```
**Expected improvement:** +20-40 ELO

#### ⚠️ HIGH PRIORITY (Implement this week)

**5. Add data augmentation**
```python
# training/scripts/data_loader_lc0.py - Add augmentation in dataset
def augment_position(inputs, policy_target, flip_horizontal=True):
    """
    Augment chess position with horizontal flip.

    Args:
        inputs: (112, 8, 8) board representation
        policy_target: (1858,) policy target
        flip_horizontal: Whether to flip board horizontally

    Returns:
        Augmented inputs and policy target
    """
    if not flip_horizontal or random.random() < 0.5:
        return inputs, policy_target

    # Flip board horizontally (files a-h → h-a)
    inputs_flipped = inputs[:, :, ::-1].copy()

    # Remap policy indices (complex - need LC0 policy map)
    # For now, simple approximation:
    # This needs proper implementation with lc0_policy_map
    policy_flipped = remap_policy_horizontal(policy_target)

    return inputs_flipped, policy_flipped
```
**Expected improvement:** 2x effective dataset, +50-100 ELO

**6. Early stopping with patience**
```python
# train_modal_lc0_fixed.py - Add early stopping
best_val_loss = float('inf')
patience = 3
patience_counter = 0

for epoch in range(num_epochs):
    # ... training ...

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save checkpoint
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
```
**Expected improvement:** Prevents overfitting late in training, +30-50 ELO

#### ✅ MEDIUM PRIORITY (Nice to have)

**7. Better train/val split**
```python
# train_modal_lc0_fixed.py:142
train_split=0.95  # Change to:
train_split=0.90  # More validation data for better monitoring
```

**8. Add learning rate warmup**
```python
# Warmup for first 10% of training
warmup_steps = int(0.1 * num_epochs)
scheduler = torch.optim.lr_scheduler.ChainedScheduler([
    torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_steps)
])
```

**9. Label smoothing for policy**
```python
# training/scripts/models/pt_losses.py
def policy_loss(target, output, smoothing=0.05):
    """Add label smoothing to prevent overconfident predictions"""
    n_classes = output.size(-1)
    smoothed_target = target * (1 - smoothing) + smoothing / n_classes
    return F.cross_entropy(output, smoothed_target)
```

---

## 🔴 CRITICAL ISSUE #2: Move Generation Speed (Timeout Losses)

### Current Performance Analysis

**From MCTS.py profiling output:**
- Legal moves generation: ~5-10ms
- NN prediction: ~50-100ms (CPU)
- MCTS search (200 sims): ~6000ms (200 × 30ms)
- Total per move: **~6-8 seconds**
- Game time budget: 60 seconds / ~40 moves = **1.5 seconds per move**
- **Result: Consistently losing on time! 4-5x too slow!**

### Bottleneck Analysis

**Primary Bottlenecks (measured):**

1. **Sequential NN evaluation during MCTS** (80% of time)
   - Each simulation calls NN once: 200 sims × 50ms = 10 seconds
   - No batching → GPU underutilized
   - Solution: Batch inference

2. **Board copying in child_finder** (10% of time)
   - `board.copy()` called for every legal move
   - ~30-40 legal moves × 200 sims = 6000-8000 copies per move
   - Even optimized `board.copy()` (vs deepcopy) is expensive at scale
   - Solution: Reuse positions, better caching

3. **Cache miss rate** (5% of time)
   - Transposition table exists but cleared each game
   - Could be shared across games for opening positions
   - Solution: Persistent cache with LRU eviction

4. **Suboptimal simulation count** (5% of time)
   - Fixed 200 simulations regardless of position complexity
   - Simple positions waste time
   - Solution: Dynamic simulation budget

---

### Solutions - Priority Order

#### 🔥 IMMEDIATE (Implement today)

**1. Reduce MCTS simulation count (Quick win)**
```python
# src/MCTS.py:371
MAX_SIMULATIONS = 200  # Change to:
MAX_SIMULATIONS = 50   # 4x faster, minimal strength loss

# Dynamic adjustment:
if len(legal_moves) < 5:  # Forced/obvious position
    max_simulations = min(20, MAX_SIMULATIONS)
elif len(legal_moves) > 30:  # Complex position
    max_simulations = min(100, MAX_SIMULATIONS)
else:
    max_simulations = MAX_SIMULATIONS
```
**Expected impact:** 4x faster (6s → 1.5s per move), -50 to -100 ELO
**Verdict:** Worth it to avoid timeout losses!

**2. Optimize NN inference - Use FP16**
```python
# src/models/lc0_inference.py - LC0ModelLoader
def load_model(self):
    # ... existing loading ...

    # Convert to FP16 for 2x speedup
    if self.device == "cpu":
        # FP16 on CPU requires special support
        pass  # Keep FP32 on CPU
    else:
        self.model = self.model.half()  # FP16 on GPU
        print("✅ Model converted to FP16 for faster inference")
```
**Expected impact:** 2x faster on GPU (100ms → 50ms), minimal ELO loss
**Note:** Only helps if deploying on GPU

**3. Persistent position cache across games**
```python
# src/MCTS.py:205-209
# Change from game-scoped cache to global persistent cache
from functools import lru_cache

# Replace dict with LRU cache (10k positions = ~10MB)
position_cache = {}  # Remove this

@lru_cache(maxsize=10000)
def get_cached_evaluation(fen_hash):
    """Cache NN evaluations with LRU eviction"""
    return None  # Will be filled by caller

def node_evaluator(node):
    fen = node.state.fen()
    fen_hash = hash(fen)

    cached = get_cached_evaluation(fen_hash)
    if cached is not None:
        return cached

    # ... evaluate with NN ...

    # Cache with LRU
    get_cached_evaluation.__wrapped__  # Access underlying cache
    get_cached_evaluation(fen_hash)  # Manually add
```
**Expected impact:** +20-30% speedup in repeated positions (especially opening)

#### ⚠️ HIGH PRIORITY (Implement this week)

**4. Batch NN inference in MCTS**

This is the **MOST IMPORTANT** optimization but requires significant refactoring.

**Current (Sequential):**
```python
# MCTS simulation
for i in range(num_simulations):
    leaf = select_leaf(root)
    policy, value = model.evaluate(leaf.board)  # Individual inference!
    expand(leaf, policy)
    backup(leaf, value)
```

**Optimized (Batched):**
```python
# Collect multiple leaves before inference
batch_size = 8
leaf_batch = []

for i in range(num_simulations):
    leaf = select_leaf_with_virtual_loss(root)  # Add virtual loss
    leaf_batch.append(leaf)

    if len(leaf_batch) >= batch_size or i == num_simulations - 1:
        # Batch inference
        boards = [leaf.state for leaf in leaf_batch]
        policies, values = model.batch_evaluate(boards)  # Single batched call!

        # Expand and backup
        for leaf, policy, value in zip(leaf_batch, policies, values):
            remove_virtual_loss(leaf)
            expand(leaf, policy)
            backup(leaf, value)

        leaf_batch.clear()
```

**Implementation changes needed:**

```python
# src/models/lc0_inference.py - Add batch evaluation
class LC0ModelLoader:
    def batch_evaluate(self, boards):
        """
        Evaluate multiple positions at once.

        Args:
            boards: List of chess.Board objects

        Returns:
            (policies, values) - batched predictions
        """
        # Convert all boards to tensors
        board_tensors = [self.board_to_tensor(b) for b in boards]
        batch = torch.stack(board_tensors).to(self.device)

        # Single forward pass
        with torch.no_grad():
            policy_out, value_out, _ = self.model(batch)

        # Convert to move probabilities
        policies = []
        for i, board in enumerate(boards):
            policy_dict = self.tensor_to_policy(policy_out[i], board)
            policies.append(policy_dict)

        values = value_out.cpu().numpy()

        return policies, values
```

**Expected impact:** 5-8x faster MCTS (6s → 0.8-1.2s), no ELO loss!
**Effort:** High (2-4 hours implementation + testing)

**5. Fast opening move heuristic**
```python
# src/MCTS.py - In test_func() main logic
move_number = ctx.board.fullmove_number

if move_number <= 8:  # Opening phase
    print("⚡ OPENING: Using fast NN policy (no MCTS)")
    # Just use top NN move
    best_move = max(move_probs.items(), key=lambda x: x[1])[0]
    return best_move
```
**Expected impact:** Save 5-10 seconds in opening, +30-50 ELO from better time management

#### ✅ MEDIUM PRIORITY

**6. Adaptive simulation budget based on policy entropy**
```python
def calculate_simulation_budget(policy_probs, base_simulations=50):
    """
    If policy is confident (low entropy), use fewer simulations.
    If policy is uncertain (high entropy), use more simulations.
    """
    # Calculate entropy
    entropy = -sum(p * math.log(p + 1e-8) for p in policy_probs.values())
    max_entropy = math.log(len(policy_probs))
    normalized_entropy = entropy / max_entropy

    # Scale simulations
    if normalized_entropy < 0.3:  # Very confident
        return int(base_simulations * 0.5)
    elif normalized_entropy > 0.7:  # Very uncertain
        return int(base_simulations * 1.5)
    else:
        return base_simulations
```

**7. Improve early stopping in MCTS**
```python
# src/MCTS.py:128-141
def should_stop_early(self):
    if len(self.root_node.children) < 2:
        return True

    sorted_children = sorted(self.root_node.children, key=lambda c: c.visits, reverse=True)
    best_visits = sorted_children[0].visits
    second_visits = sorted_children[1].visits if len(sorted_children) > 1 else 0

    # More aggressive early stopping
    if self.root_node.visits > 10:  # Reduced from 5
        # If best move has 80% of visits (vs 70%), stop
        if best_visits > 0.8 * self.root_node.visits:
            return True
        # OR if best move has 3x more visits than second best
        if best_visits > 3 * second_visits:
            return True

    return False
```

---

## 🔴 CRITICAL ISSUE #3: Self-Play Training System

### Current State
- **Self-play: Not implemented**
- Only supervised learning from Lichess 2200+ games
- No continuous improvement mechanism

### Why Self-Play Matters

**Benefits:**
1. **Learns to fix own weaknesses:** Model plays itself, identifies bad patterns
2. **Superhuman play:** AlphaZero proved self-play → beyond human level
3. **Infinite data:** Generate unlimited training examples
4. **Adaptive learning:** Curriculum automatically adjusts difficulty

**Risks:**
1. **Cold start problem:** Needs decent baseline model (1500+ ELO)
2. **Computational cost:** Requires many games (10k-100k per iteration)
3. **Training instability:** Can diverge or collapse to trivial strategies
4. **Time constraint:** 36-hour hackathon may not allow full convergence

### Recommendation: Hybrid Approach

**Phase 1: Supervised Learning (Hours 0-12)**
- Train on Lichess 2200+ data
- Get to ~1800-2000 ELO baseline
- This is the foundation

**Phase 2: Self-Play Refinement (Hours 12-30)**
- Generate self-play games with current model
- Mix 70% self-play + 30% Lichess data
- Fine-tune model on mixed dataset
- Iterate 2-3 times

**Phase 3: Final Tuning (Hours 30-36)**
- Lock model, only tune hyperparameters
- Deploy best version

---

### Implementation Plan

#### Architecture: Self-Play Training Loop

```
┌─────────────────────────────────────────────────────┐
│  SELF-PLAY TRAINING LOOP                            │
│                                                     │
│  1. Current Model (v1)                              │
│          ↓                                          │
│  2. Play N games against itself                    │
│     - Use MCTS for move selection                  │
│     - Add exploration (temperature, Dirichlet)     │
│     - Record (state, policy, outcome)              │
│          ↓                                          │
│  3. Preprocess games → training data               │
│     - Convert to 112-channel format                │
│     - Generate policy targets from MCTS visits     │
│     - Label with actual game outcome (WDL)         │
│          ↓                                          │
│  4. Mix with human data                            │
│     - 70% self-play games                          │
│     - 30% Lichess 2200+ games                      │
│          ↓                                          │
│  5. Train new model (v2)                           │
│     - Standard supervised learning                 │
│     - Same architecture as v1                      │
│          ↓                                          │
│  6. Evaluate: v2 vs v1                             │
│     - Play 100 games head-to-head                  │
│     - If v2 wins >55%, accept as new current       │
│     - Else, reject and tune parameters             │
│          ↓                                          │
│  7. Repeat from step 2 with v2                     │
└─────────────────────────────────────────────────────┘
```

#### File Structure

```
training/
├── scripts/
│   ├── self_play.py              # NEW - Generate self-play games
│   ├── self_play_modal.py        # NEW - Parallel self-play on Modal
│   ├── evaluate_models.py        # NEW - Head-to-head evaluation
│   ├── preprocess_selfplay.py    # NEW - Convert self-play games to training data
│   └── train_hybrid.py           # NEW - Train on mixed data
```

#### Component 1: Self-Play Game Generation

```python
# training/scripts/self_play.py
"""
Generate self-play training games.

Uses current model + MCTS to play games against itself.
Adds exploration (temperature, Dirichlet noise) for diversity.
Records training examples (state, MCTS policy, outcome).
"""

import chess
import torch
from pathlib import Path
from tqdm import tqdm
import json

# Import our MCTS and model
import sys
sys.path.append("src")
from MCTS import MonteCarlo, Node
from models.lc0_inference import LC0ModelLoader

def play_self_play_game(
    model_loader,
    num_simulations=100,
    temperature=1.0,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    max_moves=200
):
    """
    Play one self-play game.

    Args:
        model_loader: Loaded LC0 model
        num_simulations: MCTS simulations per move
        temperature: Exploration temperature (higher = more random)
        dirichlet_alpha: Dirichlet noise concentration
        dirichlet_epsilon: Fraction of Dirichlet noise to add
        max_moves: Maximum game length

    Returns:
        List of training examples: [(board_state, mcts_policy, outcome), ...]
    """
    board = chess.Board()
    examples = []

    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        # Run MCTS from current position
        root = Node(board.copy())
        mcts = MonteCarlo(root)

        # Set up evaluator
        def child_finder(node, montecarlo):
            for move in node.state.legal_moves:
                new_board = node.state.copy()
                new_board.push(move)
                child = Node(new_board, move=move)
                node.add_child(child)

        def node_evaluator(node):
            if node.state.is_checkmate():
                return -float('inf')
            elif node.state.is_game_over():
                return 0.0
            return model_loader.evaluate_position(node.state)

        mcts.child_finder = child_finder
        mcts.node_evaluator = node_evaluator

        # Run simulations
        mcts.simulate(num_simulations)

        # Add Dirichlet noise to root (for exploration)
        if dirichlet_epsilon > 0:
            add_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon)

        # Get MCTS visit distribution (this is our improved policy)
        visit_counts = {child.move: child.visits for child in root.children}
        total_visits = sum(visit_counts.values())
        mcts_policy = {move: visits / total_visits for move, visits in visit_counts.items()}

        # Sample move based on temperature
        move = sample_move(visit_counts, temperature)

        # Record training example
        examples.append({
            'fen': board.fen(),
            'mcts_policy': {m.uci(): p for m, p in mcts_policy.items()},
            'move_count': move_count
        })

        # Make move
        board.push(move)
        move_count += 1

    # Label all examples with game outcome
    outcome = get_game_outcome(board)

    for i, example in enumerate(examples):
        # Outcome from perspective of player who made the move
        player = chess.WHITE if i % 2 == 0 else chess.BLACK
        example['outcome'] = outcome if player == chess.WHITE else -outcome

    return examples


def add_dirichlet_noise(root, alpha, epsilon):
    """Add Dirichlet noise to root node for exploration."""
    import numpy as np

    children = root.children
    if not children:
        return

    noise = np.random.dirichlet([alpha] * len(children))

    for child, noise_value in zip(children, noise):
        # Mix MCTS prior with noise
        if child.policy_value is not None:
            child.policy_value = (1 - epsilon) * child.policy_value + epsilon * noise_value


def sample_move(visit_counts, temperature):
    """Sample move based on visit counts and temperature."""
    import numpy as np

    moves = list(visit_counts.keys())
    visits = np.array([visit_counts[m] for m in moves])

    if temperature == 0:
        # Greedy: pick most visited
        return moves[np.argmax(visits)]

    # Temperature scaling
    probs = visits ** (1.0 / temperature)
    probs = probs / probs.sum()

    return np.random.choice(moves, p=probs)


def get_game_outcome(board):
    """
    Get game outcome from White's perspective.

    Returns:
        1.0 = White win
        0.0 = Draw
        -1.0 = Black win
    """
    if not board.is_game_over():
        return 0.0  # Incomplete game treated as draw

    result = board.result()

    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    else:
        return 0.0


def generate_self_play_games(
    model_path,
    num_games=100,
    output_dir="training/data/selfplay",
    **game_kwargs
):
    """
    Generate multiple self-play games.

    Args:
        model_path: Path to model checkpoint
        num_games: Number of games to generate
        output_dir: Where to save games
        **game_kwargs: Arguments passed to play_self_play_game()
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model_loader = LC0ModelLoader(
        repo_id=None,  # Load from local path
        model_file=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model_loader.load_model()

    # Generate games
    all_examples = []

    for game_num in tqdm(range(num_games), desc="Self-play games"):
        examples = play_self_play_game(model_loader, **game_kwargs)

        # Save individual game
        game_file = output_dir / f"game_{game_num:05d}.json"
        with open(game_file, 'w') as f:
            json.dump(examples, f)

        all_examples.extend(examples)

    print(f"\n✅ Generated {num_games} games with {len(all_examples)} training positions")
    print(f"Saved to {output_dir}")

    return all_examples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--output-dir", default="training/data/selfplay")

    args = parser.parse_args()

    generate_self_play_games(
        model_path=args.model_path,
        num_games=args.num_games,
        num_simulations=args.simulations,
        output_dir=args.output_dir
    )
```

#### Component 2: Parallel Self-Play on Modal

```python
# training/scripts/self_play_modal.py
"""
Massively parallel self-play game generation on Modal.

Launches N workers in parallel, each generating games independently.
Aggregates results and saves to Modal volume.
"""

import modal

app = modal.App("chesshacks-selfplay")

# Same image as training
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy",
        "tqdm",
        "python-chess",
        "huggingface-hub",
    )
    .add_local_dir("training/scripts", remote_path="/root/scripts")
    .add_local_dir("src", remote_path="/root/src")
)

volume = modal.Volume.from_name("chess-training-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",  # Each worker gets a GPU
    timeout=3600,
    volumes={"/data": volume},
)
def generate_games_worker(
    model_path: str,
    games_per_worker: int,
    worker_id: int,
    simulations: int = 100
):
    """
    Generate self-play games on one worker.
    """
    import sys
    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root/scripts")

    from self_play import generate_self_play_games

    output_dir = f"/data/selfplay/worker_{worker_id}"

    examples = generate_self_play_games(
        model_path=model_path,
        num_games=games_per_worker,
        output_dir=output_dir,
        num_simulations=simulations
    )

    return len(examples)


@app.local_entrypoint()
def main(
    model_path: str = "/data/models/best_lc0_model.pt",
    total_games: int = 1000,
    num_workers: int = 10,
    simulations: int = 100
):
    """
    Generate self-play games in parallel.

    Args:
        model_path: Path to model in Modal volume
        total_games: Total games to generate
        num_workers: Number of parallel workers
        simulations: MCTS simulations per move
    """
    games_per_worker = total_games // num_workers

    print(f"🚀 Launching {num_workers} workers to generate {total_games} games")
    print(f"   Games per worker: {games_per_worker}")
    print(f"   Simulations per move: {simulations}")

    # Launch all workers in parallel
    results = list(
        generate_games_worker.map(
            [model_path] * num_workers,
            [games_per_worker] * num_workers,
            range(num_workers),
            [simulations] * num_workers
        )
    )

    total_positions = sum(results)

    print(f"\n✅ Generated {total_games} games with {total_positions} training positions")
```

#### Component 3: Convert Self-Play to Training Data

```python
# training/scripts/preprocess_selfplay.py
"""
Convert self-play games to LC0 training format.

Reads JSON game files, converts to 112-channel .npz format.
"""

def convert_selfplay_to_lc0(
    selfplay_dir: str,
    output_dir: str,
    positions_per_file: int = 50000
):
    """
    Convert self-play games to .npz training data.

    Similar to preprocess_pgn_to_lc0.py but:
    - Reads from JSON instead of PGN
    - Uses MCTS visit distribution as policy target (better than game move!)
    - Uses actual game outcome as value target
    """
    import json
    from pathlib import Path
    import numpy as np
    from tqdm import tqdm

    selfplay_dir = Path(selfplay_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all game files
    game_files = sorted(selfplay_dir.glob("**/*.json"))
    print(f"Found {len(game_files)} self-play games")

    # Process games
    inputs_batch = []
    policies_batch = []
    values_batch = []
    moves_left_batch = []

    file_idx = 0

    for game_file in tqdm(game_files, desc="Converting games"):
        with open(game_file) as f:
            examples = json.load(f)

        for example in examples:
            # Convert FEN to 112-channel representation
            board = chess.Board(example['fen'])
            input_tensor = board_to_lc0_planes(board)

            # Convert MCTS policy to 1858-dim target
            policy_target = mcts_policy_to_lc0_target(example['mcts_policy'], board)

            # Convert outcome to WDL
            outcome = example['outcome']
            wdl_target = outcome_to_wdl(outcome)

            # Moves left (rough estimate from move count)
            moves_left = max(0, 80 - example['move_count'])

            inputs_batch.append(input_tensor)
            policies_batch.append(policy_target)
            values_batch.append(wdl_target)
            moves_left_batch.append(moves_left)

            # Save batch if full
            if len(inputs_batch) >= positions_per_file:
                save_batch(
                    output_dir / f"selfplay_{file_idx:04d}.npz",
                    inputs_batch,
                    policies_batch,
                    values_batch,
                    moves_left_batch
                )
                file_idx += 1
                inputs_batch.clear()
                policies_batch.clear()
                values_batch.clear()
                moves_left_batch.clear()

    # Save remaining
    if inputs_batch:
        save_batch(
            output_dir / f"selfplay_{file_idx:04d}.npz",
            inputs_batch,
            policies_batch,
            values_batch,
            moves_left_batch
        )

    print(f"✅ Saved {file_idx + 1} .npz files to {output_dir}")


def outcome_to_wdl(outcome):
    """
    Convert outcome to WDL (win/draw/loss) distribution.

    Args:
        outcome: 1.0 (win), 0.0 (draw), -1.0 (loss)

    Returns:
        [loss_prob, draw_prob, win_prob]
    """
    if outcome > 0.5:  # Win
        return [0.0, 0.0, 1.0]
    elif outcome < -0.5:  # Loss
        return [1.0, 0.0, 0.0]
    else:  # Draw
        return [0.0, 1.0, 0.0]
```

#### Component 4: Training on Mixed Data

```python
# training/scripts/train_hybrid.py
"""
Train on mixed self-play + human data.

Combines:
- 70% self-play games (recent iterations)
- 30% Lichess 2200+ games (for human opening knowledge)
"""

def create_hybrid_dataloader(
    selfplay_dir: str,
    human_dir: str,
    selfplay_ratio: float = 0.7,
    batch_size: int = 256
):
    """
    Create dataloader that samples from both datasets.

    Args:
        selfplay_dir: Self-play .npz files
        human_dir: Lichess .npz files
        selfplay_ratio: Fraction of batches from self-play (0.7 = 70%)
        batch_size: Batch size

    Returns:
        Combined DataLoader
    """
    # Implementation: Interleave batches from two datasets
    # Pseudo-code:
    # while True:
    #     if random() < selfplay_ratio:
    #         yield batch from selfplay_loader
    #     else:
    #         yield batch from human_loader
    pass  # Full implementation needed
```

#### Component 5: Model Evaluation (Head-to-Head)

```python
# training/scripts/evaluate_models.py
"""
Evaluate two models by playing them against each other.

Determines if new model is better than current champion.
"""

def play_match(
    model1_path: str,
    model2_path: str,
    num_games: int = 100,
    simulations: int = 100
):
    """
    Play models against each other.

    Returns:
        win_rate: Fraction of games won by model1
    """
    # Implementation: Similar to self_play.py but two different models
    pass
```

---

### Self-Play Training Schedule

**Iteration 1 (Hours 12-18):**
- Generate 1000 self-play games with baseline model
- Convert to training data (~20k positions)
- Mix with 30k Lichess positions
- Train for 3 epochs
- Evaluate: If +50 ELO vs baseline, accept

**Iteration 2 (Hours 18-24):**
- Generate 2000 games with improved model
- Mix with Lichess data (70/30)
- Train for 3 epochs
- Evaluate: Require +50 ELO improvement

**Iteration 3 (Hours 24-30):**
- Generate 3000 games
- Train final iteration
- Select best model from all iterations

**Fallback:** If self-play doesn't improve, revert to supervised learning

---

## 🟡 ISSUE #4: Training Data Quality (Blunder Handling)

### The Problem

**User's concern is valid:**
> If we only train on 2200+ ELO games, the bot might not know how to punish blunders!

**Why this matters:**
1. High-ELO players rarely blunder → bot doesn't learn blunder patterns
2. When opponent blunders, bot might not recognize the mistake
3. When bot blunders (MCTS failures), it doesn't know how to recover
4. Bot trained on "perfect" play may be fragile to imperfect positions

### Evidence from Chess AI Research

**AlphaZero approach:**
- Trained ONLY on self-play (perfect play)
- Still dominated human champions
- Conclusion: Quality > handling human errors

**Leela Chess Zero approach:**
- Trained on mixed data initially
- Later switched to pure self-play
- Found that high-quality data + self-play > mixed-quality data

**Our situation (different!):**
- 36-hour hackathon, not 3 months
- Competing against other bots (not humans)
- Other bots WILL make mistakes
- Need to punish blunders to win

### Recommended Solution: Stratified Data Mix

**Don't train on pure 2200+ OR pure mixed data. Use strategic mix:**

**Strategy: 80/15/5 Split**
- **80% high-ELO (2200+):** Core strength, good fundamentals
- **15% medium-ELO (1800-2200):** Some blunders, recovery patterns
- **5% tactical puzzles:** Explicitly label "this is a blunder, punish it!"

**Implementation:**

```python
# training/scripts/download_games.py - Modified filtering
def filter_games_stratified(min_elo_high=2200, min_elo_medium=1800):
    """
    Download games with stratified sampling.

    Target composition:
    - 800k positions from 2200+ games (80%)
    - 150k positions from 1800-2200 games (15%)
    - 50k positions from tactical puzzles (5%)

    Total: 1M positions
    """
    high_elo_games = download_lichess_games(
        min_elo=min_elo_high,
        target_positions=800000,
        time_control="classical"
    )

    medium_elo_games = download_lichess_games(
        min_elo=min_elo_medium,
        max_elo=min_elo_high,
        target_positions=150000,
        time_control="rapid"  # More blunders in rapid
    )

    # Add tactical puzzles (positions where one side has clear advantage after best move)
    tactical_puzzles = download_lichess_puzzles(
        target_positions=50000,
        min_rating=1800
    )

    return combine_datasets([high_elo_games, medium_elo_games, tactical_puzzles])
```

**Rationale:**
- 80% high-ELO ensures solid fundamentals
- 15% medium-ELO exposes bot to realistic blunders
- 5% tactical puzzles explicitly teaches exploitation

**Alternative if time is limited:**
- Keep current 2200+ data
- Add self-play (which naturally creates varied positions)
- Self-play games will include "blunders" from MCTS exploration

---

## 🟡 ISSUE #5: Dataset Scaling for Larger Model

### Current Setup vs. Target

**Current:**
- 1.5M positions
- 256 filters (mentioned in user message, but code shows 128)
- 10 residual blocks
- Batch size 512 (mentioned, but code shows 256)
- 15 epochs

**Target for final model:**
- **10M+ positions** (6-7x more data)
- **256-512 filters** (2-4x wider)
- **15-20 residual blocks** (1.5-2x deeper)
- **Batch size 512-1024** (2-4x larger)
- **8-12 epochs** (fewer epochs, more data)

**Challenge:** Preprocessing 10M positions takes ~10 hours even with Modal parallelization

### Scaling Strategy

#### Data Acquisition Plan

**Option 1: Download more Lichess games (Recommended)**
```bash
# Download monthly database files
# Lichess has ~200M games total, filter to 2200+
# Target: 10M positions from ~500k games

modal run training/scripts/preprocess_modal_lc0.py \
    --num-games 500000 \
    --min-elo 2200 \
    --time-control classical,rapid
```

**Timeline:**
- Download: 2-3 hours (parallel downloads)
- Preprocess: 6-8 hours (parallel on Modal with 20 workers)
- Total: 10 hours for 10M positions

**Option 2: Use existing datasets**
- CCRL games (computer chess rating list)
- Chess.com games (need API access)
- Engine games (Stockfish vs Stockfish)

#### Model Scaling Parameters

**For 10M dataset, recommended model:**

```python
# Final model configuration
num_filters = 256  # Up from 128
num_residual_blocks = 15  # Up from 6-10
se_ratio = 8  # Keep squeeze-excitation
batch_size = 512  # Up from 256
num_epochs = 8  # Down from 15 (more data = fewer epochs needed)

# Expected parameters: ~35M (vs current ~15M)
# Expected inference time: 80-100ms on CPU (vs 50ms)
# Expected ELO: 2200-2400 (vs 2000-2200)
```

**Training time estimate:**
- 10M positions / 512 batch = 19,531 batches per epoch
- 8 epochs = 156,248 batches total
- H100 GPU: ~0.5s per batch = ~22 hours
- A10G GPU: ~1.0s per batch = ~43 hours (too long!)
- **Recommendation: Use H100 GPU on Modal**

#### Incremental Scaling Approach

**Don't jump straight to 10M. Scale gradually:**

**Stage 1: Baseline (Current)**
- 1.5M positions, 128x6 model
- Train time: 4 hours
- Target ELO: 1900-2100

**Stage 2: First Scale-Up (Hours 12-18)**
- 4M positions, 192x10 model
- Train time: 8 hours
- Target ELO: 2000-2200

**Stage 3: Final Model (Hours 24-32)**
- 10M positions, 256x15 model
- Train time: 20 hours
- Target ELO: 2200-2400

**Stage 4: Deploy Best (Hours 32-36)**
- If Stage 3 doesn't finish, deploy Stage 2
- If Stage 3 finishes but overfits, deploy Stage 2
- If Stage 3 succeeds, deploy Stage 3

---

## 📊 Implementation Priority Matrix

### Week 1 (Immediate - Next 48 Hours)

| Priority | Task | Effort | ELO Impact | Time Saved |
|----------|------|--------|------------|------------|
| 🔥 P0 | Remove 1000-batch limit in training | 5 min | +150-200 | N/A |
| 🔥 P0 | Reduce MCTS simulations (200→50) | 5 min | -50 to -100 | +4s/move |
| 🔥 P0 | Add dropout to model | 30 min | +30-60 | N/A |
| 🔥 P0 | Increase weight decay (0.0001→0.0005) | 2 min | +20-40 | N/A |
| ⚠️ P1 | Persistent position cache across games | 1 hour | +20-30 | +0.5s/move |
| ⚠️ P1 | Fast opening moves (skip MCTS) | 30 min | +30-50 | +10s/game |
| ⚠️ P1 | Early stopping (patience=3) | 30 min | +30-50 | N/A |

**Total time: ~3 hours**
**Expected improvement: +200-400 net ELO, +5-6s per move faster**

### Week 2 (High Priority - This Week)

| Priority | Task | Effort | ELO Impact | Time Saved |
|----------|------|--------|------------|------------|
| ⚠️ P1 | Batch NN inference in MCTS | 4 hours | 0 | +5s/move |
| ⚠️ P1 | Data augmentation (horizontal flip) | 2 hours | +50-100 | N/A |
| ⚠️ P1 | Download + preprocess 4M positions | 4 hours | +100-150 | N/A |
| ⚠️ P1 | Train improved model (192x10) | 8 hours | +200-300 | N/A |

**Total time: 18 hours (mostly automated)**
**Expected improvement: +350-550 ELO, +5s per move faster**

### Week 3 (Self-Play - If Time Permits)

| Priority | Task | Effort | ELO Impact |
|----------|------|--------|------------|
| ✅ P2 | Implement self-play game generation | 6 hours | TBD |
| ✅ P2 | Implement self-play on Modal (parallel) | 4 hours | TBD |
| ✅ P2 | Generate 1000 self-play games | 2 hours | TBD |
| ✅ P2 | Train hybrid model (iteration 1) | 6 hours | +50-150 |

**Total time: 18 hours**
**Expected improvement: +50-200 ELO (uncertain, high risk)**

---

## 🎯 Success Metrics

### Training Metrics to Monitor

**Overfitting indicators:**
- Val loss stops improving while train loss decreases
- Val loss increases while train loss decreases
- Large gap between train/val accuracy

**Good training signs:**
- Val loss closely tracks train loss
- Both losses decrease steadily
- Policy accuracy > 35% on validation
- Value MSE < 0.15

### In-Game Performance Metrics

**Speed metrics:**
- Average time per move: **Target <1.5s** (currently 6-8s)
- Timeout losses: **Target 0%** (currently common)
- Cache hit rate: **Target >50%** in openings

**Strength metrics:**
- ELO rating: **Target 2200+**
- Win rate vs Stockfish depth=5: **Target >40%**
- Tactical puzzle accuracy: **Target >60%**

---

## 📝 Concrete Action Plan

### Day 1 (Today - Next 8 Hours)

**Morning (4 hours):**
1. ✅ Fix training loop (remove 1000-batch limit) - 10 min
2. ✅ Add dropout to model - 30 min
3. ✅ Increase weight decay - 5 min
4. ✅ Add early stopping - 30 min
5. ✅ Test training locally on sample data - 30 min
6. 🚀 Launch improved training run on Modal (4M positions if available, else 1.5M) - 15 min setup
   - Let it run in background (4-8 hours)

**Afternoon (4 hours):**
7. ✅ Reduce MCTS simulations in inference - 10 min
8. ✅ Add persistent cache - 1 hour
9. ✅ Add fast opening moves - 30 min
10. ✅ Test inference speed locally - 30 min
11. ✅ Deploy to test slot - 15 min
12. Monitor training run, check for overfitting

**Evening:**
- Training should complete
- Download new model
- Test locally
- Deploy to Slot 2

### Day 2 (Tomorrow - Next 24 Hours)

**Morning:**
1. Implement batch inference in MCTS - 4 hours
2. Test speed improvements - 1 hour
3. Deploy optimized inference to Slot 1

**Afternoon:**
4. Start downloading 4M+ positions - 2 hours
5. Preprocess in parallel on Modal - 4 hours
6. Implement data augmentation - 2 hours

**Evening:**
7. Launch large training run (4M positions, 192x10 model) - overnight
8. Monitor for overfitting

### Day 3-4 (Optional - Self-Play)

**Only if base model is >2000 ELO:**
1. Implement self-play game generation - 6 hours
2. Generate 1000 games - 2 hours
3. Preprocess self-play data - 2 hours
4. Train hybrid model - 6 hours
5. Evaluate improvement

**If self-play doesn't help:**
- Revert to best supervised model
- Focus on hyperparameter tuning

---

## 🔬 Testing & Validation

### Before Deploying Changes

**Must test:**
1. Training completes without errors
2. Model loads correctly in inference
3. Inference time <2s per move
4. No illegal moves generated
5. Bot doesn't crash on edge cases (stalemate, repetition)

### Evaluation Protocol

**Against known baselines:**
- Stockfish depth=3: Should win >80%
- Stockfish depth=5: Should win >30%
- Random player: Should win 100%

**Self-evaluation:**
- New model vs old model: 100 games
- Accept if win rate >55%

---

## ⚠️ Risk Mitigation

### High-Risk Items

**1. Batch inference refactoring**
- Risk: Breaks MCTS, introduces bugs
- Mitigation: Keep sequential version as fallback, test thoroughly

**2. Self-play training**
- Risk: Doesn't converge in time, wastes GPU hours
- Mitigation: Only attempt if baseline >2000 ELO, set time limit

**3. Large model (256x15) training**
- Risk: Doesn't finish before deadline, overfits
- Mitigation: Start with 192x10, only scale up if time permits

### Rollback Plan

**If anything breaks:**
1. Revert to last working commit
2. Deploy best known model to all slots
3. Focus on inference optimizations only

---

## 📌 Summary - Top 5 Must-Do Items

1. **Fix training overfitting** (30 min) → +200 ELO
2. **Speed up inference** (1 hour) → Avoid timeouts
3. **Train on more data** (overnight) → +200 ELO
4. **Batch inference** (4 hours) → 5x faster
5. **Deploy best model** (ongoing) → Win tournament!

**Expected final result:**
- ELO: 2200-2400
- Speed: <1.5s per move
- Reliability: No timeouts, no illegal moves
- Confidence: High chance of Queen's Crown! 👑

---

*This plan should be updated as we implement and discover new issues.*
