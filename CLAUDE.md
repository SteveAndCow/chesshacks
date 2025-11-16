# ChessHacks Bot - Project Documentation

**Last Updated:** 2025-11-15
**Competition:** ChessHacks 36-Hour Hackathon
**Track:** Queen's Crown (Highest ELO Rating)

---

## 🎯 Competition Requirements

### Core Rules
- **Neural network must be a critical component** (engine fails without it)
- Must generate only **legal moves** (illegal moves = disqualification)
- Cannot use pre-trained chess models or existing engines (e.g., Stockfish)
- Must train your own models

### Platform Constraints
- **Game time limit:** 1 minute per game (dynamic time management required)
- **Build time limit:** 3 minutes (CPU + GPU) when deploying
- **Bot slots:** 3 available (each with separate ELO; best slot counts)
- **File size:** No files > 100MB in git (use HuggingFace for model weights)

### Technical Requirements
- **Bot implementation:** `/src/main.py` with `get_move(pgn: str) -> str` function
- **Input format:** PGN (Portable Game Notation) string
- **Output format:** UCI notation (e.g., `e2e4`, `e1g1`, `e7e8q`)
- **Deployment files:** `/src`, `serve.py`, `requirements.txt`
- **Imports:** Use relative imports (e.g., `from .utils import ...`)

---

## 🧠 Strategy: Two Model Approaches

### Approach 1: Minimal Leela Chess Zero (PyTorch)
- **Repository:** https://github.com/Rocketknight1/minimal_lczero
- **Architecture:** ResNet-style CNN + MCTS
- **Key Optimizations:** Squeeze-Excitation blocks, WDL value head, illegal move masking
- **Training:** Modal (cloud GPU), 2200+ ELO data only
- **Weights:** HuggingFace (download & cache for inference)
- **Target:** 2000-2200 ELO with priority optimizations

### Approach 2: ChessFormers
- **Repository:** https://github.com/Atenrev/chessformers
- **Architecture:** Transformer with relative position bias + MCTS
- **Key Optimizations:** Chess-aware attention, multi-task learning, illegal move masking
- **Training:** Modal (cloud GPU), 2200+ ELO data only
- **Weights:** HuggingFace (download & cache for inference)
- **Target:** 2100-2300 ELO with priority optimizations

### Engineering Philosophy
**See DESIGN_DECISIONS.md for detailed engineering analysis**

**Core Principles:**
1. **Data Quality > Quantity:** Clean 1M high-ELO games beats dirty 10M
2. **Fast Inference = More Search:** Optimize for <100ms forward pass
3. **Illegal Move Masking:** +100-200 ELO, prevents disqualification
4. **Dynamic Time Management:** Spend time on complex positions, fast moves on simple ones
5. **Parallel MCTS:** 4-8x speedup via virtual loss and batch inference

## 🏗️ Bot Architecture

### Pipeline Flow
```
PGN Input → Neural Network → Move Probabilities + Position Value → MCTS Search → UCI Move Output
```

### Key Components
1. **Neural Network**: Policy (move probabilities) + Value (position evaluation)
2. **MCTS Search**: Explores promising moves guided by NN predictions
3. **Time Management**: Dynamically allocate search time based on remaining game time

---

## 📊 Training Pipeline

### Data Source
- **Source:** Lichess database or similar high-ELO chess matches
- **Format:** PGN files → Preprocessed tensors
- **Filtering:** Min 2000 Elo (both players)

### Training Infrastructure
- **Platform:** Modal (cloud GPU)
- **Model Storage:** HuggingFace Hub
- **Inference:** Download and cache weights on first run

### Dataset Preprocessing
Follow the respective repository's preprocessing guidelines:
- **Minimal LCZero:** See repo documentation
- **ChessFormers:** See repo documentation

---

## ☁️ Modal Training Setup

### Quick Start
1. **Modal account:** Authenticate with `modal token new`
2. **HuggingFace token:** Get write token from https://huggingface.co/settings/tokens
3. **Create Modal secret:** `modal secret create huggingface-secret HF_TOKEN=hf_...`
4. **Upload training data:** `modal volume put chess-training-data <local_path> /data/processed`
5. **Launch training:** Follow repo-specific instructions for Minimal LCZero or ChessFormers

### Deployment Strategy
- **Slot 1:** Minimal LCZero model
- **Slot 2:** ChessFormers model
- **Slot 3:** Best performing variant with tuned hyperparameters

**Remember:** Best ELO from best slot counts for final ranking!

---

## 📁 Project Structure

```
chesshacks/
├── CLAUDE.md              # This file
├── requirements.txt       # Python dependencies (REQUIRED for deployment)
├── serve.py              # Backend server (REQUIRED for deployment)
│
├── src/                  # Bot implementation (REQUIRED for deployment)
│   └── main.py          # Contains get_move(pgn: str) -> str function
│
├── training/            # Training code (follows Minimal LCZero / ChessFormers structure)
│   ├── data/           # PGN files and preprocessed tensors
│   ├── scripts/        # Training scripts for Modal
│   └── configs/        # Training configurations
│
└── devtools/           # Local development environment (exclude from deployment)
    └── (Next.js frontend for testing)
```

**Deployment Checklist:**
- ✅ Include: `/src`, `serve.py`, `requirements.txt`
- ❌ Exclude: `/devtools`, `.venv`, `.env.local` (add to `.gitignore`)
- ❌ Exclude: Model weights > 100MB (use HuggingFace instead)

---

## 🚀 Development Workflow

### Local Testing
```bash
cd devtools
npm run dev  # Starts Next.js frontend + Python backend
# Visit http://localhost:3000 to test bot
```

### Training on Modal
```bash
# Follow Minimal LCZero or ChessFormers training instructions
# Upload trained weights to HuggingFace
```

### Bot Implementation (`/src/main.py`)
```python
def get_move(pgn: str, wtime: int = 60000, btime: int = 60000) -> str:
    """
    Generate best move using NN-guided MCTS.

    Args:
        pgn: Board state in PGN format
        wtime: White time remaining (milliseconds)
        btime: Black time remaining (milliseconds)

    Returns:
        Move in UCI format (e.g., "e2e4", "e1g1", "e7e8q")
    """
    # 1. Parse PGN to get board state
    board = chess.Board()
    # ... parse PGN moves

    # 2. Load cached NN model (lazy initialization)
    if model is None:
        model = load_model_from_huggingface()  # Downloads once, caches locally

    # 3. Calculate time budget for this move
    time_left = wtime if board.turn == chess.WHITE else btime
    move_time = calculate_time_budget(board, time_left)  # Dynamic allocation

    # 4. Run MCTS search
    # Key optimizations:
    # - Parallel search (4-8 workers)
    # - Transposition table caching
    # - Illegal move masking
    # - Early stopping if move is obvious
    root = MCTSNode(board)
    simulations = min(800, int(move_time / 50))  # Adaptive simulation count

    for _ in range(simulations):
        leaf = select_leaf(root)  # UCB selection
        if not leaf.is_terminal():
            # Batch inference optimization (collect multiple leaves)
            policy, value = model.evaluate(leaf.board)
            policy = mask_illegal_moves(policy, leaf.board)  # Critical!
            expand(leaf, policy)
            backup(leaf, value)

    # 5. Select best move
    best_move = root.best_child().move

    # 6. Return in UCI format
    return best_move.uci()
```

**Critical Implementation Details:**
- **Illegal move masking:** Set policy to 0 for illegal moves, renormalize
- **Transposition table:** Cache NN evaluations by board hash
- **Time management:** Allocate more time for complex/critical positions
- **Parallel MCTS:** Use virtual loss to prevent thread collisions
- **FP16 inference:** 2x speedup with minimal accuracy loss


### Deployment
```bash
# Push to GitHub
git add src/ serve.py requirements.txt
git commit -m "Update bot"
git push

# Deploy via ChessHacks dashboard
# - Connect GitHub repo
# - Assign to slot (1, 2, or 3)
# - Monitor build logs and ELO rating
```

---

## 📚 Key Resources

### Repositories
- [Minimal LCZero](https://github.com/Rocketknight1/minimal_lczero) - PyTorch implementation
- [ChessFormers](https://github.com/Atenrev/chessformers) - Transformer-based approach

### Documentation
- [ChessHacks Docs](https://docs.chesshacks.dev) - Platform documentation
- [Lichess Database](https://database.lichess.org/) - Training data source
- [python-chess](https://python-chess.readthedocs.io/) - Chess library

### Training Infrastructure
- [Modal](https://modal.com) - Cloud GPU training
- [HuggingFace Hub](https://huggingface.co) - Model weight storage

---

## ✅ Implementation Priorities

**Goal:** Queen's Crown (Highest ELO) - Target 2200+ ELO

### Phase 1: Foundation (Hours 0-8)
**Priority: Get something working**
1. ✅ Platform setup and documentation
2. [ ] Download high-quality dataset (Lichess 2200+ ELO, classical games)
3. [ ] Set up training pipeline (Minimal LCZero OR ChessFormers - pick one first)
4. [ ] Train small baseline model on Modal (~1-2 hours)
5. [ ] Implement basic `/src/main.py` with NN inference + simple MCTS
6. [ ] Deploy to Slot 1 - verify it works and doesn't crash

### Phase 2: Critical Optimizations (Hours 8-20)
**Priority: Must-have ELO gains (+500-800 ELO)**
1. [ ] **Illegal move masking** (+100-200 ELO, prevents DQ)
2. [ ] **Multi-task learning:** Add WDL (result) head (+100-150 ELO)
3. [ ] **Enhanced input:** 16+ channel representation (+50-100 ELO)
4. [ ] **Parallel MCTS:** 4-8 workers with virtual loss (4-8x faster)
5. [ ] **Dynamic time management:** Complex positions get more time (+50-100 ELO)
6. [ ] **Transposition table:** Cache NN evaluations (+20-50 ELO)
7. [ ] **FP16 quantization:** 2x faster inference
8. [ ] Train improved model, deploy to Slot 2

### Phase 3: Advanced Optimizations (Hours 20-30)
**Priority: Nice-to-have ELO gains (+200-400 ELO)**
1. [ ] **Data augmentation:** Horizontal flip (2x effective data)
2. [ ] **Better value head:** Separate win/draw/loss predictions
3. [ ] **Opening optimization:** Fast moves in opening (save 5-10s)
4. [ ] **Model architecture:** Tune depth/width for inference speed
5. [ ] Train second approach (whichever we didn't do in Phase 1)
6. [ ] Deploy best variant to Slot 3

### Phase 4: Final Tuning (Hours 30-36)
**Priority: Squeeze last ELO points**
1. [ ] Hyperparameter tuning (c_puct, temperature, simulations)
2. [ ] A/B test different configurations across slots
3. [ ] Monitor ELO ratings, redeploy best performers
4. [ ] Self-play training if time permits (risky but high reward)

### Key Decision Points
- **Hour 8:** If baseline works, proceed to Phase 2. If not, debug.
- **Hour 20:** Evaluate which model (LCZero vs ChessFormers) is stronger, focus effort there.
- **Hour 30:** Lock in best model, only tune hyperparameters.

**See DESIGN_DECISIONS.md for detailed optimization strategies**
