# Chess Engine Training Pipeline

## Directory Structure
```
training/
├── configs/           # Model configurations (YAML files)
│   ├── cnn_baseline.yaml
│   ├── transformer_tiny.yaml
│   └── transformer_full.yaml
├── data/             # Training data (gitignored)
│   ├── raw/          # Downloaded PGN files
│   └── processed/    # Preprocessed numpy arrays
├── models/           # Saved checkpoints (gitignored)
├── scripts/          # Training code (committed to git)
│   ├── models/       # Model architectures
│   │   ├── base.py   # Base interface
│   │   ├── cnn.py    # CNN models
│   │   └── transformer.py  # Transformer models
│   ├── model_factory.py    # Create models from config
│   ├── data_loader.py      # PyTorch data loading
│   ├── preprocess.py       # PGN → tensors
│   ├── train_modal.py      # Modal training script
│   └── download_games.py   # Download chess games
└── notebooks/        # Experiments (gitignored)
```

## 🚀 Quick Start (Complete Workflow)

### 1. Setup

```bash
# Install Modal
pip install modal

# Authenticate Modal (opens browser)
modal setup

# Create HuggingFace token secret
modal secret create huggingface-secret HF_TOKEN=your_hf_token_here
```

### 2. Get Training Data

Option A: Use existing dataset (fastest)
```bash
# Download from Kaggle/HuggingFace
# Example: https://www.kaggle.com/datasets/arevel/chess-games
```

Option B: Download from Lichess
```bash
cd training/scripts
python download_games.py
```

### 3. Preprocess Data

```bash
python preprocess.py --pgn data/raw/games.pgn --output data/processed --max-games 100000
```

This creates:
- `data/processed/boards.npy` - (N, 12, 8, 8) board tensors
- `data/processed/moves.npy` - (N,) move indices
- `data/processed/values.npy` - (N,) game outcomes

### 4. Upload Data to Modal (if using Modal training)

```bash
# Upload to Modal Volume for cloud training
modal volume put chess-training-data data/processed /data/processed
```

### 5. Train on Modal

```bash
# Train CNN baseline (fast, ~30 min)
modal run training/scripts/train_modal.py --config training/configs/cnn_baseline.yaml

# Or train Transformer (slower, ~1-2 hours)
modal run training/scripts/train_modal.py --config training/configs/transformer_tiny.yaml
```

Modal will:
- Spin up A100 GPU
- Train model
- Upload to HuggingFace
- Shut down GPU

### 6. Use Model in Bot

Update `src/main.py`:

```python
from src.models.inference import ChessModelLoader
import chess

# Load model
model_loader = ChessModelLoader(
    repo_id="your-username/chesshacks-bot",
    model_name="cnn_baseline"  # or "transformer_tiny"
)
model_loader.load_model()

@chess_manager.entrypoint
def get_move(ctx: GameContext):
    # Use model for move selection
    move_probs, value = model_loader.predict(ctx.board)

    # Log probabilities for ChessHacks
    ctx.logProbabilities(move_probs)

    # Return best move
    best_move = max(move_probs.items(), key=lambda x: x[1])[0]
    return best_move
```

## 🏗️ Model Architectures

### CNN (Baseline)
- ResNet-style convolutional network
- 5 residual blocks, 128 channels
- ~10-20M parameters
- Fast training (~30 min on A100)
- Good baseline performance

### Transformer (Novel)
- ChessFormer-inspired architecture
- Relative position encoding
- 4 layers, 8 heads, 256d
- ~20-50M parameters
- Slower training (~1-2 hours)
- Better positional understanding

## 🔀 Branching Strategy

```
main (always working)
├── feature/cnn-baseline     ← Train CNN first
└── feature/transformer      ← Experiment with Transformer
```

To experiment:
```bash
# Save current work
git checkout main
git branch feature/my-experiment

# Try new model
git checkout feature/my-experiment
# Edit configs, train, test

# If good: merge to main
# If bad: stay on branch, main is safe
```

## 📊 Config Files Explained

Each config has:
```yaml
model:
  type: "cnn_lite"  # Which architecture
  params:           # Architecture-specific params
    num_channels: 128

training:
  batch_size: 256   # Batch size
  epochs: 10        # Training epochs
  lr: 0.001         # Learning rate

huggingface:
  repo_id: "your-username/chesshacks-bot"  # Where to upload
  model_name: "cnn_baseline"               # Model filename
```

## 🎯 Training Tips

1. **Start small**: Train on 10k-100k games first
2. **Use CNN baseline first**: Faster to train, proven to work
3. **Monitor validation loss**: Should decrease over time
4. **Try multiple configs in parallel**:
   ```bash
   modal run train_modal.py --config cnn_baseline.yaml &
   modal run train_modal.py --config transformer_tiny.yaml &
   ```
5. **Compare results**: Use validation accuracy to pick best model

## 🐛 Troubleshooting

### "No training data found"
- Run `preprocess.py` first
- Check that `data/processed/*.npy` files exist
- Upload to Modal Volume if using cloud training

### "Model not found on HuggingFace"
- Check `repo_id` in config matches your HF username
- Ensure HF_TOKEN is set in Modal secrets
- Training must complete successfully to upload

### "Out of memory"
- Reduce `batch_size` in config
- Use smaller model (`cnn_lite` instead of `cnn`)
- Use T4 GPU instead of A100 (cheaper, less memory)

## 💡 Advanced: Self-Play Training

For true AlphaZero-style training (if you have time):

1. Train initial model on master games
2. Generate self-play games using current model
3. Train new model on self-play data
4. Repeat steps 2-3

This requires more infrastructure but can discover novel strategies.
