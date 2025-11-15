# Chess Bot Model Documentation

## Model Information

**Model Name:** CNN Baseline (cnn_baseline.pt)
**Architecture:** CNN-Lite-128
**Parameters:** 9,685,924
**Training Framework:** PyTorch 2.0+

## Model Source

**HuggingFace Repository:** [steveandcow/chesshacks-bot](https://huggingface.co/steveandcow/chesshacks-bot)
**Model File:** `cnn_baseline.pt`

The model is automatically downloaded from HuggingFace Hub on first run and cached locally in `.model_cache/`.

## Architecture Details

### Input
- **16-channel board representation** (16, 8, 8)
  - Channels 0-11: Piece positions (6 types × 2 colors)
  - Channel 12: Kingside castling rights
  - Channel 13: Queenside castling rights
  - Channel 14: En passant file indicator
  - Channel 15: Halfmove clock (normalized)

### Outputs
1. **Policy Head:** 4096 logits (from-square × to-square move encoding)
2. **Value Head:** Single scalar in [-1, 1] (position evaluation)
3. **Result Head:** 3-class logits (win/draw/loss prediction)

### Model Architecture
- **3 convolutional layers** with ReLU activation
- **128 filters** per layer
- **Batch normalization** after each conv layer
- **Global average pooling** before fully connected heads
- **Dropout** (0.1) for regularization

## Training Details

### Dataset
- **Source:** Lichess master games database (October 2024)
- **Filter:** Min 2000 Elo (both players)
- **Training samples:** 15,420 positions from 200 games
- **Data split:** 90% train (13,878), 10% validation (1,542)

### Training Configuration
- **GPU:** Modal T4
- **Epochs:** 10
- **Batch size:** 256
- **Optimizer:** Adam (lr=0.001, weight_decay=0.0001)
- **LR Schedule:** Step decay (0.1x at epochs 5, 10)
- **Loss weights:** Policy=1.0, Value=1.0, Result=0.5

### Training Results
- **Best validation loss:** 7.59 (epoch 4)
- **Final validation accuracy:** 4.54%
- **Training time:** ~10 minutes on T4

## Performance Characteristics

### Current Capabilities
- **Playing strength:** ~800-1000 Elo (estimated)
- **Move selection:** Neural network policy + MCTS (256 simulations)
- **Position evaluation:** NN value head guides tree search
- **Legal move generation:** python-chess library

### Limitations
- Trained on small dataset (15k vs 7.7M available positions)
- Low move prediction accuracy (4.5%)
- Weak positional understanding
- Conservative evaluations (values near 0)

### Expected Performance with Full Training
With 7.7M training samples:
- **Playing strength:** 1800-2000 Elo
- **Move accuracy:** 35-40%
- **Better position evaluation:** Values in [-1, +1] range
- **Stronger tactical awareness**

## Usage

### Local Testing
```bash
python serve.py
# Opens http://localhost:5058
```

### Model Loading
The bot automatically:
1. Downloads model from HuggingFace Hub (cached after first download)
2. Loads CNN-Lite architecture
3. Initializes on CPU (or GPU if available)
4. Falls back to random policy if model loading fails

### Dependencies
- PyTorch >= 2.0.0
- huggingface-hub >= 0.20.0
- python-chess == 1.999
- numpy

See `requirements.txt` for complete list.

## Reproducibility

### Model Versioning
- **HuggingFace commit:** `5c6147cee3d677d0b2ffafd5df74b7331b66f0b8`
- Model weights are pinned to specific HF Hub snapshot

### Training Code
Training scripts available at:
- `training/scripts/train_modal.py` - Main training script
- `training/configs/cnn_baseline.yaml` - Training configuration
- `training/scripts/models/cnn.py` - Model architecture

### Reproducing Training
```bash
# Setup Modal and HuggingFace credentials
modal secret create huggingface-secret HF_TOKEN=your_token

# Upload training data to Modal Volume
modal volume put chess-training-data data/processed_16ch /processed

# Run training
modal run training/scripts/train_modal.py --config training/configs/cnn_baseline.yaml
```

## Future Improvements

1. **Scale up training:** Use full 7.7M dataset
2. **Larger model:** Switch to transformer architecture
3. **Better move encoding:** Include promotion/capture flags
4. **Self-play training:** Generate additional training data
5. **Opening book:** Memorize common openings

## License & Attribution

- **Model:** Trained for ChessHacks 2024 competition
- **Training data:** Lichess database (public domain)
- **Framework:** PyTorch (BSD License)
- **Architecture inspiration:** AlphaZero, Leela Chess Zero

---

**Last Updated:** 2024-11-15
**Model Version:** v1.0-baseline
**Training Platform:** Modal.com (T4 GPU)
