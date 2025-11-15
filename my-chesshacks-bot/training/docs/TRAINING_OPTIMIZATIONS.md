# Training Optimization Guide

## Quick Reference: Adjusting Training Settings

### GPU Selection (Line 48 in train_modal.py)

```python
# For testing/debugging (cheapest, slower)
gpu="T4"  # ~$0.50/hr, good for quick tests

# For production (recommended, balanced)
gpu="A100"  # ~$2.00/hr, fast training

# For fastest training (if needed)
gpu="H100"  # ~$3-4/hr, 2x faster than A100
```

### Timeout Settings (Line 49)

```python
# Quick test runs
timeout=600  # 10 minutes

# CNN training
timeout=3600  # 1 hour

# Transformer training
timeout=3600 * 2  # 2 hours

# Full training
timeout=3600 * 4  # 4 hours (current setting)
```

### Batch Size Tuning

**CNN Models:**
```yaml
# Small GPU or testing
batch_size: 128

# A100 GPU (default)
batch_size: 256

# H100 or more memory
batch_size: 512
```

**Transformer Models:**
```yaml
# Default (fits A100)
batch_size: 128

# Smaller model or more memory
batch_size: 256
```

### Data Sampling Strategy

**Current Settings:**
- CNN: 100k samples (good for quick iteration)
- Transformer: 500k samples (good balance)

**Alternative Strategies:**

**Ultra-fast testing:**
```yaml
max_samples: 10000  # 10k samples, ~5-10 min training
epochs: 5
```

**Quick baseline:**
```yaml
max_samples: 50000  # 50k samples, ~15-20 min training
epochs: 8
```

**Production quality:**
```yaml
max_samples: 1000000  # 1M samples, ~1-2 hr training
epochs: 15
```

**Maximum quality:**
```yaml
max_samples: null  # All data (~77M samples), ~3-4 hr training
epochs: 20
```

### Learning Rate Schedules

**Step Decay (CNN - stable):**
```yaml
lr_scheduler: "step"
lr: 0.001
lr_step_size: 5  # Reduce LR every 5 epochs
lr_gamma: 0.1    # Multiply by 0.1
```

**Cosine Annealing (Transformer - smoother):**
```yaml
lr_scheduler: "cosine"
lr: 0.0001
warmup_epochs: 2  # Gradual warmup
```

**No scheduling (simplest):**
```yaml
# Just remove lr_scheduler from config
lr: 0.001
```

### Loss Weights

**Current (balanced):**
```yaml
policy_weight: 1.0
value_weight: 1.0
result_weight: 0.5
```

**Focus on move accuracy:**
```yaml
policy_weight: 2.0  # Emphasize move prediction
value_weight: 0.5
result_weight: 0.3
```

**Focus on position evaluation:**
```yaml
policy_weight: 0.5
value_weight: 2.0   # Emphasize value accuracy
result_weight: 1.0
```

## Performance Tuning by Use Case

### Use Case 1: Quick Bot Testing
**Goal:** Get any working model ASAP

```yaml
# Config modifications:
max_samples: 10000
epochs: 5
batch_size: 256
gpu: "T4"  # Cheaper

# Training time: ~5-10 minutes
# Cost: ~$0.10
# Performance: Basic, but functional
```

### Use Case 2: Competition Entry
**Goal:** Good performance in reasonable time

```yaml
# Config modifications:
max_samples: 100000  # CNN
max_samples: 500000  # Transformer
epochs: 10-15
batch_size: 256
gpu: "A100"

# Training time: 30-45 min (CNN), 1-2 hr (Transformer)
# Cost: ~$1-4
# Performance: Competitive (1800-2200 Elo)
```

### Use Case 3: Maximum Performance
**Goal:** Best possible model

```yaml
# Config modifications:
max_samples: null  # Use all data
epochs: 20
batch_size: 256
gpu: "A100"

# Training time: 3-4 hours
# Cost: ~$6-8
# Performance: Strong (2200+ Elo)
```

## Monitoring Training

### What to Watch

**Loss Values:**
```
Good training:
Epoch 1: Loss=8.5
Epoch 5: Loss=4.2  ← Decreasing
Epoch 10: Loss=2.8

Problem:
Epoch 1: Loss=8.5
Epoch 5: Loss=9.2  ← Increasing (learning rate too high?)
```

**Accuracy:**
```
Expected progression:
Epoch 1: Accuracy=5%
Epoch 5: Accuracy=20%
Epoch 10: Accuracy=35%
Final: Accuracy=40-45%
```

**Overfitting Check:**
```
Good (some overfitting is OK):
Train Loss: 2.5
Val Loss: 2.8    ← Slightly higher

Bad (severe overfitting):
Train Loss: 1.2
Val Loss: 4.5    ← Much higher (stop training earlier)
```

### When to Stop Early

**Signs to stop:**
1. Val loss increasing for 3+ epochs
2. Val loss >> Train loss (2x difference)
3. Accuracy plateaued for 5+ epochs
4. Running out of time before competition

**How to stop:**
- Press Ctrl+C in terminal
- Model will still be uploaded from last checkpoint
- Or adjust `epochs` in config before starting

## Cost Optimization

### Estimated Costs

| Setup | GPU | Time | Cost |
|-------|-----|------|------|
| Quick test | T4 | 10 min | $0.10 |
| CNN baseline | A100 | 45 min | $1.50 |
| Transformer | A100 | 2 hrs | $4.00 |
| Full training | A100 | 4 hrs | $8.00 |

### Cost-Saving Tips

1. **Start with T4 for testing**
   ```python
   gpu="T4"  # Validate everything works first
   ```

2. **Use smaller datasets initially**
   ```yaml
   max_samples: 10000  # Quick validation
   ```

3. **Reduce epochs for testing**
   ```yaml
   epochs: 5  # Just check convergence
   ```

4. **Switch to A100 only for production**
   ```python
   gpu="A100"  # When you need the full model
   ```

5. **Set aggressive timeouts**
   ```python
   timeout=1800  # 30 min max for testing
   ```

## Troubleshooting

### "Out of Memory" Error

**Solution 1: Reduce batch size**
```yaml
batch_size: 128  # Try half
```

**Solution 2: Use smaller model**
```yaml
model:
  type: "cnn_lite"  # Instead of "cnn"
```

**Solution 3: Fewer samples**
```yaml
max_samples: 50000
```

### "Training too slow"

**Solution 1: Check GPU**
```python
# Make sure using A100, not T4
gpu="A100"
```

**Solution 2: Increase batch size**
```yaml
batch_size: 512  # If memory allows
```

**Solution 3: Reduce data**
```yaml
max_samples: 100000  # Faster iterations
```

### "Model not learning" (loss not decreasing)

**Solution 1: Check learning rate**
```yaml
lr: 0.001  # Try higher if loss flat
lr: 0.0001  # Try lower if loss exploding
```

**Solution 2: Check data**
```bash
# Verify 16 channels
modal volume ls chess-training-data /data/processed
```

**Solution 3: Simplify**
```yaml
# Remove scheduler
# Just use: lr: 0.001
```

## Quick Start Commands

### Test Run (5 min, cheap)
```bash
# Edit config: max_samples: 10000, epochs: 3, gpu: "T4"
modal run training/scripts/train_modal.py --config training/configs/cnn_baseline.yaml
```

### Production CNN (45 min)
```bash
# Default config already optimized
modal run training/scripts/train_modal.py --config training/configs/cnn_baseline.yaml
```

### Production Transformer (2 hr)
```bash
# Default config already optimized
modal run training/scripts/train_modal.py --config training/configs/transformer_tiny.yaml
```

---

*For more details, see CLAUDE.md and ARCHITECTURE_IMPROVEMENTS.md*
