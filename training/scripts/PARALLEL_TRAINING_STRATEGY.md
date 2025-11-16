# Parallel Training Strategy: 3 Models in <5 Hours

**Created:** 2025-11-16
**Dataset:** 5.5M positions (4M + 1.5M)
**Time Constraint:** <5 hours until hackathon deadline
**Hardware:** Modal H100 GPUs (3 parallel instances)

---

## 🎯 Executive Summary

**RECOMMENDATION: Stick with 5.5M positions**

Based on deep research and calculations:
- ✅ **5.5M positions = 2-3 hours training time** (with buffer for issues)
- ❌ **10M+ positions = 4-8 hours** (too risky with <5 hours remaining)
- ✅ **3 models in parallel** on separate H100 GPUs
- ✅ **All finish within ~3 hours** (fastest path to results)

---

## ⏱️ Training Time Estimates

### Model 1: LC0 128x6 (Baseline)
**Architecture:** 128 filters, 6 residual blocks
**Parameters:** ~1.5M
**Batch size:** 256

**Time breakdown:**
- Positions per epoch: 5.5M
- Batches per epoch: 21,484
- Time per batch (H100): ~25ms (optimized with torch.compile + TF32)
- **Per epoch: ~10-12 minutes**
- Validation: ~2 minutes
- **Total per epoch: ~12-14 minutes**

**Full training (10 epochs):**
- **Total time: ~2-2.5 hours**
- **Expected early stopping: epoch 7-8 = 1.5-2 hours**

### Model 2: LC0 128x10 (Deeper)
**Architecture:** 128 filters, 10 residual blocks
**Parameters:** ~2.5M
**Batch size:** 256

**Time breakdown:**
- Time per batch (H100): ~40ms (deeper network)
- **Per epoch: ~15-18 minutes**
- Validation: ~3 minutes
- **Total per epoch: ~18-21 minutes**

**Full training (10 epochs):**
- **Total time: ~3-3.5 hours**
- **Expected early stopping: epoch 7-8 = 2.5-3 hours**

### Model 3: ChessTransformer-Lite
**Architecture:** 128d model, 2 layers, 4 heads, 512 FFN
**Parameters:** ~0.8M
**Batch size:** 256

**Time breakdown:**
- Time per batch (H100): ~35ms (transformer attention)
- **Per epoch: ~13-16 minutes**
- Validation: ~2 minutes
- **Total per epoch: ~15-18 minutes**

**Full training (10 epochs):**
- **Total time: ~2.5-3 hours**
- **Expected early stopping: epoch 7-8 = 2-2.5 hours**

---

## 📊 Parallel Training Timeline

Since all models run on **separate H100 GPUs in parallel**, they finish simultaneously:

```
Hour 0:00 → Launch all 3 training jobs
Hour 0:05 → All models start training (after setup)
Hour 1:30 → Model 1 (128x6) likely done
Hour 2:30 → Model 3 (Transformer) likely done
Hour 3:00 → Model 2 (128x10) likely done
Hour 3:00 → ALL MODELS COMPLETE ✅
```

**Total wall-clock time: ~3 hours** (with 2-hour buffer for hackathon deadline)

---

## 🔬 Research-Based Recommendations

### Dataset Size Analysis

**From Research:**
- DeepChess: 2M positions, 200 epochs
- Chessformers pre-trained: 3.5M games (estimated 700M positions), 13 epochs
- LeelaZero: Millions to billions of positions, continuous training
- Typical supervised learning: 50-200 epochs for millions of positions

**For 5.5M positions:**
- ✅ **10 epochs is appropriate** (with early stopping)
- ✅ **Sufficient diversity** (from 2200+ ELO games)
- ✅ **Fast iteration** (can retrain if needed)

**If we increase dataset:**
| Dataset Size | Epochs | Training Time (Model 2) | Risk |
|--------------|--------|-------------------------|------|
| 5.5M | 10 | ~3 hours | ✅ LOW |
| 10M | 10 | ~5-6 hours | ⚠️ HIGH (cutting it close) |
| 15M | 10 | ~8-9 hours | ❌ TOO LONG |
| 20M | 10 | ~10-12 hours | ❌ IMPOSSIBLE |

**Conclusion:** 5.5M is the sweet spot for <5 hours constraint.

### Optimal Model Configurations

Based on research from:
- LeelaZero training runs: 64x6, 128x10, 192x15, 256x20
- Chessformers: Various transformer sizes
- Minimal_lczero benchmarks: Smaller models train faster, larger models perform better

**Our 3-model strategy balances:**
1. **Speed** (small model, fast results)
2. **Capacity** (deeper model, better learning)
3. **Diversity** (transformer architecture)

---

## 🚀 Optimal Configurations

### Configuration 1: LC0 Baseline (FASTEST)
**Why:** Fast training, proven architecture, good baseline

```bash
# Model: 128x6 (1.5M params)
modal run training/scripts/train_modal_lc0_v2.py \
  --num-epochs 10 \
  --batch-size 256 \
  --num-filters 128 \
  --num-residual-blocks 6 \
  --hf-repo steveandcow/chesshacks-lc0
```

**Expected:**
- Training time: ~1.5-2 hours
- Target ELO: 1800-2000
- Best for: Reliable baseline, fast iteration

### Configuration 2: LC0 Deeper (STRONGEST)
**Why:** More capacity, better learning, highest expected performance

```bash
# Model: 128x10 (2.5M params)
modal run training/scripts/train_modal_lc0_v2.py \
  --num-epochs 10 \
  --batch-size 256 \
  --num-filters 128 \
  --num-residual-blocks 10 \
  --hf-repo steveandcow/chesshacks-lc0
```

**Expected:**
- Training time: ~2.5-3 hours
- Target ELO: 1900-2100
- Best for: Highest performance, worth the extra time

### Configuration 3: ChessTransformer-Lite (EXPERIMENTAL)
**Why:** Different architecture, might excel at different aspects

**Option A: Use existing transformer training script**
```bash
modal run training/scripts/train_modal.py \
  --config training/configs/transformer_lite.yaml
```

**Option B: Need to check if transformer training script exists**

**Expected:**
- Training time: ~2-2.5 hours
- Target ELO: 1700-1900
- Best for: Diversity, learning what works

---

## 📈 Epochs vs. Dataset Size Trade-off

### Research Insights:

**From LeelaZero:**
- "Use many millions of positions" more important than epoch count
- Self-play reinforcement learning: continuous addition of new data
- Supervised learning on existing data: 50-200 epochs typical

**From Chessformers:**
- 3.5M games → 13 epochs to convergence
- Larger datasets → fewer epochs needed
- Smaller datasets → more epochs needed

**For 5.5M positions (medium size):**
- ✅ **10 epochs with early stopping** (patience=6) is optimal
- ✅ **Data diversity matters more** than pure quantity
- ✅ **High-quality 2200+ ELO data** better than 10x low-quality data

### Should We Increase Dataset?

**NO - Here's why:**

1. **Time constraint:** <5 hours, need 2-hour buffer
2. **Quality > Quantity:** 5.5M high-ELO positions is excellent
3. **Overfitting fixes:** Our v2 pipeline addresses this (proper shuffling, dropout, LR schedule)
4. **Fast iteration:** If models underperform, we can quickly retrain with more data

**If first run succeeds with 2+ hours remaining:**
- ✅ Can launch 2nd round with 10M positions
- ✅ Deploy best model from first round while 2nd trains
- ✅ Iterate based on actual performance

---

## 🎓 Training Calculation Details

### H100 GPU Performance

**Specifications:**
- 80GB HBM3 memory (3TB/s bandwidth)
- 4th-gen Tensor Cores (3x faster than A100 for FP16/BF16)
- Transformer Engine for extra speedup
- TF32 enabled: 2-3x speedup over FP32

**Optimizations Applied:**
- ✅ `torch.compile()` (20-30% speedup)
- ✅ TF32 precision (2-3x speedup)
- ✅ Batch size 256 (optimal for GPU utilization)
- ✅ Mixed precision training (FP16/BF16)
- ✅ Efficient data loading (50k positions/sec pipeline)

### Calculation Example: LC0 128x6

**Per batch:**
- Batch size: 256 positions
- Forward pass: ~10ms (inputs → policy, value, moves_left)
- Backward pass: ~15ms (gradients → weight updates)
- **Total: ~25ms per batch**

**Per epoch:**
- Total positions: 5,500,000
- Batches: 5,500,000 / 256 = 21,484
- Training: 21,484 × 0.025s = 537 seconds = **9 minutes**
- Validation (5% of data): ~1.5 minutes
- Overhead (logging, checkpointing): ~1 minute
- **Total: ~12 minutes per epoch**

**Full training:**
- 10 epochs × 12 min = 120 minutes = **2 hours**
- Early stopping at epoch 7-8: **1.5-2 hours**

### Scaling to Larger Datasets

| Positions | Batches/Epoch | Time/Epoch (128x6) | 10 Epochs |
|-----------|---------------|---------------------|-----------|
| 5.5M | 21,484 | 12 min | 2.0 hours |
| 10M | 39,063 | 22 min | 3.7 hours |
| 15M | 58,594 | 33 min | 5.5 hours ❌ |
| 20M | 78,125 | 44 min | 7.3 hours ❌ |

**Conclusion:** 5.5M fits comfortably, 10M is risky, 15M+ is impossible.

---

## 🔧 Technical Configurations

### LC0 Model Hyperparameters

**Fixed settings (from v2 fixes):**
```python
learning_rate = 0.001
dropout = 0.15  # Increased from 0.1
patience = 6    # Increased from 3
weight_decay = 0.0005
optimizer = "adam"
scheduler = "warmup_cosine"
warmup_epochs = 2
```

**Variable settings (per model):**
```python
# Model 1 (128x6):
num_filters = 128
num_residual_blocks = 6

# Model 2 (128x10):
num_filters = 128
num_residual_blocks = 10
```

### ChessTransformer Hyperparameters

**Recommended configuration:**
```python
d_model = 128          # Embedding dimension
nhead = 4              # Attention heads
num_layers = 2         # Transformer layers
dim_feedforward = 512  # FFN hidden size
dropout = 0.15         # Match LC0 dropout
learning_rate = 0.001  # Match LC0 LR
```

---

## 🎯 Launch Plan (Step-by-Step)

### Pre-flight Checklist

1. ✅ **Data uploaded to Modal volume**
   ```bash
   modal volume ls chess-training-data
   # Should see: /data/lc0_processed/*.npz files
   ```

2. ✅ **HuggingFace token configured**
   ```bash
   modal secret list
   # Should see: huggingface-secret
   ```

3. ✅ **Fixed training scripts ready**
   - `train_modal_lc0_v2.py` (LC0 with all 5 fixes)
   - Need to verify transformer training script

### Launch Sequence

**Step 1: Launch Model 1 (LC0 128x6) - FASTEST**
```bash
# Open terminal 1
modal run training/scripts/train_modal_lc0_v2.py \
  --num-epochs 10 \
  --batch-size 256 \
  --num-filters 128 \
  --num-residual-blocks 6 \
  --hf-repo steveandcow/chesshacks-lc0
```

**Step 2: Launch Model 2 (LC0 128x10) - STRONGEST**
```bash
# Open terminal 2
modal run training/scripts/train_modal_lc0_v2.py \
  --num-epochs 10 \
  --batch-size 256 \
  --num-filters 128 \
  --num-residual-blocks 10 \
  --hf-repo steveandcow/chesshacks-lc0
```

**Step 3: Launch Model 3 (Transformer) - EXPERIMENTAL**
```bash
# Open terminal 3
# Need to verify transformer training script exists
# If not, skip this and do LC0 192x6 instead
modal run training/scripts/train_modal.py \
  --config training/configs/transformer_lite.yaml
```

**Alternative Step 3: If no transformer script, use LC0 192x6**
```bash
modal run training/scripts/train_modal_lc0_v2.py \
  --num-epochs 10 \
  --batch-size 256 \
  --num-filters 192 \
  --num-residual-blocks 6 \
  --hf-repo steveandcow/chesshacks-lc0
```

### Monitoring

**Watch training progress:**
- Each terminal shows real-time training logs
- Monitor train/val loss gap (should be 4-6%, not 12%)
- Check LR schedule (warmup in first 2 epochs)
- Verify epochs are improving (no plateau)

**Expected milestones:**
- **15 min:** All 3 models start training
- **30 min:** Epoch 1 complete, check metrics
- **1.5 hours:** Model 1 (128x6) likely done
- **2.5 hours:** Model 3 likely done
- **3 hours:** Model 2 (128x10) likely done

### After Training

**Verify uploads:**
```bash
# Check HuggingFace repo
# Should see new model files:
# - checkpoints/lc0_v2_128x6_*.pt
# - checkpoints/lc0_v2_128x10_*.pt
# - latest_v2_128x6.pt
# - latest_v2_128x10.pt
```

**Deploy best model:**
1. Check validation losses (lower is better)
2. Deploy to ChessHacks Slot 1 (best model)
3. Deploy to Slot 2 (second best)
4. Deploy to Slot 3 (experimental/third)

---

## 🆘 Contingency Plans

### If Training Fails Early

**Issue: Data not found**
```bash
# Re-upload data
modal run training/scripts/preprocess_modal_lc0.py
```

**Issue: Out of memory**
```bash
# Reduce batch size to 128
--batch-size 128
```

**Issue: Transformer script missing**
```bash
# Fall back to 3rd LC0 model (192x6)
modal run training/scripts/train_modal_lc0_v2.py --num-filters 192
```

### If Time Runs Out

**Priority order:**
1. **Model 2 (128x10)** - Strongest, deploy even if incomplete
2. **Model 1 (128x6)** - Fast, likely finished
3. **Model 3** - Experimental, lowest priority

**Minimum viable:**
- Even 3-4 epochs of Model 2 might beat competitors
- Deploy whatever we have, iterate later

### If Models Finish Early (>2 hours remaining)

**Option 1: Retrain with more data**
```bash
# Add more data chunks, retrain
modal run training/scripts/train_modal_lc0_v2.py \
  --num-epochs 12 \
  --num-filters 128 \
  --num-residual-blocks 10
```

**Option 2: Train larger model**
```bash
# 192x10 or 256x10
modal run training/scripts/train_modal_lc0_v2.py \
  --num-filters 192 \
  --num-residual-blocks 10
```

**Option 3: Fine-tune best model**
```bash
# Continue training from checkpoint
# Lower LR, more epochs
```

---

## 📊 Expected Performance

### ELO Estimates (Based on Research)

| Model | Params | Training Time | Expected ELO | Confidence |
|-------|--------|---------------|--------------|------------|
| LC0 128x6 | 1.5M | 1.5-2h | 1800-2000 | HIGH |
| LC0 128x10 | 2.5M | 2.5-3h | 1900-2100 | HIGH |
| LC0 192x6 | 2.5M | 2-2.5h | 1850-2050 | MEDIUM |
| Transformer-Lite | 0.8M | 2-2.5h | 1700-1900 | MEDIUM |

**Comparison to baselines:**
- Random legal moves: ~600 ELO
- Beginner (material only): ~1000 ELO
- Intermediate (basic tactics): ~1400 ELO
- Advanced (pattern recognition): ~1800 ELO
- Expert (deep calculation): ~2000+ ELO

**Target:** Queen's Crown (Highest ELO) → Need 2000+ ELO

### Success Criteria

**Minimum Success:**
- ✅ At least 1 model trains successfully
- ✅ Model generates legal moves consistently
- ✅ ELO > 1500 (beats random + basic players)

**Good Success:**
- ✅ All 3 models train successfully
- ✅ Best model ELO > 1800
- ✅ Models deployed to all 3 slots

**Excellent Success:**
- ✅ All 3 models complete with <2 hours
- ✅ Best model ELO > 2000
- ✅ Time for 2nd iteration/fine-tuning

---

## 🎯 Final Recommendation

### STICK WITH 5.5M POSITIONS

**Reasoning:**
1. ✅ **Time-safe:** 3 hours training + 2 hours buffer
2. ✅ **Quality data:** 2200+ ELO, well-preprocessed
3. ✅ **Proven approach:** Similar to successful chess AI papers
4. ✅ **Fixed pipeline:** v2 fixes address overfitting
5. ✅ **Fast iteration:** Can retrain if needed

### Launch Strategy

**Now (Immediately):**
1. Launch all 3 models in parallel
2. Monitor first 30 minutes closely
3. Verify training is progressing correctly

**After 1.5 hours:**
1. Check if Model 1 (128x6) finished
2. If yes, deploy to Slot 1 for testing
3. Continue monitoring Models 2 & 3

**After 3 hours:**
1. All models should be complete
2. Deploy best to Slot 1, second-best to Slot 2
3. If time permits, iterate or fine-tune

**If time remaining >2 hours:**
1. Consider 2nd training round with 10M positions
2. Or fine-tune best model with lower LR
3. Or train ensemble of models

---

## 📚 Research Citations

1. **LeelaZero Project:** 2.5B self-play games, various network sizes
2. **Chessformers (Atenrev):** 3.5M games, 13 epochs, transformer architecture
3. **DeepChess:** 2M positions, 200 epochs, supervised learning
4. **minimal_lczero:** 50k positions/sec data pipeline, PyTorch implementation
5. **H100 Performance:** 3-6x faster than A100, TensorFloat32 acceleration

---

## ✅ Action Items

- [ ] Verify Modal data volume has 5.5M positions
- [ ] Confirm HuggingFace token is configured
- [ ] Check if transformer training script exists (if not, use 3rd LC0 config)
- [ ] Open 3 terminal windows for parallel launches
- [ ] Launch all 3 models simultaneously
- [ ] Monitor training progress for first 30 minutes
- [ ] Deploy models as they complete training
- [ ] Test deployed models on ChessHacks platform

**Time to launch: NOW!** 🚀
