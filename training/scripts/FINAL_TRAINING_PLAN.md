# Final Training Plan: <5 Hours to Hackathon Deadline

**Created:** 2025-11-16
**Dataset:** 5.5M positions (ready to use)
**Time Remaining:** <5 hours
**Strategy:** 3 LC0 models in parallel

---

## 🎯 Executive Decision: LAUNCH NOW

### Why 3 LC0 Models (Not 2 LC0 + 1 Transformer)?

**Data format mismatch:**
- LC0 uses **112-channel** input format (at `/data/lc0_processed`)
- Transformer uses **16-channel** input format (at `/data/processed`)
- We have 5.5M positions in LC0 format (ready)
- Transformer data would require preprocessing (1-2 hours lost)

**Time-optimal strategy:**
- ✅ **3 LC0 models** with different architectures
- ✅ **All use same data** (no preprocessing delay)
- ✅ **Launch immediately** (no setup time wasted)
- ✅ **Proven approach** (LC0 is battle-tested)

---

## 📊 Final Model Selection

### Model 1: LC0 128x6 (Baseline - FASTEST)
**Why:** Fast training, reliable results, good baseline
- **Architecture:** 128 filters, 6 residual blocks
- **Parameters:** ~1.5M
- **Training time:** 1.5-2 hours
- **Expected ELO:** 1800-2000

### Model 2: LC0 128x10 (Deeper - STRONGEST)
**Why:** More capacity, better learning, highest expected performance
- **Architecture:** 128 filters, 10 residual blocks
- **Parameters:** ~2.5M
- **Training time:** 2.5-3 hours
- **Expected ELO:** 1900-2100

### Model 3: LC0 192x6 (Wider - ALTERNATIVE)
**Why:** Different architecture, more filters, good middle ground
- **Architecture:** 192 filters, 6 residual blocks
- **Parameters:** ~2.5M
- **Training time:** 2-2.5 hours
- **Expected ELO:** 1850-2050

---

## ⏱️ Timeline (3 Hours Total)

```
Hour 0:00 → Launch all 3 models (use launch script)
Hour 0:05 → All models start training
Hour 1:30 → Model 1 (128x6) completes ✅
Hour 1:35 → Deploy Model 1 to Slot 1 for testing
Hour 2:00 → Model 3 (192x6) completes ✅
Hour 2:05 → Deploy Model 3 to Slot 2
Hour 2:45 → Model 2 (128x10) completes ✅
Hour 2:50 → Deploy Model 2 (if better, swap with Slot 1)
Hour 3:00 → All training complete, models deployed
Hour 3:00-5:00 → Test, iterate, fine-tune if time permits
```

---

## 🚀 Launch Commands

### Option 1: Launch All at Once (RECOMMENDED)

```bash
# Make script executable
chmod +x training/scripts/launch_all_training.sh

# Launch all 3 models
bash training/scripts/launch_all_training.sh
```

This will:
- ✅ Launch all 3 models in parallel
- ✅ Create log files for each model
- ✅ Show progress in separate log files
- ✅ Complete in ~3 hours

### Option 2: Launch Individually

**Terminal 1: Model 1 (128x6)**
```bash
modal run training/scripts/train_modal_lc0_v2.py \
  --num-epochs 10 \
  --batch-size 256 \
  --num-filters 128 \
  --num-residual-blocks 6 \
  --hf-repo steveandcow/chesshacks-lc0
```

**Terminal 2: Model 2 (128x10)**
```bash
modal run training/scripts/train_modal_lc0_v2.py \
  --num-epochs 10 \
  --batch-size 256 \
  --num-filters 128 \
  --num-residual-blocks 10 \
  --hf-repo steveandcow/chesshacks-lc0
```

**Terminal 3: Model 3 (192x6)**
```bash
modal run training/scripts/train_modal_lc0_v2.py \
  --num-epochs 10 \
  --batch-size 256 \
  --num-filters 192 \
  --num-residual-blocks 6 \
  --hf-repo steveandcow/chesshacks-lc0
```

---

## 📝 Pre-flight Checklist

### Before Launching

- [ ] **Modal authenticated**
  ```bash
  modal token current
  # If not: modal token new
  ```

- [ ] **Data exists in Modal volume**
  ```bash
  modal volume ls chess-training-data lc0_processed
  # Should show .npz files
  ```

- [ ] **HuggingFace secret configured**
  ```bash
  modal secret list
  # Should show: huggingface-secret
  # If not: modal secret create huggingface-secret HF_TOKEN=hf_...
  ```

- [ ] **Training scripts up to date**
  ```bash
  ls -lh training/scripts/train_modal_lc0_v2.py
  ls -lh training/scripts/data_loader_lc0_v2.py
  ```

### After Launching

- [ ] **Verify all 3 models started**
  - Check Modal dashboard: https://modal.com/apps
  - Should see 3 active runs

- [ ] **Monitor first 15 minutes**
  - Check logs for errors
  - Verify training is progressing
  - Confirm LR warmup is working

- [ ] **Set reminder for 1.5 hours**
  - Model 1 should be done
  - Download and deploy to test

---

## 🔍 Monitoring Training

### Watch Logs (If using launch script)

```bash
# Model 1 (128x6)
tail -f training/logs/model1_128x6_*.log

# Model 2 (128x10)
tail -f training/logs/model2_128x10_*.log

# Model 3 (192x6)
tail -f training/logs/model3_192x6_*.log
```

### What to Look For

**✅ Good signs:**
- Train loss decreasing steadily
- Val loss following train loss
- Train/val gap 4-6% (not 12%!)
- LR warmup in first 2 epochs
- Later epochs still improving (no plateau)

**❌ Bad signs:**
- Val loss increasing while train decreases
- Train/val gap >10% (overfitting)
- Training stuck/not progressing
- Out of memory errors

### Key Metrics to Track

| Epoch | Train Loss | Val Loss | Gap | LR | Notes |
|-------|-----------|----------|-----|-----|-------|
| 1 | ~4.15 | ~4.45 | 0.30 | 0.00055 | Warmup |
| 2 | ~3.95 | ~4.15 | 0.20 | 0.00100 | Full LR |
| 3 | ~3.75 | ~3.95 | 0.20 | 0.00090 | Improving |
| 4 | ~3.60 | ~3.75 | 0.15 | 0.00075 | Good gap |
| 5 | ~3.50 | ~3.60 | 0.10 | 0.00060 | Still improving! |
| 6 | ~3.45 | ~3.55 | 0.10 | 0.00045 | On track |
| 7-8 | ~3.40 | ~3.50 | 0.10 | 0.00030 | May early stop |

---

## 📦 After Training Completes

### Download Models from HuggingFace

Models will be automatically uploaded to:
- `https://huggingface.co/steveandcow/chesshacks-lc0`

Files uploaded:
- `checkpoints/lc0_v2_128x6_ep*_loss*.pt`
- `checkpoints/lc0_v2_128x10_ep*_loss*.pt`
- `checkpoints/lc0_v2_192x6_ep*_loss*.pt`
- `latest_v2_128x6.pt`
- `latest_v2_128x10.pt`
- `latest_v2_192x6.pt`

### Deploy to ChessHacks Platform

**Priority order (based on validation loss):**

1. **Slot 1:** Best model (lowest val_loss)
2. **Slot 2:** Second best
3. **Slot 3:** Third/experimental

**Deployment steps:**
1. Go to ChessHacks dashboard
2. Connect GitHub repo
3. Select bot slot (1, 2, or 3)
4. Wait for build (max 3 minutes)
5. Monitor ELO rating

---

## 🎲 Decision Tree

### If All 3 Models Train Successfully

✅ **Best case scenario!**

**Actions:**
1. Deploy all 3 to separate slots
2. Monitor ELO ratings for 30-60 minutes
3. Identify best performer
4. If time permits (>1 hour), fine-tune best model

### If Only 2 Models Complete

⚠️ **Still good!**

**Actions:**
1. Deploy completed models to Slot 1 & 2
2. Wait for 3rd model or cancel it
3. Focus on deploying and testing completed models

### If Only 1 Model Completes

⚠️ **Minimum viable!**

**Actions:**
1. Deploy completed model to Slot 1
2. Troubleshoot failed models
3. If time permits, launch simplified versions

### If All Models Fail

❌ **Debug mode!**

**Troubleshooting:**
1. Check Modal logs for errors
2. Verify data exists: `modal volume ls chess-training-data`
3. Test with smaller dataset: `--num-epochs 2`
4. Reduce batch size: `--batch-size 128`
5. Try single model first, then parallel

---

## 💰 Cost Estimate

### H100 Pricing (Modal)
- ~$5-6 per GPU hour
- 3 GPUs × 3 hours = 9 GPU hours
- **Total: $45-54**

### Budget Options

**If cost is concern:**
```bash
# Use A100 instead (3x cheaper, 3x slower)
# Edit train_modal_lc0_v2.py:
# Change: gpu="H100" → gpu="A100"
# Expected time: ~6-9 hours (might not fit in deadline!)
```

**Recommendation:** Stick with H100 for speed given time constraint.

---

## 🆘 Emergency Procedures

### If Training Too Slow (After 1 Hour)

**Problem:** Model 1 not done after 1.5 hours

**Solution:**
1. Check Modal dashboard for GPU utilization
2. Verify data loading isn't bottleneck
3. Reduce epochs: Kill job, relaunch with `--num-epochs 5`
4. Continue with partial training

### If Out of Memory

**Problem:** CUDA out of memory error

**Solution:**
```bash
# Reduce batch size
--batch-size 128  # Instead of 256
```

### If Data Not Found

**Problem:** "No .npz files found"

**Solution:**
```bash
# Check volume
modal volume ls chess-training-data lc0_processed

# If empty, preprocess data
modal run training/scripts/preprocess_modal_lc0.py
```

### If HuggingFace Upload Fails

**Problem:** Upload failed, but model trained

**Solution:**
1. Model checkpoint saved locally in Modal
2. Can re-upload manually
3. Or continue with next model and fix later

---

## ✨ Success Metrics

### Minimum Success
- ✅ At least 1 model completes training
- ✅ Model deploys to ChessHacks platform
- ✅ Bot plays legal moves
- ✅ ELO > 1500

### Good Success
- ✅ All 3 models complete training
- ✅ Best model ELO > 1800
- ✅ Models deployed to all slots
- ✅ No illegal moves

### Excellent Success
- ✅ All models done with 2+ hours to spare
- ✅ Best model ELO > 2000
- ✅ Time for fine-tuning or 2nd iteration
- ✅ Competitive for Queen's Crown (Highest ELO)

---

## 📊 Research Summary

### Why 5.5M Positions is Optimal

**From literature:**
- DeepChess: 2M positions → good results
- Chessformers: 3.5M games → competitive
- LeelaZero: Millions → strong play
- Our data: 5.5M high-quality (2200+ ELO) positions

**Quality indicators:**
- ✅ All games from 2200+ ELO players
- ✅ Diverse openings and positions
- ✅ Clean preprocessing (112-channel format)
- ✅ Proven data pipeline

### Why 10 Epochs is Right

**From research:**
- Small datasets (2M): 200 epochs needed
- Medium datasets (5-10M): 50-100 epochs typical
- Large datasets (50M+): 10-20 epochs sufficient
- With early stopping: Let model decide

**Our approach:**
- ✅ 10 epochs maximum
- ✅ Early stopping (patience=6)
- ✅ Expected convergence: epoch 7-8
- ✅ Prevents overfitting with our fixes

---

## 🎯 Final Recommendations

### DO THIS NOW

1. **✅ Launch all 3 models immediately**
   ```bash
   bash training/scripts/launch_all_training.sh
   ```

2. **✅ Monitor first 15 minutes**
   - Verify all 3 started successfully
   - Check logs for errors

3. **✅ Set timer for 1.5 hours**
   - Model 1 should be ready to deploy

4. **✅ Prepare deployment**
   - Have ChessHacks dashboard ready
   - Test deployment process with old model if available

### DON'T DO THIS

- ❌ **Don't wait for more data** - 5.5M is enough
- ❌ **Don't preprocess transformer data** - wastes 1-2 hours
- ❌ **Don't use larger models** - 128x10 is max for time
- ❌ **Don't train locally** - H100 is 10x+ faster
- ❌ **Don't train sequentially** - parallel is key

---

## 🏁 Race to Finish Line

**Current time:** Now
**Deadline:** <5 hours
**Training time:** ~3 hours
**Buffer:** 2 hours for deployment, testing, debugging

**Timeline:**
```
Now        → Launch training (0:00)
+1.5h      → Model 1 done, deploy (1:30)
+2.5h      → Model 3 done, deploy (2:30)
+3h        → Model 2 done, deploy (3:00)
+3-5h      → Test, iterate, optimize (3:00-5:00)
Deadline   → Submit best model (5:00)
```

**Status:** ✅ ON TRACK

---

## 📚 Files Reference

**Training scripts:**
- `train_modal_lc0_v2.py` - Fixed LC0 training (use this!)
- `data_loader_lc0_v2.py` - Fixed data loader
- `launch_all_training.sh` - Parallel launch script

**Documentation:**
- `LC0_V2_FIXES.md` - Detailed fixes explanation
- `PARALLEL_TRAINING_STRATEGY.md` - Full strategy analysis
- `FINAL_TRAINING_PLAN.md` - This file (quick reference)

**Models:**
- `/training/scripts/models/lccnn.py` - LC0 architecture
- `/training/scripts/models/pt_layers.py` - Network layers
- `/training/scripts/models/pt_losses.py` - Loss functions

---

## ✅ Ready to Launch?

**Pre-flight complete?**
- [ ] Modal authenticated
- [ ] Data in volume
- [ ] HuggingFace secret set
- [ ] Scripts up to date

**Understand the plan?**
- [ ] 3 models in parallel
- [ ] ~3 hours training time
- [ ] Deploy as they complete
- [ ] Monitor and iterate

**Committed to timeline?**
- [ ] Launch immediately
- [ ] Check after 1.5 hours
- [ ] Deploy when ready
- [ ] No scope creep!

---

## 🚀 LAUNCH COMMAND

```bash
bash training/scripts/launch_all_training.sh
```

**Good luck! Let's win Queen's Crown! 👑**
