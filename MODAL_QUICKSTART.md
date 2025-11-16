# Modal Training Quick Start

## Step-by-Step Guide to Train on Modal

### **Step 1: Install Modal CLI**

```bash
pip install modal
modal token new  # Opens browser to authenticate
```

### **Step 2: Create Modal Volume**

```bash
modal volume create chess-training-data
```

### **Step 3: Upload Your PGN File**

```bash
# Upload your 288MB dataset
modal volume put chess-training-data training/data/raw/twic_2025_combined.pgn /pgn/twic_2025_combined.pgn

# Verify upload
modal volume ls chess-training-data /pgn
```

**Expected output:**
```
twic_2025_combined.pgn    288 MB
```

### **Step 4: Run Preprocessing on Modal**

```bash
modal run training/scripts/preprocess_modal_lc0.py --min-elo 2000
```

**What this does:**
- Processes your 288MB PGN file (~50k-80k games)
- Filters to ELO 2000+
- Converts to 112-channel format
- Saves as .npz chunks in `/data/lc0_processed/`

**Expected time:** 30-60 minutes
**Expected output:** 1-1.5M positions (~8-10 GB)

**Progress output:**
```
============================================================
PARALLEL PGN PREPROCESSING ON MODAL
============================================================
Min ELO: 2000

✅ Found 1 PGN files
  1. twic_2025_combined.pgn

Starting parallel preprocessing...
Processing twic_2025_combined.pgn...

Games processed: 45,234
Positions saved: 1,234,567
Output chunks: 25
✅ Preprocessing complete!
```

### **Step 5: Verify Preprocessed Data**

```bash
modal volume ls chess-training-data /lc0_processed
```

**Expected output:**
```
twic_2025_combined_chunk0000.npz    400 MB
twic_2025_combined_chunk0001.npz    400 MB
...
```

### **Step 6: Set Up HuggingFace (One-time)**

```bash
# Get token from https://huggingface.co/settings/tokens
# Create with WRITE permission

# Add to Modal secrets
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

### **Step 7: Train on Modal**

```bash
# Small model (fast, for testing)
modal run training/scripts/train_modal_lc0_fixed.py \
    --num-epochs 5 \
    --num-filters 64 \
    --num-residual-blocks 4

# Medium model (recommended for competition)
modal run training/scripts/train_modal_lc0_fixed.py \
    --num-epochs 10 \
    --num-filters 128 \
    --num-residual-blocks 6

# Large model (if you have time)
modal run training/scripts/train_modal_lc0_fixed.py \
    --num-epochs 15 \
    --num-filters 256 \
    --num-residual-blocks 10
```

**Expected time:**
- 64x4: ~2 hours (1700-1900 ELO)
- 128x6: ~6 hours (2000-2200 ELO) ← **Recommended**
- 256x10: ~15 hours (2200-2400 ELO)

**Training output:**
```
============================================================
LC0 MODEL TRAINING ON MODAL
============================================================

🖥️  Device: cuda
GPU: Tesla T4
GPU Memory: 15.00 GB

📦 Creating LC0 model...
Total parameters: 15,234,816

📊 Loading data...
✅ Found 25 .npz files

🎯 Starting training for 10 epochs...

Epoch 1/10
============================================================
Training: 100%|█████| 1000/1000 [08:45<00:00, loss=2.3456]
Validation: 100%|████| 200/200 [00:32<00:00, loss=2.1234]
✅ New best model! (val_loss: 2.1234)

...

🎉 TRAINING COMPLETE
Model uploaded to https://huggingface.co/your-username/chesshacks-lc0
```

---

## Important Notes

### **Policy Encoding: Simplified vs LC0**

The Modal preprocessing uses **simplified 4096 encoding** (from_square * 64 + to_square) instead of LC0's proper 1858-move encoding. This is faster to implement but slightly less accurate.

**To use proper LC0 encoding:**
1. Upload `models/policy_index.py` to Modal
2. Update `move_to_policy_index()` in `preprocess_modal_lc0.py`
3. Use 1858 policy dimension in training

**For hackathon speed, simplified encoding is fine!**

### **GPU Selection**

Modal GPU options (change in `train_modal_lc0_fixed.py` line 48):
- `gpu="T4"`: Cheapest (~$0.60/hr), good for 64x4 and 128x6
- `gpu="A10G"`: Faster (~$1.20/hr), better for 256x10
- `gpu="H100"`: Fastest (~$4/hr), overkill for hackathon

### **Monitoring Training**

Watch logs in real-time:
```bash
modal app logs chesshacks-training-lc0
```

Stop training if needed:
```bash
# Ctrl+C or kill the process
# Model checkpoints are saved every epoch
```

---

## Troubleshooting

### **"No .npz files found"**

Check preprocessing completed:
```bash
modal volume ls chess-training-data /lc0_processed
```

If empty, run preprocessing again.

### **"HF_TOKEN not found"**

Create HuggingFace secret:
```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token
```

### **"Out of GPU memory"**

Reduce batch size in `train_modal_lc0_fixed.py`:
```python
batch_size = 128  # Try 64 or 32
```

### **Upload taking forever**

Compress PGN first:
```bash
gzip training/data/raw/twic_2025_combined.pgn
modal volume put chess-training-data training/data/raw/twic_2025_combined.pgn.gz /pgn/
```

Then update preprocessing to handle `.gz` files.

---

## Quick Reference Commands

```bash
# Check Modal status
modal app list

# View volume contents
modal volume ls chess-training-data /lc0_processed

# Download preprocessed data (optional)
modal volume get chess-training-data /lc0_processed data/lc0_processed_backup

# Check training logs
modal app logs chesshacks-training-lc0

# Delete and retry
modal volume rm chess-training-data /lc0_processed/*
```

---

## Expected Results

After training completes:

✅ **Model saved** to HuggingFace
✅ **Checkpoint** in Modal logs
✅ **Performance** logged (val_loss, accuracy)

**Download for inference:**
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="your-username/chesshacks-lc0",
    filename="lc0_128x6.pt"
)
```

---

## Next Steps After Training

1. **Download model** from HuggingFace
2. **Integrate** into `src/main.py`
3. **Test locally** with test positions
4. **Deploy** to competition slot
5. **Monitor ELO** rating
6. **Iterate** if time permits

---

## Full Workflow (Copy-Paste)

```bash
# 1. Setup (one-time)
pip install modal
modal token new
modal volume create chess-training-data
modal secret create huggingface-secret HF_TOKEN=hf_your_token

# 2. Upload data
modal volume put chess-training-data training/data/raw/twic_2025_combined.pgn /pgn/twic_2025_combined.pgn

# 3. Preprocess
modal run training/scripts/preprocess_modal_lc0.py --min-elo 2000

# 4. Train
modal run training/scripts/train_modal_lc0_fixed.py \
    --num-epochs 10 \
    --num-filters 128 \
    --num-residual-blocks 6

# 5. Done! Model on HuggingFace
```

**Total time:** ~7-8 hours (60 min preprocessing + 6 hours training)

Good luck! 🚀
