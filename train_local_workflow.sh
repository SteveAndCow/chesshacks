#!/bin/bash

# Local training workflow for M1 Mac
# Preprocesses subset of data and trains a small model

set -e  # Exit on error

echo "=================================================="
echo "LOCAL LC0 TRAINING WORKFLOW (M1 Mac)"
echo "=================================================="

# Activate virtual environment
source .venv/bin/activate

# Configuration
PGN_INPUT="training/data/raw/twic_2025_combined.pgn"
PROCESSED_OUTPUT="training/data/lc0_processed/twic_2025.npz"
MIN_ELO=2000
NUM_EPOCHS=5
BATCH_SIZE=128
NUM_FILTERS=64
NUM_BLOCKS=4

echo ""
echo "Configuration:"
echo "  Input PGN: $PGN_INPUT"
echo "  Output: $PROCESSED_OUTPUT"
echo "  Min ELO: $MIN_ELO"
echo "  Epochs: $NUM_EPOCHS"
echo "  Model: ${NUM_FILTERS}x${NUM_BLOCKS}"
echo ""

# Step 1: Preprocess PGN
echo "=================================================="
echo "Step 1: Preprocessing PGN to 112-channel format"
echo "=================================================="

if [ -f "$PROCESSED_OUTPUT" ]; then
    echo "⚠️  Processed file already exists: $PROCESSED_OUTPUT"
    read -p "Delete and reprocess? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm "$PROCESSED_OUTPUT"
        echo "Deleted existing file"
    else
        echo "Skipping preprocessing"
    fi
fi

if [ ! -f "$PROCESSED_OUTPUT" ]; then
    echo "Processing $PGN_INPUT..."
    python training/scripts/preprocess_pgn_to_lc0.py \
        --input "$PGN_INPUT" \
        --output "$PROCESSED_OUTPUT" \
        --min-elo $MIN_ELO

    echo "✅ Preprocessing complete"
else
    echo "✅ Using existing preprocessed data"
fi

# Check file size
FILE_SIZE=$(du -h "$PROCESSED_OUTPUT" | cut -f1)
echo "Processed file size: $FILE_SIZE"

# Step 2: Train model
echo ""
echo "=================================================="
echo "Step 2: Training LC0 model"
echo "=================================================="

python training/scripts/train_local_lc0.py \
    --data-dir "$(dirname $PROCESSED_OUTPUT)" \
    --epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --filters $NUM_FILTERS \
    --blocks $NUM_BLOCKS \
    --max-batches 500

echo ""
echo "=================================================="
echo "✅ WORKFLOW COMPLETE!"
echo "=================================================="
echo ""
echo "Model saved to: training/models/"
echo ""
echo "Next steps:"
echo "  1. Check model: ls -lh training/models/"
echo "  2. Test inference: python training/scripts/test_inference.py"
echo "  3. If good, train larger model on Modal"
echo ""
