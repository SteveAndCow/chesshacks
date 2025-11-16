#!/bin/bash
# Launch All 3 Training Jobs in Parallel
# Run this script to start all 3 models training simultaneously
#
# Usage: bash training/scripts/launch_all_training.sh

set -e

echo "=================================================="
echo "LAUNCHING 3 PARALLEL TRAINING JOBS"
echo "=================================================="
echo ""
echo "⏱️  Expected completion: 2.5-3 hours"
echo "💰 Cost: ~$15-20 for 3x H100 GPUs"
echo ""

# Check Modal is logged in
if ! modal token current &>/dev/null; then
    echo "❌ Modal not authenticated!"
    echo "   Run: modal token new"
    exit 1
fi

echo "✅ Modal authenticated"
echo ""

# Check HuggingFace secret exists
if ! modal secret list | grep -q "huggingface-secret"; then
    echo "⚠️  WARNING: huggingface-secret not found!"
    echo "   Create with: modal secret create huggingface-secret HF_TOKEN=hf_..."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "=================================================="
echo "LAUNCHING MODELS"
echo "=================================================="
echo ""

# Create log directory
mkdir -p training/logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "📊 MODEL 1: LC0 128x6 (Baseline - FASTEST)"
echo "   Expected: 1.5-2 hours"
echo "   Log: training/logs/model1_128x6_${TIMESTAMP}.log"
echo ""
nohup modal run training/scripts/train_modal_lc0_v2.py \
    --num-epochs 10 \
    --batch-size 256 \
    --num-filters 128 \
    --num-residual-blocks 6 \
    --hf-repo steveandcow/chesshacks-lc0 \
    > training/logs/model1_128x6_${TIMESTAMP}.log 2>&1 &
MODEL1_PID=$!
echo "   ✅ Launched (PID: $MODEL1_PID)"
echo ""
sleep 5

echo "📊 MODEL 2: LC0 128x10 (Deeper - STRONGEST)"
echo "   Expected: 2.5-3 hours"
echo "   Log: training/logs/model2_128x10_${TIMESTAMP}.log"
echo ""
nohup modal run training/scripts/train_modal_lc0_v2.py \
    --num-epochs 10 \
    --batch-size 256 \
    --num-filters 128 \
    --num-residual-blocks 10 \
    --hf-repo steveandcow/chesshacks-lc0 \
    > training/logs/model2_128x10_${TIMESTAMP}.log 2>&1 &
MODEL2_PID=$!
echo "   ✅ Launched (PID: $MODEL2_PID)"
echo ""
sleep 5

echo "📊 MODEL 3: Transformer 256x6h8 (Hybrid LC0+Transformer)"
echo "   Expected: 2.5-3 hours"
echo "   Log: training/logs/model3_transformer_${TIMESTAMP}.log"
echo ""
nohup modal run training/scripts/train_modal_transformer_lc0.py \
    --num-epochs 10 \
    --batch-size 256 \
    --num-filters 256 \
    --num-blocks 6 \
    --heads 8 \
    --hf-repo steveandcow/chesshacks-lc0 \
    > training/logs/model3_transformer_${TIMESTAMP}.log 2>&1 &
MODEL3_PID=$!
echo "   ✅ Launched (PID: $MODEL3_PID)"
echo ""

echo "=================================================="
echo "ALL 3 MODELS LAUNCHED!"
echo "=================================================="
echo ""
echo "Process IDs:"
echo "  Model 1 (LC0 128x6):        $MODEL1_PID"
echo "  Model 2 (LC0 128x10):       $MODEL2_PID"
echo "  Model 3 (Transformer 256x6): $MODEL3_PID"
echo ""
echo "Monitor training:"
echo "  tail -f training/logs/model1_128x6_${TIMESTAMP}.log"
echo "  tail -f training/logs/model2_128x10_${TIMESTAMP}.log"
echo "  tail -f training/logs/model3_transformer_${TIMESTAMP}.log"
echo ""
echo "Check Modal dashboard:"
echo "  https://modal.com/apps"
echo ""
echo "Expected completion: ~3 hours from now"
echo "Check back at: $(date -d '+3 hours' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -v+3H '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo '3 hours from now')"
echo ""
echo "✅ Training in progress..."
