#!/bin/bash
# Launch All 3 Training Jobs - OPTIMIZED FOR 7.5M POSITIONS
# Uses 7 epochs instead of 10 for better time management
#
# Training time: ~3 hours (vs 4 hours with 10 epochs)
# Same effective training: 7 × 7.5M ≈ 10 × 5.5M
#
# Usage: bash training/scripts/launch_7_5M_training.sh

set -e

echo "=================================================="
echo "LAUNCHING 3 PARALLEL TRAINING JOBS"
echo "OPTIMIZED FOR 7.5M POSITIONS (7 EPOCHS)"
echo "=================================================="
echo ""
echo "⏱️  Expected completion: 3 hours"
echo "📊 Training: 7 epochs × 7.5M = 52.5M position-epochs"
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
echo "LAUNCHING MODELS (7 EPOCHS EACH)"
echo "=================================================="
echo ""

# Create log directory
mkdir -p training/logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "📊 MODEL 1: LC0 128x6 (Baseline - FASTEST)"
echo "   Expected: 1.8-2.0 hours (7 epochs)"
echo "   Log: training/logs/model1_128x6_7ep_${TIMESTAMP}.log"
echo ""
nohup modal run training/scripts/train_modal_lc0_v2.py \
    --num-epochs 7 \
    --batch-size 256 \
    --num-filters 128 \
    --num-residual-blocks 6 \
    --hf-repo steveandcow/chesshacks-lc0 \
    > training/logs/model1_128x6_7ep_${TIMESTAMP}.log 2>&1 &
MODEL1_PID=$!
echo "   ✅ Launched (PID: $MODEL1_PID)"
echo ""
sleep 5

echo "📊 MODEL 2: LC0 128x10 (Deeper - STRONGEST)"
echo "   Expected: 2.8-3.0 hours (7 epochs)"
echo "   Log: training/logs/model2_128x10_7ep_${TIMESTAMP}.log"
echo ""
nohup modal run training/scripts/train_modal_lc0_v2.py \
    --num-epochs 7 \
    --batch-size 256 \
    --num-filters 128 \
    --num-residual-blocks 10 \
    --hf-repo steveandcow/chesshacks-lc0 \
    > training/logs/model2_128x10_7ep_${TIMESTAMP}.log 2>&1 &
MODEL2_PID=$!
echo "   ✅ Launched (PID: $MODEL2_PID)"
echo ""
sleep 5

echo "📊 MODEL 3: Transformer 256x6h8 (Hybrid LC0+Transformer)"
echo "   Expected: 2.5-2.8 hours (7 epochs)"
echo "   Log: training/logs/model3_transformer_7ep_${TIMESTAMP}.log"
echo ""
nohup modal run training/scripts/train_modal_transformer_lc0.py \
    --num-epochs 7 \
    --batch-size 256 \
    --num-filters 256 \
    --num-blocks 6 \
    --heads 8 \
    --hf-repo steveandcow/chesshacks-lc0 \
    > training/logs/model3_transformer_7ep_${TIMESTAMP}.log 2>&1 &
MODEL3_PID=$!
echo "   ✅ Launched (PID: $MODEL3_PID)"
echo ""

echo "=================================================="
echo "ALL 3 MODELS LAUNCHED!"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  Dataset: 7.5M positions"
echo "  Epochs: 7 (instead of 10)"
echo "  Effective training: 52.5M position-epochs"
echo "  (Similar to: 10 epochs × 5.5M = 55M)"
echo ""
echo "Process IDs:"
echo "  Model 1 (LC0 128x6):        $MODEL1_PID"
echo "  Model 2 (LC0 128x10):       $MODEL2_PID"
echo "  Model 3 (Transformer 256x6): $MODEL3_PID"
echo ""
echo "Monitor training:"
echo "  tail -f training/logs/model1_128x6_7ep_${TIMESTAMP}.log"
echo "  tail -f training/logs/model2_128x10_7ep_${TIMESTAMP}.log"
echo "  tail -f training/logs/model3_transformer_7ep_${TIMESTAMP}.log"
echo ""
echo "Check Modal dashboard:"
echo "  https://modal.com/apps"
echo ""
echo "Expected completion: ~3 hours from now"
echo "Check back at: $(date -d '+3 hours' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -v+3H '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo '3 hours from now')"
echo ""
echo "✅ Training in progress..."
