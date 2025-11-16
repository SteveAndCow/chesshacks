#!/bin/bash
# Pre-Launch Checklist for Training Pipeline
# Run this before launching training to verify everything is ready

echo "=================================================="
echo "PRE-LAUNCH CHECKLIST"
echo "=================================================="
echo ""

ERRORS=0
WARNINGS=0

# Check 1: Modal Authentication
echo "1️⃣  Checking Modal authentication..."
if modal token current &>/dev/null; then
    echo "   ✅ Modal authenticated"
else
    echo "   ❌ Modal NOT authenticated"
    echo "      Run: modal token new"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 2: Data Volume
echo "2️⃣  Checking data volume..."
if modal volume ls chess-training-data lc0_processed &>/dev/null; then
    FILE_COUNT=$(modal volume ls chess-training-data lc0_processed 2>/dev/null | grep -c "\.npz" || echo "0")
    if [ "$FILE_COUNT" -gt 0 ]; then
        echo "   ✅ Data volume exists with $FILE_COUNT .npz files"
    else
        echo "   ⚠️  Data volume exists but no .npz files found"
        echo "      Check: modal volume ls chess-training-data lc0_processed"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "   ❌ Data volume not found or not accessible"
    echo "      Create: modal volume create chess-training-data"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 3: HuggingFace Secret
echo "3️⃣  Checking HuggingFace secret..."
if modal secret list 2>/dev/null | grep -q "huggingface-secret"; then
    echo "   ✅ HuggingFace secret configured"
else
    echo "   ❌ HuggingFace secret not found"
    echo "      Create: modal secret create huggingface-secret HF_TOKEN=hf_..."
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 4: Training Scripts
echo "4️⃣  Checking training scripts..."
SCRIPTS_OK=true

if [ -f "training/scripts/train_modal_lc0_v2.py" ]; then
    echo "   ✅ train_modal_lc0_v2.py exists"
else
    echo "   ❌ train_modal_lc0_v2.py missing"
    SCRIPTS_OK=false
fi

if [ -f "training/scripts/train_modal_transformer_lc0.py" ]; then
    echo "   ✅ train_modal_transformer_lc0.py exists"
else
    echo "   ❌ train_modal_transformer_lc0.py missing"
    SCRIPTS_OK=false
fi

if [ -f "training/scripts/data_loader_lc0_v2.py" ]; then
    echo "   ✅ data_loader_lc0_v2.py exists"
else
    echo "   ❌ data_loader_lc0_v2.py missing"
    SCRIPTS_OK=false
fi

if [ ! "$SCRIPTS_OK" = true ]; then
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 5: Python Syntax
echo "5️⃣  Validating Python syntax..."
SYNTAX_OK=true

if python3 -m py_compile training/scripts/train_modal_lc0_v2.py 2>/dev/null; then
    echo "   ✅ train_modal_lc0_v2.py syntax valid"
else
    echo "   ❌ train_modal_lc0_v2.py has syntax errors"
    SYNTAX_OK=false
fi

if python3 -m py_compile training/scripts/train_modal_transformer_lc0.py 2>/dev/null; then
    echo "   ✅ train_modal_transformer_lc0.py syntax valid"
else
    echo "   ❌ train_modal_transformer_lc0.py has syntax errors"
    SYNTAX_OK=false
fi

if python3 -m py_compile training/scripts/data_loader_lc0_v2.py 2>/dev/null; then
    echo "   ✅ data_loader_lc0_v2.py syntax valid"
else
    echo "   ❌ data_loader_lc0_v2.py has syntax errors"
    SYNTAX_OK=false
fi

if [ ! "$SYNTAX_OK" = true ]; then
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 6: Launch Scripts
echo "6️⃣  Checking launch scripts..."
if [ -f "training/scripts/launch_7_5M_training.sh" ]; then
    echo "   ✅ launch_7_5M_training.sh exists (for 7.5M dataset)"
    chmod +x training/scripts/launch_7_5M_training.sh
fi

if [ -f "training/scripts/launch_all_training.sh" ]; then
    echo "   ✅ launch_all_training.sh exists (for 5.5M dataset)"
    chmod +x training/scripts/launch_all_training.sh
fi
echo ""

# Check 7: Log Directory
echo "7️⃣  Checking log directory..."
if [ -d "training/logs" ]; then
    echo "   ✅ Log directory exists"
else
    echo "   ⚠️  Log directory doesn't exist, will create"
    mkdir -p training/logs
    echo "   ✅ Created training/logs"
fi
echo ""

# Summary
echo "=================================================="
echo "CHECKLIST SUMMARY"
echo "=================================================="
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED"
    echo ""
    echo "Ready to launch training!"
    echo ""
    echo "Choose your dataset size:"
    echo "  - For 7.5M positions (7 epochs, ~3 hours):"
    echo "    bash training/scripts/launch_7_5M_training.sh"
    echo ""
    echo "  - For 5.5M positions (10 epochs, ~3 hours):"
    echo "    bash training/scripts/launch_all_training.sh"
    echo ""
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo "⚠️  $WARNINGS WARNING(S) - Review above"
    echo ""
    echo "You can proceed, but check warnings first"
    exit 0
else
    echo "❌ $ERRORS ERROR(S) - Must fix before launching"
    echo ""
    echo "Fix the errors above before proceeding"
    exit 1
fi
