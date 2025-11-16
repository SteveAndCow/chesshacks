"""
Modal training script for LC0-format 112-channel chess models - VERSION 2 (FIXED)

CRITICAL FIXES APPLIED:
1. ✅ Proper epoch shuffling (via data_loader_lc0_v2.py)
2. ✅ Slower LR decay with warmup (prevents premature convergence)
3. ✅ Increased early stopping patience (3 → 6 epochs)
4. ✅ Increased dropout (0.1 → 0.15 for better regularization)
5. ✅ Better validation (removed 200 batch limit for more reliable metrics)

Expected improvements:
- Reduced overfitting (train/val gap should drop from 12% to 4-6%)
- Better late-epoch improvements (no more plateau)
- Lower validation loss (-0.5 to -0.8 expected)

Usage:
    modal run training/scripts/train_modal_lc0_v2.py --num-epochs 10
"""
import modal
from pathlib import Path

# Define Modal app
app = modal.App("chesshacks-training-lc0-v2")

# Define container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy",
        "tqdm",
        "huggingface-hub",
        "pytorch-lightning",
    )
    .add_local_dir(
        local_path="training/scripts/models",
        remote_path="/root/models"
    )
    .add_local_file(
        local_path="training/scripts/data_loader_lc0_v2.py",  # FIXED VERSION
        remote_path="/root/data_loader_lc0.py"  # Keep same import name for compatibility
    )
)

# Modal Volume for persistent data storage
data_volume = modal.Volume.from_name("chess-training-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",  # H100 = 8-10x faster than T4, FASTEST for hackathon deadline
    timeout=3600 * 6,  # 6 hours max
    volumes={"/data": data_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def train_lc0_model(
    num_epochs: int = 10,  # Increased from 6 (with better early stopping, can train longer)
    batch_size: int = 256,
    learning_rate: float = 0.001,
    num_filters: int = 128,
    num_residual_blocks: int = 6,
    se_ratio: int = 8,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.6,
    moves_left_loss_weight: float = 0.5,
    q_ratio: float = 0.0,
    hf_repo: str = "steveandcow/chesshacks-lc0",
    dropout: float = 0.15,  # FIX #4: Increased from 0.1 to 0.15 for better regularization
    max_positions: int = None,  # Limit dataset size (None = use all data)
):
    """
    Train LC0 model on Modal GPU with CRITICAL FIXES applied.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        num_filters: Number of filters in conv layers
        num_residual_blocks: Number of residual blocks
        se_ratio: Squeeze-excitation ratio
        policy_loss_weight: Weight for policy loss
        value_loss_weight: Weight for value loss
        moves_left_loss_weight: Weight for moves-left loss
        q_ratio: Ratio for Q vs WDL in value target
        dropout: Dropout rate (increased to 0.15 for better regularization)
        max_positions: Max training positions to use (None = all). Recommended:
                      - 500K-1M: Fast baseline (~30-45 min)
                      - 2M-3M: Medium quality (~1.5-2 hrs)
                      - 5M-7.5M: High quality (~4-5 hrs)
    """
    import torch
    import sys
    import os
    from tqdm import tqdm
    from huggingface_hub import HfApi

    # Add modules to path
    sys.path.insert(0, "/root")

    from data_loader_lc0 import create_train_val_loaders  # Uses v2 (fixed) version
    from models.lccnn import LeelaZeroNet
    from models.pt_losses import policy_loss, value_loss, moves_left_loss

    print("="*60)
    print("LC0 MODEL TRAINING ON MODAL - VERSION 2 (FIXED)")
    print("="*60)
    print("\n🔧 APPLIED FIXES:")
    print("  1. ✅ Proper epoch shuffling (prevents memorization)")
    print("  2. ✅ Slower LR decay with warmup")
    print("  3. ✅ Increased early stopping patience (6 epochs)")
    print("  4. ✅ Increased dropout (0.15 for regularization)")
    print("  5. ✅ Full validation set (no batch limits)")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Enable TF32 for 2-3x speedup on Ampere/Hopper GPUs (H100, A100)
        torch.set_float32_matmul_precision('high')
        print("⚡ Enabled TensorFloat32 for faster training")

    # Create model with increased dropout
    print("\n📦 Creating LC0 model...")
    model = LeelaZeroNet(
        num_filters=num_filters,
        num_residual_blocks=num_residual_blocks,
        se_ratio=se_ratio,
        policy_loss_weight=policy_loss_weight,
        value_loss_weight=value_loss_weight,
        moves_left_loss_weight=moves_left_loss_weight,
        q_ratio=q_ratio,
        optimizer="adam",
        learning_rate=learning_rate,
        dropout=dropout  # FIX #4: Increased dropout for regularization
    ).to(device)

    # Compile model for 20-30% speedup (PyTorch 2.0+)
    print("🔥 Compiling model with torch.compile()...")
    model = torch.compile(model, mode="reduce-overhead")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load data
    print("\n📊 Loading data...")
    data_dir = "/data/lc0_processed"

    if not Path(data_dir).exists():
        print(f"⚠️  Data not found at {data_dir}")
        print("Upload data with: modal run training/scripts/preprocess_modal_lc0.py")
        return {"error": "No training data found"}

    # Check what's in the volume
    print(f"Contents of {data_dir}:")
    for item in sorted(Path(data_dir).iterdir())[:10]:
        print(f"  - {item.name}")

    try:
        # Configure data loader with optional dataset size limit
        loader_kwargs = {
            "data_dir": data_dir,
            "batch_size": batch_size,
            "train_split": 0.95,  # Use more data for training (95% vs 90%)
            "shuffle_buffer_size": 100000,
            "num_workers": 0,  # Modal handles parallelism
            "streaming": True
        }

        # Add max_positions if specified
        if max_positions is not None:
            loader_kwargs["max_positions"] = max_positions
            print(f"📊 Limiting dataset to {max_positions:,} positions")
        else:
            print(f"📊 Using all available positions in dataset")

        train_loader, val_loader = create_train_val_loaders(**loader_kwargs)
    except Exception as e:
        print(f"❌ Failed to create data loaders: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Data loading failed: {e}"}

    # Setup optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0005  # Good regularization
    )

    # FIX #2: Better learning rate schedule with warmup
    print("\n📈 Setting up LR scheduler with warmup...")
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

    # Warmup for first 2 epochs (0.1x → 1.0x learning rate)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=2  # 2 epochs warmup
    )

    # Then cosine decay for remaining epochs (1.0x → 0.1x learning rate)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - 2,  # Remaining epochs after warmup
        eta_min=learning_rate * 0.1  # FIX #2: Only decay to 10% (not 1%)
    )

    # Combine warmup + cosine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[2]  # Switch after 2 epochs
    )

    print(f"  - Warmup: 2 epochs (0.1x → 1.0x LR)")
    print(f"  - Cosine decay: {num_epochs - 2} epochs (1.0x → 0.1x LR)")

    # Training loop with improved early stopping
    print(f"\n🎯 Starting training for {num_epochs} epochs...")
    best_val_loss = float('inf')
    model_path = None
    patience = 6  # FIX #3: Increased from 3 to 6 epochs
    patience_counter = 0

    print(f"  - Early stopping patience: {patience} epochs")

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        # Train
        model.train()
        train_loss = 0
        train_policy_loss = 0
        train_value_loss = 0
        train_moves_left_loss = 0
        num_batches = 0

        try:
            for inputs, policies, values, moves_left in tqdm(train_loader, desc="Training"):
                inputs = inputs.to(device)
                policies = policies.to(device)
                values = values.to(device)
                moves_left = moves_left.to(device)

                optimizer.zero_grad()

                # Forward pass
                policy_out, value_out, moves_left_out = model(inputs)

                # Compute losses using LC0's specialized loss functions
                p_loss = policy_loss(policies, policy_out)
                v_loss = value_loss(values, value_out)
                ml_loss = moves_left_loss(moves_left.unsqueeze(1), moves_left_out)

                # Combined loss
                loss = (
                    policy_loss_weight * p_loss +
                    value_loss_weight * v_loss +
                    moves_left_loss_weight * ml_loss
                )

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_policy_loss += p_loss.item()
                train_value_loss += v_loss.item()
                train_moves_left_loss += ml_loss.item()
                num_batches += 1

                # Train on full dataset each epoch for better generalization

        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Training failed: {e}"}

        train_loss /= num_batches
        train_policy_loss /= num_batches
        train_value_loss /= num_batches
        train_moves_left_loss /= num_batches

        # Validate
        model.eval()
        val_loss = 0
        val_policy_loss = 0
        val_value_loss = 0
        val_moves_left_loss = 0
        val_batches = 0

        with torch.no_grad():
            for inputs, policies, values, moves_left in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                policies = policies.to(device)
                values = values.to(device)
                moves_left = moves_left.to(device)

                policy_out, value_out, moves_left_out = model(inputs)

                p_loss = policy_loss(policies, policy_out)
                v_loss = value_loss(values, value_out)
                ml_loss = moves_left_loss(moves_left.unsqueeze(1), moves_left_out)

                loss = (
                    policy_loss_weight * p_loss +
                    value_loss_weight * v_loss +
                    moves_left_loss_weight * ml_loss
                )

                val_loss += loss.item()
                val_policy_loss += p_loss.item()
                val_value_loss += v_loss.item()
                val_moves_left_loss += ml_loss.item()
                val_batches += 1

                # FIX #5: NO BATCH LIMIT - validate on full validation set
                # This gives more reliable validation metrics for early stopping

        val_loss /= val_batches
        val_policy_loss /= val_batches
        val_value_loss /= val_batches
        val_moves_left_loss /= val_batches

        # Log results
        print(f"\nTrain - Loss: {train_loss:.4f}, Policy: {train_policy_loss:.4f}, "
              f"Value: {train_value_loss:.4f}, ML: {train_moves_left_loss:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Policy: {val_policy_loss:.4f}, "
              f"Value: {val_value_loss:.4f}, ML: {val_moves_left_loss:.4f}")

        # Calculate and display overfitting gap
        overfit_gap = val_loss - train_loss
        overfit_pct = (overfit_gap / train_loss) * 100
        print(f"Train/Val Gap: {overfit_gap:.4f} ({overfit_pct:.1f}% overfitting)")

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lr_ratio = current_lr / learning_rate
        print(f"Learning rate: {current_lr:.6f} ({lr_ratio:.1%} of initial)")

        # Save best model and handle early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter on improvement
            print(f"✅ New best model! (val_loss: {val_loss:.4f})")

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "config": {
                    "num_filters": num_filters,
                    "num_residual_blocks": num_residual_blocks,
                    "se_ratio": se_ratio,
                    "dropout": dropout,
                    "policy_loss_weight": policy_loss_weight,
                    "value_loss_weight": value_loss_weight,
                    "moves_left_loss_weight": moves_left_loss_weight,
                },
            }

            model_path = "/tmp/best_lc0_model.pt"
            torch.save(checkpoint, model_path)
        else:
            patience_counter += 1
            print(f"⚠️  No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                break

    # Training complete
    print("\n" + "="*60)
    print("📤 Uploading model to HuggingFace...")
    print("="*60)

    if model_path is None:
        print("⚠️  No model checkpoint saved!")
        return {"error": "No model checkpoint created"}

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠️  HF_TOKEN not found!")
        return {"error": "No HuggingFace token"}

    try:
        import json
        from datetime import datetime

        # Create versioned model name with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_name = f"lc0_v2_{num_filters}x{num_residual_blocks}_ep{epoch+1}_loss{best_val_loss:.4f}_{timestamp}"

        # Create config with ALL hyperparameters
        config = {
            "model_architecture": {
                "num_filters": num_filters,
                "num_residual_blocks": num_residual_blocks,
                "se_ratio": se_ratio,
                "dropout": dropout,
            },
            "training_hyperparameters": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": 0.0005,
                "num_epochs": num_epochs,
                "epochs_trained": epoch + 1,
                "optimizer": "adam",
                "scheduler": "warmup_cosine",
                "warmup_epochs": 2,
                "min_lr_ratio": 0.1,
            },
            "loss_weights": {
                "policy_loss_weight": policy_loss_weight,
                "value_loss_weight": value_loss_weight,
                "moves_left_loss_weight": moves_left_loss_weight,
                "q_ratio": q_ratio,
            },
            "training_results": {
                "best_val_loss": float(best_val_loss),
                "best_epoch": epoch + 1,
                "early_stopped": epoch + 1 < num_epochs,
            },
            "data": {
                "train_split": 0.95,
                "shuffle_buffer_size": 100000,
                "streaming": True,
            },
            "timestamp": timestamp,
            "model_type": "lc0_minimal_v2",
            "fixes_applied": [
                "proper_epoch_shuffling",
                "warmup_lr_schedule",
                "increased_patience_6",
                "increased_dropout_0.15",
                "full_validation"
            ]
        }

        # Save config to file
        config_path = "/tmp/model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Upload both model and config
        api = HfApi()

        # Create commit message with key metrics
        commit_message = (
            f"Add {model_name} (FIXED VERSION)\n\n"
            f"Architecture: {num_filters}x{num_residual_blocks} (SE={se_ratio}, dropout={dropout})\n"
            f"Training: {epoch+1} epochs, batch_size={batch_size}, lr={learning_rate}\n"
            f"Best val_loss: {best_val_loss:.4f}\n"
            f"Fixes: epoch_shuffle, warmup_lr, patience_6, dropout_0.15, full_val\n"
            f"Loss weights: policy={policy_loss_weight}, value={value_loss_weight}, ml={moves_left_loss_weight}"
        )

        print(f"📤 Uploading model: {model_name}")
        print(f"📝 Config: {json.dumps(config, indent=2)}")

        # Upload model weights
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=f"checkpoints/{model_name}.pt",
            repo_id=hf_repo,
            token=hf_token,
            commit_message=commit_message,
        )

        # Upload config
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo=f"checkpoints/{model_name}.json",
            repo_id=hf_repo,
            token=hf_token,
            commit_message=f"Add config for {model_name}",
        )

        # Also upload as "latest_v2" for easy inference
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=f"latest_v2_{num_filters}x{num_residual_blocks}.pt",
            repo_id=hf_repo,
            token=hf_token,
            commit_message=f"Update latest v2 {num_filters}x{num_residual_blocks} model",
        )

        print(f"✅ Model uploaded to https://huggingface.co/{hf_repo}")
        print(f"   Versioned: checkpoints/{model_name}.pt")
        print(f"   Latest:    latest_v2_{num_filters}x{num_residual_blocks}.pt")

        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "hf_repo": hf_repo,
            "model_name": model_name,
            "model_path": f"checkpoints/{model_name}.pt",
            "config_path": f"checkpoints/{model_name}.json",
        }

    except Exception as e:
        print(f"⚠️  Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "error": f"Upload failed: {e}"
        }


@app.local_entrypoint()
def main(
    num_epochs: int = 10,  # Increased from 6 (better early stopping allows longer training)
    batch_size: int = 256,
    num_filters: int = 128,
    num_residual_blocks: int = 6,
    hf_repo: str = "steveandcow/chesshacks-lc0",
    max_positions: int = None,  # Limit dataset size (None = all)
):
    """
    Launch LC0 training on Modal with CRITICAL FIXES.

    Args:
        num_epochs: Number of epochs
        batch_size: Batch size
        num_filters: Number of filters (128 recommended)
        num_residual_blocks: Number of blocks (6-10 recommended)
        hf_repo: HuggingFace repo ID (e.g., "username/chesshacks-lc0")
        max_positions: Max positions to train on. Examples:
                      - 1000000 (1M): Fast baseline
                      - 2000000 (2M): Medium quality
                      - 7500000 (7.5M): Full dataset
    """
    print(f"🚀 Launching LC0 V2 (FIXED) training on Modal...")
    print(f"Config:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Architecture: {num_filters}x{num_residual_blocks}")
    print(f"  - HuggingFace repo: {hf_repo}")
    print(f"  - Dropout: 0.15 (increased)")
    print(f"  - Patience: 6 (increased)")
    print(f"  - LR schedule: warmup + slower cosine")
    if max_positions:
        print(f"  - Dataset size: {max_positions:,} positions (LIMITED)")
    else:
        print(f"  - Dataset size: ALL available positions")

    result = train_lc0_model.remote(
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_filters=num_filters,
        num_residual_blocks=num_residual_blocks,
        hf_repo=hf_repo,
        max_positions=max_positions
    )

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE")
    print("="*60)
    print(f"Results: {result}")
