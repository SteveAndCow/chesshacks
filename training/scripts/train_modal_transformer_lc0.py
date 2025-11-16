"""
Modal training script for LeelaZeroTransformer on 112-channel LC0 data.

This transformer architecture uses:
- LC0's 112-channel input format
- Transformer blocks instead of residual blocks
- LC0's policy/value/moves_left heads

Usage:
    modal run training/scripts/train_modal_transformer_lc0.py --num-epochs 10
"""
import modal
from pathlib import Path

# Define Modal app
app = modal.App("chesshacks-training-transformer-lc0")

# Define container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy",
        "tqdm",
        "huggingface-hub",
        "pytorch-lightning",
        "python-chess",
    )
    .add_local_dir(
        local_path="src/models",
        remote_path="/root/models"
    )
    .add_local_file(
        local_path="training/scripts/models/pt_losses.py",
        remote_path="/root/pt_losses.py"
    )
    .add_local_file(
        local_path="training/scripts/data_loader_lc0_v2.py",
        remote_path="/root/data_loader_lc0.py"
    )
)

# Modal Volume for persistent data storage
data_volume = modal.Volume.from_name("chess-training-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600 * 6,
    volumes={"/data": data_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def train_transformer_model(
    num_epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    num_filters: int = 256,
    num_blocks: int = 6,  # transformer depth
    heads: int = 8,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.6,
    moves_left_loss_weight: float = 0.5,
    hf_repo: str = "steveandcow/chesshacks-lc0",
    dropout: float = 0.15,
    max_positions: int = None,  # Limit dataset size (None = use all data)
):
    """
    Train LeelaZeroTransformer on Modal GPU.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        num_filters: Number of filters/embedding dimension
        num_blocks: Number of transformer blocks (depth)
        heads: Number of attention heads
        policy_loss_weight: Weight for policy loss
        value_loss_weight: Weight for value loss
        moves_left_loss_weight: Weight for moves-left loss
        hf_repo: HuggingFace repo ID
        dropout: Dropout rate
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

    from data_loader_lc0 import create_train_val_loaders
    from models.chessformer import LeelaZeroTransformer
    from pt_losses import policy_loss, value_loss, moves_left_loss

    print("="*60)
    print("LC0 TRANSFORMER TRAINING ON MODAL - VERSION 2 (FIXED)")
    print("="*60)
    print("\n🔧 APPLIED FIXES:")
    print("  1. ✅ Proper epoch shuffling (prevents memorization)")
    print("  2. ✅ Slower LR decay with warmup")
    print("  3. ✅ Increased early stopping patience (6 epochs)")
    print("  4. ✅ Increased dropout (0.15 for regularization)")
    print("  5. ✅ Full validation set (no batch limits)")
    print("\n🤖 MODEL: LeelaZeroTransformer (Hybrid LC0 + Transformer)")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Enable TF32 for 2-3x speedup on Ampere/Hopper GPUs (H100, A100)
        torch.set_float32_matmul_precision('high')
        print("⚡ Enabled TensorFloat32 for faster training")

    # Create transformer model
    print("\n📦 Creating LeelaZeroTransformer...")
    model = LeelaZeroTransformer(
        num_filters=num_filters,
        num_residual_blocks=num_blocks,  # Used as transformer depth
        se_ratio=None,  # Not used in transformer
        heads=heads,
        dropout=dropout
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
    data_dir = "/data/lc0_processed_lichess"

    if not Path(data_dir).exists():
        print(f"⚠️  Data not found at {data_dir}")
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
            "train_split": 0.95,
            "shuffle_buffer_size": 100000,
            "num_workers": 0,
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

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0005
    )

    # Better learning rate schedule with warmup
    print("\n📈 Setting up LR scheduler with warmup...")
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=2
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - 2,
        eta_min=learning_rate * 0.1
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[2]
    )

    print(f"  - Warmup: 2 epochs (0.1x → 1.0x LR)")
    print(f"  - Cosine decay: {num_epochs - 2} epochs (1.0x → 0.1x LR)")

    # Training loop
    print(f"\n🎯 Starting training for {num_epochs} epochs...")
    best_val_loss = float('inf')
    model_path = None
    patience = 6
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
                output = model(inputs)
                policy_out, value_out, moves_left_out = output.policy, output.value, output.moves_left

                # Compute losses
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

                output = model(inputs)
                policy_out, value_out, moves_left_out = output.policy, output.value, output.moves_left

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

        val_loss /= val_batches
        val_policy_loss /= val_batches
        val_value_loss /= val_batches
        val_moves_left_loss /= val_batches

        # Log results
        print(f"\nTrain - Loss: {train_loss:.4f}, Policy: {train_policy_loss:.4f}, "
              f"Value: {train_value_loss:.4f}, ML: {train_moves_left_loss:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Policy: {val_policy_loss:.4f}, "
              f"Value: {val_value_loss:.4f}, ML: {val_moves_left_loss:.4f}")

        # Calculate overfitting gap
        overfit_gap = val_loss - train_loss
        overfit_pct = (overfit_gap / train_loss) * 100
        print(f"Train/Val Gap: {overfit_gap:.4f} ({overfit_pct:.1f}% overfitting)")

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lr_ratio = current_lr / learning_rate
        print(f"Learning rate: {current_lr:.6f} ({lr_ratio:.1%} of initial)")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"✅ New best model! (val_loss: {val_loss:.4f})")

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "config": {
                    "num_filters": num_filters,
                    "num_blocks": num_blocks,
                    "heads": heads,
                    "dropout": dropout,
                    "policy_loss_weight": policy_loss_weight,
                    "value_loss_weight": value_loss_weight,
                    "moves_left_loss_weight": moves_left_loss_weight,
                },
            }

            model_path = "/tmp/best_transformer_model.pt"
            torch.save(checkpoint, model_path)
        else:
            patience_counter += 1
            print(f"⚠️  No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                break

    # Upload to HuggingFace
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

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_name = f"transformer_v2_{num_filters}x{num_blocks}h{heads}_ep{epoch+1}_loss{best_val_loss:.4f}_{timestamp}"

        config = {
            "model_architecture": {
                "type": "LeelaZeroTransformer",
                "num_filters": num_filters,
                "num_blocks": num_blocks,
                "heads": heads,
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
            "model_type": "transformer_lc0_v2",
            "fixes_applied": [
                "proper_epoch_shuffling",
                "warmup_lr_schedule",
                "increased_patience_6",
                "increased_dropout_0.15",
                "full_validation"
            ]
        }

        config_path = "/tmp/model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        api = HfApi()

        commit_message = (
            f"Add {model_name} (Transformer FIXED VERSION)\n\n"
            f"Architecture: {num_filters}x{num_blocks} transformer (heads={heads}, dropout={dropout})\n"
            f"Training: {epoch+1} epochs, batch_size={batch_size}, lr={learning_rate}\n"
            f"Best val_loss: {best_val_loss:.4f}\n"
            f"Fixes: epoch_shuffle, warmup_lr, patience_6, dropout_0.15, full_val"
        )

        print(f"📤 Uploading model: {model_name}")

        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=f"checkpoints/{model_name}.pt",
            repo_id=hf_repo,
            token=hf_token,
            commit_message=commit_message,
        )

        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo=f"checkpoints/{model_name}.json",
            repo_id=hf_repo,
            token=hf_token,
            commit_message=f"Add config for {model_name}",
        )

        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=f"latest_transformer_v2_{num_filters}x{num_blocks}h{heads}.pt",
            repo_id=hf_repo,
            token=hf_token,
            commit_message=f"Update latest transformer model",
        )

        print(f"✅ Model uploaded to https://huggingface.co/{hf_repo}")
        print(f"   Versioned: checkpoints/{model_name}.pt")
        print(f"   Latest:    latest_transformer_v2_{num_filters}x{num_blocks}h{heads}.pt")

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
    num_epochs: int = 10,
    batch_size: int = 256,
    num_filters: int = 256,
    num_blocks: int = 6,
    heads: int = 8,
    learning_rate: float = 0.001,
    hf_repo: str = "steveandcow/chesshacks-lc0",
    max_positions: int = None,  # Limit dataset size (None = all)
):
    """
    Launch LeelaZeroTransformer training on Modal.

    Args:
        num_epochs: Number of epochs
        batch_size: Batch size
        num_filters: Embedding dimension (256 recommended)
        num_blocks: Number of transformer blocks (6-12 recommended)
        heads: Number of attention heads (8 recommended)
        learning_rate: Learning rate (0.001 default, 0.0012 for transformers)
        hf_repo: HuggingFace repo ID
        max_positions: Max positions to train on. Examples:
                      - 1000000 (1M): Fast baseline
                      - 3000000 (3M): Medium quality
                      - 7500000 (7.5M): Full dataset
    """
    print(f"🚀 Launching Transformer V2 (FIXED) training on Modal...")
    print(f"Config:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Architecture: {num_filters}x{num_blocks} transformer (heads={heads})")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - HuggingFace repo: {hf_repo}")
    print(f"  - Dropout: 0.15 (increased)")
    print(f"  - Patience: 6 (increased)")
    if max_positions:
        print(f"  - Dataset size: {max_positions:,} positions (LIMITED)")
    else:
        print(f"  - Dataset size: ALL available positions")

    result = train_transformer_model.remote(
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_filters=num_filters,
        num_blocks=num_blocks,
        heads=heads,
        learning_rate=learning_rate,
        hf_repo=hf_repo,
        max_positions=max_positions
    )

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE")
    print("="*60)
    print(f"Results: {result}")
