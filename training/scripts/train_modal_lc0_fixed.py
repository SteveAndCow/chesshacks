"""
Modal training script for LC0-format 112-channel chess models.

Trains models on cloud GPUs using preprocessed .npz data and uploads to HuggingFace.

Usage:
    modal run training/scripts/train_modal_lc0_fixed.py --num-epochs 10
"""
import modal
from pathlib import Path

# Define Modal app
app = modal.App("chesshacks-training-lc0")

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
        local_path="training/scripts/data_loader_lc0.py",
        remote_path="/root/data_loader_lc0.py"
    )
)

# Modal Volume for persistent data storage
data_volume = modal.Volume.from_name("chess-training-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",  # or "A10G" for faster, "H100" for fastest
    timeout=3600 * 6,  # 6 hours max
    volumes={"/data": data_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def train_lc0_model(
    num_epochs: int = 10,
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
):
    """
    Train LC0 model on Modal GPU.

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
    """
    import torch
    import sys
    import os
    from tqdm import tqdm
    from huggingface_hub import HfApi

    # Add modules to path
    sys.path.insert(0, "/root")

    from data_loader_lc0 import create_train_val_loaders
    from models.lccnn import LeelaZeroNet
    from models.pt_losses import policy_loss, value_loss, moves_left_loss

    print("="*60)
    print("LC0 MODEL TRAINING ON MODAL")
    print("="*60)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create model
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
        learning_rate=learning_rate
    ).to(device)

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
        train_loader, val_loader = create_train_val_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            train_split=0.9,
            shuffle_buffer_size=100000,
            num_workers=0,  # Modal handles parallelism
            streaming=True
        )
    except Exception as e:
        print(f"❌ Failed to create data loaders: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Data loading failed: {e}"}

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0001
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=learning_rate * 0.01
    )

    # Training loop
    print(f"\n🎯 Starting training for {num_epochs} epochs...")
    best_val_loss = float('inf')
    model_path = None

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

                # Limit to 1000 batches per epoch for faster iteration during hackathon
                if num_batches >= 1000:
                    break

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

                # Limit validation batches
                if val_batches >= 200:
                    break

        val_loss /= val_batches
        val_policy_loss /= val_batches
        val_value_loss /= val_batches
        val_moves_left_loss /= val_batches

        # Log results
        print(f"\nTrain - Loss: {train_loss:.4f}, Policy: {train_policy_loss:.4f}, "
              f"Value: {train_value_loss:.4f}, ML: {train_moves_left_loss:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Policy: {val_policy_loss:.4f}, "
              f"Value: {val_value_loss:.4f}, ML: {val_moves_left_loss:.4f}")

        # Update learning rate
        scheduler.step()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"✅ New best model! (val_loss: {val_loss:.4f})")

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "config": {
                    "num_filters": num_filters,
                    "num_residual_blocks": num_residual_blocks,
                    "se_ratio": se_ratio,
                    "policy_loss_weight": policy_loss_weight,
                    "value_loss_weight": value_loss_weight,
                    "moves_left_loss_weight": moves_left_loss_weight,
                },
            }

            model_path = "/tmp/best_lc0_model.pt"
            torch.save(checkpoint, model_path)

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
        model_name = f"lc0_{num_filters}x{num_residual_blocks}"

        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=f"{model_name}.pt",
            repo_id=hf_repo,
            token=hf_token,
        )

        print(f"✅ Model uploaded to https://huggingface.co/{hf_repo}")

        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": num_epochs,
            "hf_repo": hf_repo,
            "model_name": model_name,
        }

    except Exception as e:
        print(f"⚠️  Upload failed: {e}")
        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": num_epochs,
            "error": f"Upload failed: {e}"
        }


@app.local_entrypoint()
def main(
    num_epochs: int = 10,
    batch_size: int = 256,
    num_filters: int = 128,
    num_residual_blocks: int = 6,
    hf_repo: str = "steveandcow/chesshacks-lc0",
):
    """
    Launch LC0 training on Modal.

    Args:
        num_epochs: Number of epochs
        batch_size: Batch size
        num_filters: Number of filters (128 recommended)
        num_residual_blocks: Number of blocks (6-10 recommended)
        hf_repo: HuggingFace repo ID (e.g., "username/chesshacks-lc0")
    """
    print(f"🚀 Launching LC0 training on Modal...")
    print(f"Config:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Architecture: {num_filters}x{num_residual_blocks}")
    print(f"  - HuggingFace repo: {hf_repo}")

    result = train_lc0_model.remote(
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_filters=num_filters,
        num_residual_blocks=num_residual_blocks,
        hf_repo=hf_repo
    )

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE")
    print("="*60)
    print(f"Results: {result}")
