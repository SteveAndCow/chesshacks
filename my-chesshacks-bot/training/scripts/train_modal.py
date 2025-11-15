"""
Modal training script for chess models.

Trains models on cloud GPUs and uploads to HuggingFace.
Usage:
    modal run training/scripts/train_modal.py --config training/configs/cnn_baseline.yaml
"""
import modal
from pathlib import Path

# Define Modal app
app = modal.App("chesshacks-training")

# Define container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy",
        "tqdm",
        "pyyaml",
        "huggingface-hub",
        "python-chess",
    )
    .copy_local_dir(
        local_path="training/scripts/models",
        remote_path="/root/models"
    )
    .copy_local_file(
        local_path="training/scripts/model_factory.py",
        remote_path="/root/model_factory.py"
    )
    .copy_local_file(
        local_path="training/scripts/data_loader.py",
        remote_path="/root/data_loader.py"
    )
    .copy_local_file(
        local_path="training/scripts/preprocess.py",
        remote_path="/root/preprocess.py"
    )
)

# Modal Volume for persistent data storage
data_volume = modal.Volume.from_name("chess-training-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",  # or "T4" for cheaper, "H100" for fastest
    timeout=3600 * 4,  # 4 hours max
    volumes={"/data": data_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),  # HF_TOKEN environment variable
    ],
)
def train_model(config_dict: dict):
    """
    Train chess model on Modal GPU.

    Args:
        config_dict: Training configuration dictionary
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import sys
    import os
    from huggingface_hub import HfApi

    # Add models to path
    sys.path.insert(0, "/root")

    from model_factory import create_model_from_config
    from data_loader import create_data_loaders

    print("="*60)
    print("CHESS MODEL TRAINING ON MODAL")
    print("="*60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create model
    print("\n📦 Creating model...")
    model = create_model_from_config(config_dict, device=device)

    # Load data
    print("\n📊 Loading data...")
    data_config = config_dict.get("data", {})
    data_dir = data_config.get("data_dir", "/data/processed")

    # Check if data exists in volume
    if not Path(data_dir).exists():
        print(f"⚠️  Data not found at {data_dir}")
        print("You need to upload preprocessed data to Modal Volume first.")
        print("See training/scripts/upload_data_to_modal.py")
        return {"error": "No training data found"}

    train_loader, val_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=config_dict["training"]["batch_size"],
        train_split=data_config.get("train_split", 0.9),
        num_workers=0  # Modal handles parallelism
    )

    # Setup training
    training_config = config_dict["training"]
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config.get("weight_decay", 0.0001)
    )

    # Learning rate scheduler
    if training_config.get("lr_scheduler") == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config.get("lr_step_size", 5),
            gamma=training_config.get("lr_gamma", 0.1)
        )
    elif training_config.get("lr_scheduler") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config["epochs"],
            eta_min=training_config["lr"] * 0.01
        )
    else:
        scheduler = None

    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    policy_weight = training_config.get("policy_weight", 1.0)
    value_weight = training_config.get("value_weight", 1.0)

    # Training loop
    print(f"\n🎯 Starting training for {training_config['epochs']} epochs...")
    best_val_loss = float('inf')

    for epoch in range(training_config["epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{training_config['epochs']}")
        print(f"{'='*60}")

        # Train
        model.train()
        train_loss = 0
        train_policy_loss = 0
        train_value_loss = 0

        for boards, moves, values in tqdm(train_loader, desc="Training"):
            boards = boards.to(device)
            moves = moves.to(device)
            values = values.to(device).unsqueeze(1)

            optimizer.zero_grad()

            policy_logits, value_pred = model(boards)

            p_loss = policy_criterion(policy_logits, moves)
            v_loss = value_criterion(value_pred, values)
            loss = policy_weight * p_loss + value_weight * v_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_policy_loss += p_loss.item()
            train_value_loss += v_loss.item()

        train_loss /= len(train_loader)
        train_policy_loss /= len(train_loader)
        train_value_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_policy_loss = 0
        val_value_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for boards, moves, values in tqdm(val_loader, desc="Validation"):
                boards = boards.to(device)
                moves = moves.to(device)
                values = values.to(device).unsqueeze(1)

                policy_logits, value_pred = model(boards)

                p_loss = policy_criterion(policy_logits, moves)
                v_loss = value_criterion(value_pred, values)
                loss = policy_weight * p_loss + value_weight * v_loss

                val_loss += loss.item()
                val_policy_loss += p_loss.item()
                val_value_loss += v_loss.item()

                predicted = torch.argmax(policy_logits, dim=1)
                correct += (predicted == moves).sum().item()
                total += moves.size(0)

        val_loss /= len(val_loader)
        val_policy_loss /= len(val_loader)
        val_value_loss /= len(val_loader)
        accuracy = correct / total

        # Log results
        print(f"\nTrain - Loss: {train_loss:.4f}, Policy: {train_policy_loss:.4f}, Value: {train_value_loss:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Policy: {val_policy_loss:.4f}, Value: {val_value_loss:.4f}")
        print(f"Val Accuracy: {accuracy:.4f}")

        # Update learning rate
        if scheduler:
            scheduler.step()
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"✅ New best model! (val_loss: {val_loss:.4f})")

            # Prepare checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_type": config_dict["model"]["type"],
                "model_params": config_dict["model"].get("params", {}),
                "architecture": model.get_architecture_name(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_accuracy": accuracy,
                "config": config_dict,
            }

            # Save locally in container
            model_path = f"/tmp/best_model.pt"
            torch.save(checkpoint, model_path)

    # Training complete - upload to HuggingFace
    print("\n" + "="*60)
    print("📤 Uploading model to HuggingFace...")
    print("="*60)

    hf_config = config_dict.get("huggingface", {})
    repo_id = hf_config.get("repo_id", "your-username/chesshacks-bot")
    model_name = hf_config.get("model_name", "model")

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=f"{model_name}.pt",
            repo_id=repo_id,
            token=os.getenv("HF_TOKEN"),
        )
        print(f"✅ Model uploaded to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️  Failed to upload to HuggingFace: {e}")
        print("Saving model to Modal Volume instead...")
        # Save to volume as backup
        volume_path = f"/data/models/{model_name}.pt"
        Path(volume_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, volume_path)
        data_volume.commit()

    print("\n✅ Training complete!")

    return {
        "model_type": config_dict["model"]["type"],
        "best_val_loss": best_val_loss,
        "final_accuracy": accuracy,
        "epochs_trained": training_config["epochs"],
        "hf_repo": repo_id,
        "model_name": model_name,
    }


@app.local_entrypoint()
def main(config: str = "training/configs/cnn_baseline.yaml"):
    """
    Local entrypoint - runs on your laptop, triggers Modal training.

    Usage:
        modal run training/scripts/train_modal.py --config training/configs/transformer_tiny.yaml
    """
    import yaml
    from pathlib import Path

    # Load config
    config_path = Path(config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config}")
        return

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    print(f"🚀 Launching training job on Modal...")
    print(f"Config: {config}")
    print(f"Model type: {config_dict['model']['type']}")

    # Launch training on Modal
    result = train_model.remote(config_dict)

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE")
    print("="*60)
    print(f"Results: {result}")
