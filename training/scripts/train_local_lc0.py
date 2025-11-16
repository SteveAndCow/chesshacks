"""
Local training script for LC0 models on M1 Mac.

Uses MPS (Metal Performance Shaders) for GPU acceleration on Apple Silicon.

Usage:
    python training/scripts/train_local_lc0.py \
        --data-dir training/data/lc0_processed \
        --epochs 5 \
        --batch-size 128
"""
import torch
import sys
from pathlib import Path
from tqdm import tqdm
import argparse

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader_lc0 import create_train_val_loaders
from models.lccnn import LeelaZeroNet
from models.pt_losses import policy_loss, value_loss, moves_left_loss


def get_device():
    """Get best available device for M1 Mac."""
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_local(
    data_dir: str,
    num_epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    num_filters: int = 64,
    num_residual_blocks: int = 4,
    save_dir: str = "training/models",
    max_batches_per_epoch: int = 500,
):
    """
    Train LC0 model locally on M1 Mac.

    Args:
        data_dir: Directory with .npz files
        num_epochs: Number of epochs
        batch_size: Batch size (128 recommended for M1)
        learning_rate: Learning rate
        num_filters: Number of filters (64 recommended for local)
        num_residual_blocks: Number of blocks (4 recommended for local)
        save_dir: Where to save model checkpoints
        max_batches_per_epoch: Limit batches per epoch for faster iteration
    """
    print("="*60)
    print("LOCAL LC0 TRAINING ON M1 MAC")
    print("="*60)

    # Setup device
    device = get_device()
    print(f"\n🖥️  Device: {device}")
    if device.type == "mps":
        print("✅ Using Apple Silicon GPU acceleration!")
    elif device.type == "cuda":
        print(f"✅ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Using CPU (will be slower)")

    # Create model
    print(f"\n📦 Creating model: {num_filters}x{num_residual_blocks}")
    model = LeelaZeroNet(
        num_filters=num_filters,
        num_residual_blocks=num_residual_blocks,
        se_ratio=8,
        policy_loss_weight=1.0,
        value_loss_weight=1.6,
        moves_left_loss_weight=0.5,
        q_ratio=0.0,
        optimizer="adam",
        learning_rate=learning_rate
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Load data
    print(f"\n📊 Loading data from {data_dir}...")
    try:
        train_loader, val_loader = create_train_val_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            train_split=0.9,
            shuffle_buffer_size=50000,  # Smaller for local
            num_workers=0,  # 0 for MPS compatibility
            streaming=True
        )
        print("✅ Data loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

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

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\n🎯 Starting training for {num_epochs} epochs...")
    best_val_loss = float('inf')

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

        pbar = tqdm(train_loader, desc="Training", total=min(max_batches_per_epoch, 1000))
        for inputs, policies, values, moves_left in pbar:
            # Move to device
            inputs = inputs.to(device)
            policies = policies.to(device)
            values = values.to(device)
            moves_left = moves_left.to(device)

            optimizer.zero_grad()

            # Forward pass
            policy_out, value_out, moves_left_out = model(inputs)

            # Compute losses
            p_loss = policy_loss(policies, policy_out)
            v_loss = value_loss(values, value_out)
            ml_loss = moves_left_loss(moves_left.unsqueeze(1), moves_left_out)

            loss = 1.0 * p_loss + 1.6 * v_loss + 0.5 * ml_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            train_policy_loss += p_loss.item()
            train_value_loss += v_loss.item()
            train_moves_left_loss += ml_loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'p_loss': f'{p_loss.item():.4f}',
                'v_loss': f'{v_loss.item():.4f}'
            })

            # Limit batches for faster iteration
            if num_batches >= max_batches_per_epoch:
                break

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
            pbar = tqdm(val_loader, desc="Validation", total=min(100, 200))
            for inputs, policies, values, moves_left in pbar:
                inputs = inputs.to(device)
                policies = policies.to(device)
                values = values.to(device)
                moves_left = moves_left.to(device)

                policy_out, value_out, moves_left_out = model(inputs)

                p_loss = policy_loss(policies, policy_out)
                v_loss = value_loss(values, value_out)
                ml_loss = moves_left_loss(moves_left.unsqueeze(1), moves_left_out)

                loss = 1.0 * p_loss + 1.6 * v_loss + 0.5 * ml_loss

                val_loss += loss.item()
                val_policy_loss += p_loss.item()
                val_value_loss += v_loss.item()
                val_moves_left_loss += ml_loss.item()
                val_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                if val_batches >= 100:  # Limit validation
                    break

        val_loss /= val_batches
        val_policy_loss /= val_batches
        val_value_loss /= val_batches
        val_moves_left_loss /= val_batches

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Policy: {train_policy_loss:.4f}, "
              f"Value: {train_value_loss:.4f}, ML: {train_moves_left_loss:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Policy: {val_policy_loss:.4f}, "
              f"Value: {val_value_loss:.4f}, ML: {val_moves_left_loss:.4f}")

        # Update learning rate
        scheduler.step()
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✅ New best model! (val_loss: {val_loss:.4f})")

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "config": {
                    "num_filters": num_filters,
                    "num_residual_blocks": num_residual_blocks,
                    "se_ratio": 8,
                },
            }

            model_path = save_path / f"lc0_{num_filters}x{num_residual_blocks}_best.pt"
            torch.save(checkpoint, model_path)
            print(f"  Saved to {model_path}")

        # Save latest checkpoint
        latest_path = save_path / f"lc0_{num_filters}x{num_residual_blocks}_latest.pt"
        torch.save(checkpoint, latest_path)

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LC0 model locally")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory with .npz files")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (128 for M1)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--filters", type=int, default=64, help="Number of filters (64 for local)")
    parser.add_argument("--blocks", type=int, default=4, help="Number of residual blocks (4 for local)")
    parser.add_argument("--save-dir", type=str, default="training/models", help="Save directory")
    parser.add_argument("--max-batches", type=int, default=500, help="Max batches per epoch")

    args = parser.parse_args()

    train_local(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_filters=args.filters,
        num_residual_blocks=args.blocks,
        save_dir=args.save_dir,
        max_batches_per_epoch=args.max_batches,
    )
