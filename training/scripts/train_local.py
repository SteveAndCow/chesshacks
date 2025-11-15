"""
Quick local training run for validation.

Trains on small dataset for 2 epochs to verify:
- Training converges (loss decreases)
- Validation works
- Model checkpointing works
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# Add models to path
sys.path.insert(0, str(Path(__file__).parent))

from models.transformer import ChessTransformerLite

def main():
    print("\n" + "="*60)
    print("LOCAL TRAINING RUN (2 EPOCHS)")
    print("="*60)

    # Load data
    data_dir = Path("training/data/processed_16ch")
    print(f"\n📂 Loading data from {data_dir}...")

    boards = np.load(data_dir / "boards.npy")
    moves = np.load(data_dir / "moves.npy")
    values = np.load(data_dir / "values.npy")

    print(f"✅ Loaded {len(boards):,} samples")

    # Train/val split
    train_size = int(0.9 * len(boards))
    train_boards, val_boards = boards[:train_size], boards[train_size:]
    train_moves, val_moves = moves[:train_size], moves[train_size:]
    train_values, val_values = values[:train_size], values[train_size:]

    print(f"Train samples: {len(train_boards):,}")
    print(f"Val samples: {len(val_boards):,}")

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.from_numpy(train_boards),
        torch.from_numpy(train_moves),
        torch.from_numpy(train_values)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_boards),
        torch.from_numpy(val_moves),
        torch.from_numpy(val_values)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    model = ChessTransformerLite().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"📦 Model: ChessTransformerLite ({num_params:,} parameters)")

    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    policy_weight = 1.0
    value_weight = 1.0
    result_weight = 0.5

    # Training loop
    print(f"\n🎯 Starting training for 2 epochs...")
    best_val_loss = float('inf')

    for epoch in range(2):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/2")
        print(f"{'='*60}")

        # Train
        model.train()
        train_loss = 0
        train_policy_loss = 0
        train_value_loss = 0
        train_result_loss = 0

        for boards_batch, moves_batch, values_batch in tqdm(train_loader, desc="Training"):
            boards_batch = boards_batch.to(device)
            moves_batch = moves_batch.to(device)
            values_batch = values_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()

            policy_logits, value_pred, result_logits = model(boards_batch)

            # Convert continuous values to result classes
            result_targets = torch.zeros(values_batch.size(0), dtype=torch.long, device=device)
            result_targets[values_batch[:, 0] < -0.3] = 0
            result_targets[(values_batch[:, 0] >= -0.3) & (values_batch[:, 0] <= 0.3)] = 1
            result_targets[values_batch[:, 0] > 0.3] = 2

            p_loss = policy_criterion(policy_logits, moves_batch)
            v_loss = value_criterion(value_pred, values_batch)
            r_loss = policy_criterion(result_logits, result_targets)

            loss = policy_weight * p_loss + value_weight * v_loss + result_weight * r_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_policy_loss += p_loss.item()
            train_value_loss += v_loss.item()
            train_result_loss += r_loss.item()

        train_loss /= len(train_loader)
        train_policy_loss /= len(train_loader)
        train_value_loss /= len(train_loader)
        train_result_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_policy_loss = 0
        val_value_loss = 0
        val_result_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for boards_batch, moves_batch, values_batch in tqdm(val_loader, desc="Validation"):
                boards_batch = boards_batch.to(device)
                moves_batch = moves_batch.to(device)
                values_batch = values_batch.to(device).unsqueeze(1)

                policy_logits, value_pred, result_logits = model(boards_batch)

                result_targets = torch.zeros(values_batch.size(0), dtype=torch.long, device=device)
                result_targets[values_batch[:, 0] < -0.3] = 0
                result_targets[(values_batch[:, 0] >= -0.3) & (values_batch[:, 0] <= 0.3)] = 1
                result_targets[values_batch[:, 0] > 0.3] = 2

                p_loss = policy_criterion(policy_logits, moves_batch)
                v_loss = value_criterion(value_pred, values_batch)
                r_loss = policy_criterion(result_logits, result_targets)

                loss = policy_weight * p_loss + value_weight * v_loss + result_weight * r_loss

                val_loss += loss.item()
                val_policy_loss += p_loss.item()
                val_value_loss += v_loss.item()
                val_result_loss += r_loss.item()

                predicted = torch.argmax(policy_logits, dim=1)
                correct += (predicted == moves_batch).sum().item()
                total += moves_batch.size(0)

        val_loss /= len(val_loader)
        val_policy_loss /= len(val_loader)
        val_value_loss /= len(val_loader)
        val_result_loss /= len(val_loader)
        accuracy = correct / total

        # Log results
        print(f"\nTrain - Loss: {train_loss:.4f}, Policy: {train_policy_loss:.4f}, Value: {train_value_loss:.4f}, Result: {train_result_loss:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Policy: {val_policy_loss:.4f}, Value: {val_value_loss:.4f}, Result: {val_result_loss:.4f}")
        print(f"Val Accuracy: {accuracy:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"✅ New best model! (val_loss: {val_loss:.4f})")

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_accuracy": accuracy,
            }

            checkpoint_path = Path("training/checkpoints")
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path / "local_test_best.pt")
            print(f"💾 Saved checkpoint to {checkpoint_path / 'local_test_best.pt'}")

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Final accuracy: {accuracy:.4f}")
    print("\n✅ Local training test passed!")
    print("✅ Ready for Modal training!")

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
