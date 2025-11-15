"""
Test training loop with real data.

Verifies:
- Data loading works
- Forward pass with real data
- Loss computation (policy, value, result)
- Backward pass and optimizer step
- No NaN or inf in losses
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from pathlib import Path

# Add models to path
sys.path.insert(0, str(Path(__file__).parent))

from models.transformer import ChessTransformerLite

def main():
    print("\n" + "="*60)
    print("TRAINING LOOP TEST")
    print("="*60)

    # Load small subset of data
    data_dir = Path("training/data/processed_16ch")

    print(f"\n📂 Loading data from {data_dir}...")
    boards = np.load(data_dir / "boards.npy")[:1000]
    moves = np.load(data_dir / "moves.npy")[:1000]
    values = np.load(data_dir / "values.npy")[:1000]

    print(f"✅ Loaded {len(boards)} samples")

    # Create dataset and loader
    dataset = TensorDataset(
        torch.from_numpy(boards),
        torch.from_numpy(moves),
        torch.from_numpy(values)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"✅ Created DataLoader with batch_size=32")

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    model = ChessTransformerLite().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"📦 Model: ChessTransformerLite ({num_params:,} parameters)")

    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    # Run 5 training iterations
    print(f"\n🔄 Running 5 training iterations...")
    print("="*60)

    model.train()

    for i, (batch_boards, batch_moves, batch_values) in enumerate(loader):
        if i >= 5:
            break

        batch_boards = batch_boards.to(device)
        batch_moves = batch_moves.to(device)
        batch_values = batch_values.to(device).unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass
        policy_logits, value_pred, result_logits = model(batch_boards)

        # Convert continuous values to result classes (loss=0, draw=1, win=2)
        result_targets = torch.zeros(batch_values.size(0), dtype=torch.long, device=device)
        result_targets[batch_values[:, 0] < -0.3] = 0  # loss
        result_targets[(batch_values[:, 0] >= -0.3) & (batch_values[:, 0] <= 0.3)] = 1  # draw
        result_targets[batch_values[:, 0] > 0.3] = 2  # win

        # Compute losses
        p_loss = policy_criterion(policy_logits, batch_moves)
        v_loss = value_criterion(value_pred, batch_values)
        r_loss = policy_criterion(result_logits, result_targets)

        loss = p_loss + v_loss + 0.5 * r_loss

        # Check for NaN/inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ Iteration {i+1}: Loss is NaN or inf!")
            return 1

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print results
        print(f"Iteration {i+1}:")
        print(f"  Total Loss:  {loss.item():.4f}")
        print(f"  Policy Loss: {p_loss.item():.4f}")
        print(f"  Value Loss:  {v_loss.item():.4f}")
        print(f"  Result Loss: {r_loss.item():.4f}")
        print()

    # Test validation mode
    print("="*60)
    print("🔍 Testing validation mode...")
    print("="*60)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (batch_boards, batch_moves, batch_values) in enumerate(loader):
            if i >= 3:
                break

            batch_boards = batch_boards.to(device)
            batch_moves = batch_moves.to(device)

            policy_logits, value_pred, result_logits = model(batch_boards)

            predicted = torch.argmax(policy_logits, dim=1)
            correct += (predicted == batch_moves).sum().item()
            total += batch_moves.size(0)

    accuracy = correct / total
    print(f"Validation accuracy: {accuracy:.4f} ({correct}/{total})")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✅ Data loading works")
    print("✅ Forward pass works")
    print("✅ Loss computation works (policy + value + result)")
    print("✅ Backward pass works")
    print("✅ No NaN or inf in losses")
    print("✅ Validation mode works")
    print("\n✅ Training loop test passed!")

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
