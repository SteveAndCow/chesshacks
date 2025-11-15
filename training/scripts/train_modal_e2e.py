"""
End-to-end Modal training: Preprocessing + Training in one pipeline.

Efficiently processes PGN and immediately trains, keeping data in memory/disk.
"""
import modal
from pathlib import Path

app = modal.App("chesshacks-e2e-training")

# Unified image with all dependencies
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
    .add_local_dir(
        local_path="training/scripts/models",
        remote_path="/root/models"
    )
    .add_local_file(
        local_path="training/scripts/model_factory.py",
        remote_path="/root/model_factory.py"
    )
    .add_local_file(
        local_path="training/scripts/data_loader.py",
        remote_path="/root/data_loader.py"
    )
    .add_local_file(
        local_path="training/scripts/preprocess.py",
        remote_path="/root/preprocess.py"
    )
)

# Modal Volume
data_volume = modal.Volume.from_name("chess-training-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    memory=65536,  # 64GB RAM for preprocessing + training
    timeout=3600 * 4,  # 4 hours for full pipeline
    volumes={"/data": data_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def preprocess_and_train(config_dict: dict, preprocess: bool = True):
    """
    End-to-end pipeline: Preprocess PGN → Train model → Upload to HF.

    Args:
        config_dict: Training configuration
        preprocess: If True, preprocess PGN first; if False, use existing data
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    import sys
    import os
    import chess.pgn
    import numpy as np
    from huggingface_hub import HfApi

    sys.path.insert(0, "/root")
    from model_factory import create_model_from_config
    from preprocess import board_to_tensor, move_to_index

    print("="*80)
    print("END-TO-END PIPELINE: PREPROCESSING + TRAINING")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ========== STEP 1: PREPROCESSING ==========
    if preprocess:
        print("\n" + "="*80)
        print("STEP 1: PREPROCESSING")
        print("="*80)

        pgn_path = "/data/raw/filtered_games.pgn"
        print(f"Input: {pgn_path}")

        if not Path(pgn_path).exists():
            return {"error": "PGN file not found. Upload first."}

        # Streaming preprocessing
        print("\n📊 Processing positions in streaming mode...")

        boards_list = []
        moves_list = []
        values_list = []

        batch_size = 10000
        total_positions = 0
        games_processed = 0

        with open(pgn_path) as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                games_processed += 1
                if games_processed % 1000 == 0:
                    print(f"Games: {games_processed:,} | Positions: {total_positions:,}")

                # Get result
                result = game.headers.get("Result", "*")
                if result == "1-0":
                    value = 1.0
                elif result == "0-1":
                    value = -1.0
                elif result == "1/2-1/2":
                    value = 0.0
                else:
                    continue

                # Process positions
                board = game.board()
                for move in game.mainline_moves():
                    board_tensor = board_to_tensor(board)
                    move_idx = move_to_index(move)

                    boards_list.append(board_tensor)
                    moves_list.append(move_idx)
                    values_list.append(value)

                    total_positions += 1

                    # Free memory periodically
                    if len(boards_list) >= 100000:  # 100k positions
                        print(f"  Converting batch to numpy (total: {total_positions:,})...")
                        # Keep as lists for now

                    board.push(move)

        print(f"\n✅ Preprocessing complete!")
        print(f"Total games: {games_processed:,}")
        print(f"Total positions: {total_positions:,}")

        # Convert to numpy arrays
        print("\n📦 Converting to numpy arrays...")
        boards_array = np.array(boards_list, dtype=np.float32)
        moves_array = np.array(moves_list, dtype=np.int64)
        values_array = np.array(values_list, dtype=np.float32)

        print(f"Shapes: boards={boards_array.shape}, moves={moves_array.shape}, values={values_array.shape}")

    else:
        # Load existing preprocessed data
        print("\n📊 Loading existing preprocessed data...")
        data_dir = config_dict.get("data", {}).get("data_dir", "/data/processed")

        boards_array = np.load(f"{data_dir}/boards.npy")
        moves_array = np.load(f"{data_dir}/moves.npy")
        values_array = np.load(f"{data_dir}/values.npy")

        total_positions = len(boards_array)
        print(f"Loaded {total_positions:,} positions")

    # ========== STEP 2: CREATE DATALOADERS ==========
    print("\n" + "="*80)
    print("STEP 2: CREATING DATALOADERS")
    print("="*80)

    # Apply max_samples limit if specified
    max_samples = config_dict.get("data", {}).get("max_samples")
    if max_samples and max_samples < len(boards_array):
        print(f"Limiting to {max_samples:,} samples")
        boards_array = boards_array[:max_samples]
        moves_array = moves_array[:max_samples]
        values_array = values_array[:max_samples]

    # Train/val split
    train_split = config_dict.get("data", {}).get("train_split", 0.9)
    split_idx = int(len(boards_array) * train_split)

    train_boards = torch.from_numpy(boards_array[:split_idx])
    train_moves = torch.from_numpy(moves_array[:split_idx])
    train_values = torch.from_numpy(values_array[:split_idx])

    val_boards = torch.from_numpy(boards_array[split_idx:])
    val_moves = torch.from_numpy(moves_array[split_idx:])
    val_values = torch.from_numpy(values_array[split_idx:])

    train_dataset = TensorDataset(train_boards, train_moves, train_values)
    val_dataset = TensorDataset(val_boards, val_moves, val_values)

    batch_size = config_dict["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Train: {len(train_dataset):,} samples")
    print(f"Val: {len(val_dataset):,} samples")

    # Free memory
    del boards_array, moves_array, values_array

    # ========== STEP 3: TRAINING ==========
    print("\n" + "="*80)
    print("STEP 3: TRAINING")
    print("="*80)

    # Create model
    print("\n📦 Creating model...")
    model = create_model_from_config(config_dict, device=device)

    # Setup training
    training_config = config_dict["training"]
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config.get("weight_decay", 0.0001)
    )

    # Learning rate scheduler
    if training_config.get("lr_scheduler") == "cosine":
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
    result_weight = training_config.get("result_weight", 0.5)

    # Training loop
    best_val_loss = float('inf')
    model_path = None

    for epoch in range(training_config["epochs"]):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{training_config['epochs']}")
        print(f"{'='*80}")

        # Train
        model.train()
        train_loss = 0

        for boards, moves, values in tqdm(train_loader, desc="Training"):
            boards = boards.to(device)
            moves = moves.to(device)
            values = values.to(device).unsqueeze(1)

            optimizer.zero_grad()

            policy_logits, value_pred, result_logits = model(boards)

            # Result targets
            result_targets = torch.zeros(values.size(0), dtype=torch.long, device=device)
            result_targets[values[:, 0] < -0.3] = 0
            result_targets[(values[:, 0] >= -0.3) & (values[:, 0] <= 0.3)] = 1
            result_targets[values[:, 0] > 0.3] = 2

            p_loss = policy_criterion(policy_logits, moves)
            v_loss = value_criterion(value_pred, values)
            r_loss = policy_criterion(result_logits, result_targets)

            loss = policy_weight * p_loss + value_weight * v_loss + result_weight * r_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for boards, moves, values in tqdm(val_loader, desc="Validation"):
                boards = boards.to(device)
                moves = moves.to(device)
                values = values.to(device).unsqueeze(1)

                policy_logits, value_pred, result_logits = model(boards)

                result_targets = torch.zeros(values.size(0), dtype=torch.long, device=device)
                result_targets[values[:, 0] < -0.3] = 0
                result_targets[(values[:, 0] >= -0.3) & (values[:, 0] <= 0.3)] = 1
                result_targets[values[:, 0] > 0.3] = 2

                p_loss = policy_criterion(policy_logits, moves)
                v_loss = value_criterion(value_pred, values)
                r_loss = policy_criterion(result_logits, result_targets)

                loss = policy_weight * p_loss + value_weight * v_loss + result_weight * r_loss

                val_loss += loss.item()

                predicted = torch.argmax(policy_logits, dim=1)
                correct += (predicted == moves).sum().item()
                total += moves.size(0)

        val_loss /= len(val_loader)
        accuracy = correct / total

        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {accuracy:.4f}")

        if scheduler:
            scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"✅ New best model! (val_loss: {val_loss:.4f})")

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

            model_path = f"/tmp/best_model.pt"
            torch.save(checkpoint, model_path)

    # ========== STEP 4: UPLOAD TO HUGGINGFACE ==========
    print("\n" + "="*80)
    print("STEP 4: UPLOADING TO HUGGINGFACE")
    print("="*80)

    if model_path is None:
        return {"error": "No model saved"}

    hf_config = config_dict.get("huggingface", {})
    repo_id = hf_config.get("repo_id")
    model_name = hf_config.get("model_name")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        return {"error": "No HF_TOKEN"}

    try:
        print(f"Uploading to: {repo_id}/{model_name}.pt")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=f"{model_name}.pt",
            repo_id=repo_id,
            token=hf_token,
        )
        print(f"✅ Model uploaded!")

        return {
            "status": "success",
            "model_type": config_dict["model"]["type"],
            "best_val_loss": best_val_loss,
            "final_accuracy": accuracy,
            "epochs_trained": training_config["epochs"],
            "hf_repo": repo_id,
            "model_name": model_name,
            "total_positions": total_positions if preprocess else len(train_dataset) + len(val_dataset),
        }
    except Exception as e:
        return {"error": str(e)}


@app.local_entrypoint()
def main(config: str = "configs/transformer_tiny.yaml", preprocess: bool = True):
    """
    Run end-to-end pipeline.

    Args:
        config: Path to training config
        preprocess: Whether to preprocess PGN first (True) or use existing data (False)

    Usage:
        # With preprocessing (full pipeline)
        modal run scripts/train_modal_e2e.py --config configs/transformer_tiny.yaml --preprocess True

        # Without preprocessing (use existing data)
        modal run scripts/train_modal_e2e.py --config configs/transformer_tiny.yaml --preprocess False
    """
    import yaml

    config_path = Path(config)
    if not config_path.exists():
        print(f"❌ Config not found: {config}")
        return

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    print(f"🚀 Launching end-to-end pipeline...")
    print(f"Config: {config}")
    print(f"Model: {config_dict['model']['type']}")
    print(f"Preprocessing: {preprocess}")

    result = preprocess_and_train.remote(config_dict, preprocess=preprocess)

    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE")
    print("="*80)
    print(f"Results: {result}")
