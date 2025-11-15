"""
Data loading utilities for chess training.

Architecture-agnostic: works with any model that accepts (12, 8, 8) board tensors.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class ChessDataset(Dataset):
    """
    Dataset for chess positions.

    Loads preprocessed data from .npy files.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None
    ):
        """
        Args:
            data_dir: Directory containing processed data (boards.npy, moves.npy, values.npy)
            split: "train" or "val" (if split files exist)
            transform: Optional data augmentation transform
        """
        data_dir = Path(data_dir)

        # Load data
        boards_file = data_dir / f"{split}_boards.npy" if split else data_dir / "boards.npy"
        moves_file = data_dir / f"{split}_moves.npy" if split else data_dir / "moves.npy"
        values_file = data_dir / f"{split}_values.npy" if split else data_dir / "values.npy"

        # Fallback to non-split files if split files don't exist
        if not boards_file.exists():
            boards_file = data_dir / "boards.npy"
            moves_file = data_dir / "moves.npy"
            values_file = data_dir / "values.npy"

        print(f"Loading data from {data_dir}...")
        self.boards = torch.from_numpy(np.load(boards_file)).float()
        self.moves = torch.from_numpy(np.load(moves_file)).long()
        self.values = torch.from_numpy(np.load(values_file)).float()

        print(f"Loaded {len(self.boards)} positions")
        print(f"  Boards shape: {self.boards.shape}")
        print(f"  Moves shape: {self.moves.shape}")
        print(f"  Values shape: {self.values.shape}")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.boards)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            board: (12, 8, 8) board state
            move: (,) move index
            value: (,) position value
        """
        board = self.boards[idx]
        move = self.moves[idx]
        value = self.values[idx]

        if self.transform:
            board = self.transform(board)

        return board, move, value


def create_data_loaders(
    data_dir: str,
    batch_size: int = 256,
    train_split: float = 0.9,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.

    Args:
        data_dir: Directory with processed data
        batch_size: Batch size for training
        train_split: Fraction of data for training (rest for validation)
        num_workers: Number of worker processes for data loading
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        (train_loader, val_loader)
    """
    data_dir = Path(data_dir)

    # Check if pre-split data exists
    train_boards_file = data_dir / "train_boards.npy"
    if train_boards_file.exists():
        print("Using pre-split train/val data")
        train_dataset = ChessDataset(data_dir, split="train")
        val_dataset = ChessDataset(data_dir, split="val")
    else:
        print("Splitting data into train/val...")
        # Load full dataset
        full_dataset = ChessDataset(data_dir, split=None)

        # Split into train/val
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducible split
        )

    print(f"Train set: {len(train_dataset)} positions")
    print(f"Val set: {len(val_dataset)} positions")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")

    # This will fail if no data exists, which is expected
    try:
        train_loader, val_loader = create_data_loaders(
            "training/data/processed",
            batch_size=32,
            num_workers=0  # 0 for testing
        )

        # Test iteration
        boards, moves, values = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  Boards: {boards.shape}")
        print(f"  Moves: {moves.shape}")
        print(f"  Values: {values.shape}")

        print("\n✅ Data loader working!")

    except FileNotFoundError as e:
        print(f"\n⚠️  No processed data found: {e}")
        print("Run preprocess.py first to generate training data")
