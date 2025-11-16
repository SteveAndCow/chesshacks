"""
Data loader for LC0-format 112-channel training data - FIXED VERSION

CRITICAL FIXES:
1. Proper epoch shuffling - seed changes between epochs to prevent memorization
2. Maintains backward compatibility with existing code

Loads preprocessed .npz files containing:
- inputs: (N, 112, 8, 8) board representations
- policies: (N, 1858) policy targets
- values: (N, 3) WDL targets
- moves_left: (N,) moves left targets

Adapted from minimal_lczero's new_data_pipeline.py but simplified
for our .npz format instead of LC0 v6 binary chunks.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pathlib import Path
from typing import Tuple, Iterator
import random
from collections import deque


class LC0Dataset(IterableDataset):
    """
    Iterable dataset that streams data from .npz files with shuffle buffer.

    This avoids loading all data into RAM at once while still providing
    good shuffling for training.

    FIXED: Now properly shuffles data differently each epoch to prevent
    the model from memorizing the data sequence.
    """

    def __init__(
        self,
        data_dir: str,
        shuffle_buffer_size: int = 100000,
        chunk_size: int = 10000,
        max_files: int = None,
        seed: int = 42
    ):
        """
        Args:
            data_dir: Directory containing .npz files
            shuffle_buffer_size: Size of shuffle buffer (larger = better shuffling)
            chunk_size: Number of positions to load from each file at once
            max_files: Maximum number of .npz files to use (None = all)
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.shuffle_buffer_size = shuffle_buffer_size
        self.chunk_size = chunk_size
        self.seed = seed

        # Find all .npz files
        self.npz_files = sorted(self.data_dir.glob("*.npz"))
        if max_files:
            self.npz_files = self.npz_files[:max_files]

        if not self.npz_files:
            raise ValueError(f"No .npz files found in {data_dir}")

        print(f"Found {len(self.npz_files)} .npz files in {data_dir}")

        # CRITICAL FIX: Track epoch count to ensure different shuffle each epoch
        self._epoch_counter = 0

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Iterate over shuffled positions from all .npz files.

        Yields:
            (inputs, policies, values, moves_left) tuples
        """
        # CRITICAL FIX: Increment epoch counter to change shuffle order each epoch
        # This ensures the model sees data in different order every epoch
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        # Create unique seed for this epoch + worker combination
        # epoch_counter changes each time __iter__ is called (once per epoch)
        epoch_seed = self.seed + worker_id + (self._epoch_counter * 12345)
        self._epoch_counter += 1

        rng = random.Random(epoch_seed)

        # Shuffle file order (different each epoch now!)
        files = self.npz_files.copy()
        rng.shuffle(files)

        # Shuffle buffer
        buffer = deque(maxlen=self.shuffle_buffer_size)

        # Stream through files
        for npz_file in files:
            # Load file
            data = np.load(npz_file)
            inputs = data['inputs']
            policies = data['policies']
            values = data['values']
            moves_left = data['moves_left']

            num_positions = len(inputs)

            # Process in chunks to avoid loading too much at once
            for start_idx in range(0, num_positions, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, num_positions)

                # Get chunk
                chunk_inputs = inputs[start_idx:end_idx]
                chunk_policies = policies[start_idx:end_idx]
                chunk_values = values[start_idx:end_idx]
                chunk_moves_left = moves_left[start_idx:end_idx]

                # Add to buffer
                for i in range(len(chunk_inputs)):
                    buffer.append((
                        chunk_inputs[i],
                        chunk_policies[i],
                        chunk_values[i],
                        chunk_moves_left[i]
                    ))

                    # Once buffer is full, start yielding
                    if len(buffer) >= self.shuffle_buffer_size:
                        # Sample random position from buffer
                        idx = rng.randint(0, len(buffer) - 1)
                        sample = buffer[idx]

                        # Remove and yield
                        if idx == len(buffer) - 1:
                            buffer.pop()
                        else:
                            buffer[idx] = buffer.pop()

                        yield (
                            torch.from_numpy(sample[0]),
                            torch.from_numpy(sample[1]),
                            torch.from_numpy(sample[2]),
                            torch.tensor(sample[3], dtype=torch.float32)
                        )

        # Drain remaining buffer
        while buffer:
            sample = buffer.pop()
            yield (
                torch.from_numpy(sample[0]),
                torch.from_numpy(sample[1]),
                torch.from_numpy(sample[2]),
                torch.tensor(sample[3], dtype=torch.float32)
            )


class LC0DatasetSimple(Dataset):
    """
    Simple dataset that loads all .npz files into memory.

    Good for small datasets or if you have enough RAM.
    Faster than streaming but requires more memory.
    """

    def __init__(self, data_dir: str, max_positions: int = None):
        """
        Args:
            data_dir: Directory containing .npz files
            max_positions: Maximum number of positions to load (None = all)
        """
        self.data_dir = Path(data_dir)

        # Load all .npz files
        npz_files = sorted(self.data_dir.glob("*.npz"))
        if not npz_files:
            raise ValueError(f"No .npz files found in {data_dir}")

        print(f"Loading {len(npz_files)} .npz files into memory...")

        inputs_list = []
        policies_list = []
        values_list = []
        moves_left_list = []

        total_positions = 0

        for npz_file in npz_files:
            data = np.load(npz_file)

            inputs_list.append(data['inputs'])
            policies_list.append(data['policies'])
            values_list.append(data['values'])
            moves_left_list.append(data['moves_left'])

            total_positions += len(data['inputs'])

            if max_positions and total_positions >= max_positions:
                break

        # Concatenate all
        self.inputs = np.concatenate(inputs_list, axis=0)
        self.policies = np.concatenate(policies_list, axis=0)
        self.values = np.concatenate(values_list, axis=0)
        self.moves_left = np.concatenate(moves_left_list, axis=0)

        if max_positions:
            self.inputs = self.inputs[:max_positions]
            self.policies = self.policies[:max_positions]
            self.values = self.values[:max_positions]
            self.moves_left = self.moves_left[:max_positions]

        print(f"Loaded {len(self.inputs):,} positions")
        print(f"Memory usage: {self.inputs.nbytes / 1e9:.2f} GB")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.inputs[idx]),
            torch.from_numpy(self.policies[idx]),
            torch.from_numpy(self.values[idx]),
            torch.tensor(self.moves_left[idx], dtype=torch.float32)
        )


def create_lc0_dataloader(
    data_dir: str,
    batch_size: int = 256,
    shuffle_buffer_size: int = 100000,
    num_workers: int = 4,
    streaming: bool = True,
    max_positions: int = None,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader for LC0-format data.

    Args:
        data_dir: Directory containing .npz files
        batch_size: Batch size for training
        shuffle_buffer_size: Size of shuffle buffer (only for streaming)
        num_workers: Number of data loading workers
        streaming: If True, use streaming dataset (less memory). If False, load all into RAM.
        max_positions: Maximum positions to load (None = all)
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader ready for training
    """
    if streaming:
        dataset = LC0Dataset(
            data_dir=data_dir,
            shuffle_buffer_size=shuffle_buffer_size,
            max_files=None
        )
        # For IterableDataset, we don't shuffle in DataLoader (already shuffled in dataset)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False  # Shuffling handled in dataset
        )
    else:
        dataset = LC0DatasetSimple(
            data_dir=data_dir,
            max_positions=max_positions
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True
        )

    return loader


def create_train_val_loaders(
    data_dir: str,
    batch_size: int = 256,
    train_split: float = 0.9,
    shuffle_buffer_size: int = 100000,
    num_workers: int = 4,
    streaming: bool = True,
    max_positions: int = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Directory containing .npz files
        batch_size: Batch size for training
        train_split: Fraction of data to use for training (rest for validation)
        shuffle_buffer_size: Size of shuffle buffer
        num_workers: Number of data loading workers
        streaming: Use streaming or load all into memory
        max_positions: Maximum positions to load

    Returns:
        (train_loader, val_loader)
    """
    data_path = Path(data_dir)
    npz_files = sorted(data_path.glob("*.npz"))

    if not npz_files:
        raise ValueError(f"No .npz files found in {data_dir}")

    # IMPROVEMENT: Shuffle files before split to avoid distribution mismatch
    # (in case files are sorted by date/source)
    rng = random.Random(42)
    npz_files_shuffled = npz_files.copy()
    rng.shuffle(npz_files_shuffled)

    # Split files into train/val
    num_train_files = int(len(npz_files_shuffled) * train_split)

    # Ensure at least 1 file for validation
    if num_train_files >= len(npz_files_shuffled):
        num_train_files = max(1, len(npz_files_shuffled) - 1)

    train_files = npz_files_shuffled[:num_train_files]
    val_files = npz_files_shuffled[num_train_files:]

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    # Create separate directories (or filter in dataset)
    # For now, we'll use streaming approach with file filtering
    if streaming:
        # Create datasets with filtered files
        train_dataset = LC0Dataset(
            data_dir=data_dir,
            shuffle_buffer_size=shuffle_buffer_size,
        )
        train_dataset.npz_files = train_files

        val_dataset = LC0Dataset(
            data_dir=data_dir,
            shuffle_buffer_size=shuffle_buffer_size // 5,  # Smaller buffer for val
        )
        val_dataset.npz_files = val_files

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=max(1, num_workers // 2),
            pin_memory=True,
            shuffle=False
        )
    else:
        # Simple in-memory approach - need to implement file filtering
        raise NotImplementedError("Non-streaming train/val split not yet implemented")

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the data loader
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_loader_lc0_v2.py <data_dir>")
        sys.exit(1)

    data_dir = sys.argv[1]

    print("Testing LC0Dataset (streaming)...")
    loader = create_lc0_dataloader(
        data_dir=data_dir,
        batch_size=32,
        streaming=True,
        num_workers=2
    )

    print("\nLoading first batch...")
    for inputs, policies, values, moves_left in loader:
        print(f"Inputs shape: {inputs.shape}")
        print(f"Policies shape: {policies.shape}")
        print(f"Values shape: {values.shape}")
        print(f"Moves left shape: {moves_left.shape}")
        print(f"\nFirst position summary:")
        print(f"  - Channels: {inputs[0].shape[0]} (should be 112)")
        print(f"  - Board size: {inputs[0].shape[1]}x{inputs[0].shape[2]} (should be 8x8)")
        print(f"  - Policy target: {torch.argmax(policies[0]).item()} (move index)")
        print(f"  - Value target: {values[0]} (WDL)")
        print(f"  - Moves left: {moves_left[0].item()}")
        break

    print("\n✅ Data loader test passed!")
