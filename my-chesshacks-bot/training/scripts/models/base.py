"""
Base class for all chess models.

All models must implement this interface to be compatible with the training pipeline.
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class ChessModelBase(nn.Module, ABC):
    """
    Base class for all chess neural networks.

    All models must:
    1. Inherit from this class
    2. Implement the forward() method with the specified signature
    3. Take (batch, 12, 8, 8) board tensors as input
    4. Return (policy_logits, value) as output
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Board state tensor of shape (batch, 12, 8, 8)
                - 12 channels: 6 piece types × 2 colors
                - 8×8: chessboard dimensions

        Returns:
            Tuple of (policy_logits, value):
            - policy_logits: (batch, 4096) - logits for each possible move
                Move encoding: from_square * 64 + to_square
            - value: (batch, 1) - position evaluation in range [-1, 1]
                -1 = losing, 0 = drawn, +1 = winning
        """
        pass

    def get_model_info(self) -> dict:
        """Return model metadata for logging/debugging."""
        num_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "name": self.__class__.__name__,
            "total_params": num_params,
            "trainable_params": trainable_params,
            "architecture": self.get_architecture_name(),
        }

    @abstractmethod
    def get_architecture_name(self) -> str:
        """Return a human-readable architecture name."""
        pass
