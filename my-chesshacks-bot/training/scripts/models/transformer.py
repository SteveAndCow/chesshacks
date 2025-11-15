"""
Transformer-based chess model (ChessFormer-inspired).

Key innovation: Relative position encoding for chess board geometry.
Transformers excel at capturing long-range patterns in chess positions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import ChessModelBase


class RelativePositionBias(nn.Module):
    """
    Learnable relative position bias for chess board.

    Instead of absolute positions, learns relationships between squares.
    Key insight from ChessFormer: board topology matters more than Euclidean distance.
    """

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

        # Bias for each (dx, dy) offset between squares
        # dx, dy in range [-7, 7] → 15 x 15 grid
        self.bias = nn.Parameter(torch.zeros(num_heads, 15, 15))

    def forward(self, seq_len=64):
        """
        Generate relative position bias matrix for all square pairs.

        Returns: (num_heads, seq_len, seq_len) bias matrix
        """
        # Create relative position indices
        positions = torch.arange(seq_len, device=self.bias.device)
        rows = positions // 8
        cols = positions % 8

        # Compute relative distances
        row_diff = rows.unsqueeze(1) - rows.unsqueeze(0)  # (64, 64)
        col_diff = cols.unsqueeze(1) - cols.unsqueeze(0)  # (64, 64)

        # Map to bias indices (add 7 to handle negative offsets)
        row_idx = (row_diff + 7).clamp(0, 14)
        col_idx = (col_diff + 7).clamp(0, 14)

        # Gather biases
        bias_matrix = self.bias[:, row_idx, col_idx]  # (num_heads, 64, 64)

        return bias_matrix


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with relative position bias."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.relative_bias = RelativePositionBias(nhead)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        """
        Args:
            src: (batch, 64, d_model) - sequence of board squares

        Returns:
            (batch, 64, d_model)
        """
        # Self-attention with relative position bias
        bias = self.relative_bias(seq_len=src.size(1))  # (nhead, 64, 64)

        # Add bias to attention scores (broadcast over batch dimension)
        src2, _ = self.self_attn(
            src, src, src,
            attn_mask=bias.repeat(src.size(0), 1, 1)  # (batch*nhead, 64, 64)
        )

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class ChessTransformer(ChessModelBase):
    """
    Transformer-based chess model inspired by ChessFormer.

    Key features:
    - Relative position encoding (learns board geometry)
    - Self-attention over all 64 squares
    - Dual heads for policy + value

    Smaller than full ChessFormer (240M params) for hackathon speed.
    """

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Input embedding: Convert (12, 8, 8) board to 64 tokens
        # Each square becomes a d_model-dimensional token
        self.input_proj = nn.Linear(12, d_model)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Policy head
        # Predicts move from each square to any other square
        self.policy_proj = nn.Linear(d_model, d_model)
        self.policy_out = nn.Linear(d_model * 2, 1)  # Concatenate source + dest

        # Value head
        self.value_fc1 = nn.Linear(d_model * 64, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, 12, 8, 8) - board state

        Returns:
            policy: (batch, 4096) - move logits
            value: (batch, 1) - position evaluation
        """
        batch_size = x.size(0)

        # Reshape board to sequence of squares: (batch, 64, 12)
        x = x.view(batch_size, 12, 64).permute(0, 2, 1)  # (batch, 64, 12)

        # Project to d_model dimensions
        x = self.input_proj(x)  # (batch, 64, d_model)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Policy head: Predict move from each from_square to each to_square
        # Simplified: Use linear projection instead of full from-to attention
        policy_features = self.policy_proj(x)  # (batch, 64, d_model)

        # Create all from-to pairs (simplified approach)
        # For each from_square, concatenate with all to_squares
        from_squares = policy_features.unsqueeze(2).expand(-1, -1, 64, -1)  # (batch, 64, 64, d_model)
        to_squares = policy_features.unsqueeze(1).expand(-1, 64, -1, -1)  # (batch, 64, 64, d_model)

        # Concatenate and predict move probability
        move_pairs = torch.cat([from_squares, to_squares], dim=-1)  # (batch, 64, 64, 2*d_model)
        policy = self.policy_out(move_pairs).squeeze(-1)  # (batch, 64, 64)
        policy = policy.view(batch_size, 4096)  # (batch, 4096)

        # Value head: Global position evaluation
        value_input = x.view(batch_size, -1)  # (batch, 64*d_model)
        value = F.relu(self.value_fc1(value_input))
        value = torch.tanh(self.value_fc2(value))  # (batch, 1)

        return policy, value

    def get_architecture_name(self) -> str:
        return f"Transformer-{self.num_layers}L-{self.nhead}H-{self.d_model}D"


class ChessTransformerLite(ChessModelBase):
    """
    Lightweight transformer for faster training.

    Smaller model for quick iteration:
    - Fewer layers (2-3 instead of 4-6)
    - Smaller hidden dim (128 instead of 256)
    - Fewer attention heads (4 instead of 8)
    """

    def __init__(
        self,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Input embedding
        self.input_proj = nn.Linear(12, d_model)

        # Transformer encoder (using PyTorch built-in for speed)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Policy head (simplified)
        self.policy_fc = nn.Linear(d_model * 64, 4096)

        # Value head
        self.value_fc1 = nn.Linear(d_model * 64, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        batch_size = x.size(0)

        # Reshape to sequence
        x = x.view(batch_size, 12, 64).permute(0, 2, 1)  # (batch, 64, 12)

        # Project and apply transformer
        x = self.input_proj(x)
        x = self.transformer(x)  # (batch, 64, d_model)

        # Flatten for heads
        x_flat = x.view(batch_size, -1)  # (batch, 64*d_model)

        # Policy head
        policy = self.policy_fc(x_flat)  # (batch, 4096)

        # Value head
        value = F.relu(self.value_fc1(x_flat))
        value = torch.tanh(self.value_fc2(value))  # (batch, 1)

        return policy, value

    def get_architecture_name(self) -> str:
        return f"Transformer-Lite-{self.num_layers}L-{self.nhead}H-{self.d_model}D"
