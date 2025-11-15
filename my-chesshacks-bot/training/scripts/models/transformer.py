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
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.bias = nn.Parameter(torch.zeros(num_heads, 15, 15))

    def forward(self, seq_len=64):
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

class LearnedPositionalBias(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, input):
        return input + self.bias


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        #self.relative_bias = RelativePositionBias(nhead)

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
        bias = self.relative_bias(seq_len=src.size(1))  # (nhead, 64, 64)

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
        self.input_proj = nn.Linear(12, d_model)

        # Transformer encoder layers

        self.positional_encoding = LearnedPositionalBias(64, d_model)

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
        batch_size = x.size(0)

        # Reshape board to sequence of squares: (batch, 64, 12)
        x = x.view(batch_size, 12, 64).permute(0, 2, 1)  # (batch, 64, 12)

        # Project to d_model dimensions
        x = self.input_proj(x)  # (batch, 64, d_model)
        x = self.positional_encoding(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        policy_features = self.policy_proj(x)  # (batch, 64, d_model)

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
