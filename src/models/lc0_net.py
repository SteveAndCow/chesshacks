"""
Lightweight LC0 network for inference only (no PyTorch Lightning dependency).

This is a stripped-down version of lccnn.py that only includes the forward pass.
"""
import torch
from torch import nn
from torch.nn import functional as F
from typing import NamedTuple
from collections import OrderedDict
from math import sqrt


class ModelOutput(NamedTuple):
    policy: torch.Tensor
    value: torch.Tensor
    moves_left: torch.Tensor


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, se_ratio):
        super().__init__()
        se_channels = channels // se_ratio
        self.fc1 = nn.Linear(channels, se_channels)
        self.fc2 = nn.Linear(se_channels, 2 * channels)

    def forward(self, x):
        # x: (batch, channels, 8, 8)
        pooled = x.mean(dim=[2, 3])  # Global average pooling
        fc1_out = F.relu(self.fc1(pooled))
        fc2_out = self.fc2(fc1_out)

        # Split into weight and bias
        w, b = fc2_out[:, :x.shape[1]], fc2_out[:, x.shape[1]:]
        w = w.unsqueeze(-1).unsqueeze(-1)
        b = b.unsqueeze(-1).unsqueeze(-1)

        return torch.sigmoid(w) * x + b


class ConvBlock(nn.Module):
    def __init__(self, input_channels, filter_size, output_channels):
        super().__init__()
        padding = filter_size // 2
        self.conv = nn.Conv2d(
            input_channels, output_channels,
            kernel_size=filter_size, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, num_filters, se_ratio):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.se = SqueezeExcitation(num_filters, se_ratio) if se_ratio > 0 else None

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        out += residual
        return F.relu(out)


class ConvolutionalPolicyHead(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv = nn.Conv2d(num_filters, 80, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(80)
        # LC0 policy: 1858 possible moves
        self.fc = nn.Linear(80 * 8 * 8, 1858)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.reshape(x.shape[0], -1)  # Flatten
        return self.fc(x)


class ConvolutionalValueOrMovesLeftHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_filters, hidden_dim, relu):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, num_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.fc1 = nn.Linear(num_filters * 8 * 8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.use_relu = relu

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.use_relu:
            x = F.relu(x)
        return x


class LeelaZeroNet(nn.Module):
    """
    Leela Chess Zero network architecture (inference only).

    Input: (batch, 112, 8, 8) - 112 channels of board state
    Output:
        - policy: (batch, 1858) - move probabilities
        - value: (batch, 3) - win/draw/loss probabilities
        - moves_left: (batch, 1) - estimated moves until game end
    """

    def __init__(
        self,
        num_filters,
        num_residual_blocks,
        se_ratio,
        **kwargs  # Ignore training-specific kwargs
    ):
        super().__init__()
        self.input_block = ConvBlock(
            input_channels=112, filter_size=3, output_channels=num_filters
        )
        residual_blocks = OrderedDict(
            [
                (f"residual_block_{i}", ResidualBlock(num_filters, se_ratio))
                for i in range(num_residual_blocks)
            ]
        )
        self.residual_blocks = nn.Sequential(residual_blocks)
        self.policy_head = ConvolutionalPolicyHead(num_filters=num_filters)
        # Value head: 3 dimensions for WDL (win/draw/loss)
        self.value_head = ConvolutionalValueOrMovesLeftHead(
            input_dim=num_filters,
            output_dim=3,
            num_filters=32,
            hidden_dim=128,
            relu=False,
        )
        # Moves left head
        self.moves_left_head = ConvolutionalValueOrMovesLeftHead(
            input_dim=num_filters,
            output_dim=1,
            num_filters=8,
            hidden_dim=128,
            relu=True,
        )

    def forward(self, input_planes: torch.Tensor) -> ModelOutput:
        flow = input_planes.reshape(-1, 112, 8, 8)
        flow = self.input_block(flow)
        flow = self.residual_blocks(flow)
        policy_out = self.policy_head(flow)
        value_out = self.value_head(flow)
        moves_left_out = self.moves_left_head(flow)
        return ModelOutput(policy_out, value_out, moves_left_out)
