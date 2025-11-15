"""
CNN-based chess models (baseline approach).

These models use convolutional neural networks to process the chess board.
Good baseline with proven effectiveness.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ChessModelBase


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ChessCNN(ChessModelBase):
    """
    CNN with residual blocks for chess (AlphaZero-style).

    Architecture:
    - Input conv layer
    - N residual blocks
    - Dual heads (policy + value)
    """

    def __init__(self, num_residual_blocks=5, num_channels=128):
        super().__init__()

        self.num_residual_blocks = num_residual_blocks
        self.num_channels = num_channels

        # Initial convolution (16 channels: 12 pieces + 4 game state)
        self.conv_input = nn.Conv2d(16, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_residual_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Result classification head (win/draw/loss)
        self.result_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.result_bn = nn.BatchNorm2d(32)
        self.result_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.result_fc2 = nn.Linear(256, 3)  # 3 classes

    def forward(self, x):
        # Input: (batch, 12, 8, 8)
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 8 * 8)
        policy = self.policy_fc(policy)  # (batch, 4096)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # (batch, 1) in [-1, 1]

        # Result classification head
        result = F.relu(self.result_bn(self.result_conv(x)))
        result = result.view(-1, 32 * 8 * 8)
        result = F.relu(self.result_fc1(result))
        result = self.result_fc2(result)  # (batch, 3) logits

        return policy, value, result

    def get_architecture_name(self) -> str:
        return f"CNN-ResNet-{self.num_residual_blocks}x{self.num_channels}"


class ChessCNNLite(ChessModelBase):
    """
    Lightweight CNN for faster training/inference.

    Good for quick iteration during hackathon.
    ~5-10M parameters vs ~50M for full CNN.
    """

    def __init__(self, num_channels=128):
        super().__init__()

        self.num_channels = num_channels

        # Simple CNN (no residual blocks, 16 input channels)
        self.conv1 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Result classification head (win/draw/loss)
        self.result_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.result_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.result_fc2 = nn.Linear(256, 3)  # 3 classes

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Policy head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(-1, 32 * 8 * 8)
        policy = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(-1, 32 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        # Result classification head
        result = F.relu(self.result_conv(x))
        result = result.view(-1, 32 * 8 * 8)
        result = F.relu(self.result_fc1(result))
        result = self.result_fc2(result)  # (batch, 3) logits

        return policy, value, result

    def get_architecture_name(self) -> str:
        return f"CNN-Lite-{self.num_channels}"
