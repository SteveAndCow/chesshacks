"""
Lightweight LC0 network for inference only (matches training architecture exactly).

This is copied from pt_layers.py and lccnn.py but without PyTorch Lightning.
"""
import torch
from torch import nn
from torch.nn import functional as F
from typing import NamedTuple
from collections import OrderedDict


class ModelOutput(NamedTuple):
    policy: torch.Tensor
    value: torch.Tensor
    moves_left: torch.Tensor


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, se_ratio):
        super().__init__()
        self.se_ratio = se_ratio
        self.pooler = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Sequential(
            nn.Linear(channels, int(channels // se_ratio), bias=False), nn.ReLU()
        )
        self.expand = nn.Linear(int(channels // se_ratio), channels * 2, bias=False)
        self.channels = channels
        nn.init.xavier_normal_(self.squeeze[0].weight)
        nn.init.xavier_normal_(self.expand.weight)

    def forward(self, x):
        pooled = self.pooler(x).view(-1, self.channels)
        squeezed = self.squeeze(pooled)
        expanded = self.expand(squeezed).view(-1, self.channels * 2, 1, 1)
        gammas, betas = torch.split(expanded, self.channels, dim=1)
        gammas = torch.sigmoid(gammas)
        return gammas * x + betas


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, filter_size):
        super().__init__()
        self.conv_layer = nn.Conv2d(
            input_channels, output_channels, filter_size, bias=False, padding="same"
        )
        self.conv_layer.weight.clamp_weights = True
        self.batchnorm = nn.BatchNorm2d(output_channels, affine=True)
        nn.init.xavier_normal_(self.conv_layer.weight)

    def forward(self, inputs):
        out = self.conv_layer(inputs)
        out = self.batchnorm(out.float())
        return F.relu(out)


class ResidualBlock(nn.Module):
    def __init__(self, channels, se_ratio):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            bias=False,
            padding="same",
        )
        self.conv1.weight.clamp_weights = True
        self.batch_norm = nn.BatchNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            bias=False,
            padding="same",
        )
        self.conv2.weight.clamp_weights = True
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        self.squeeze_excite = SqueezeExcitation(channels, se_ratio)

    def forward(self, inputs):
        out1 = self.conv1(inputs)
        out1 = F.relu(self.batch_norm(out1.float()))
        out2 = self.conv2(out1)
        out2 = self.squeeze_excite(out2)
        return F.relu(inputs + out2)


class ConvolutionalPolicyHead(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv_block = ConvBlock(
            filter_size=3, input_channels=num_filters, output_channels=num_filters
        )
        # No l2_reg on the final convolution
        self.conv = nn.Conv2d(num_filters, 80, 3, bias=True, padding="same")
        nn.init.xavier_normal_(self.conv.weight)

        # Import policy map - make it a non-trainable parameter
        from .lc0_policy_map import make_map
        policy_map_array = make_map()
        self.fc1 = nn.parameter.Parameter(
            torch.tensor(policy_map_array, requires_grad=False, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, inputs):
        flow = self.conv_block(inputs)
        flow = self.conv(flow)
        h_conv_pol_flat = flow.reshape(-1, 80 * 8 * 8)
        return h_conv_pol_flat @ self.fc1.type(h_conv_pol_flat.dtype)


class ConvolutionalValueOrMovesLeftHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_filters, hidden_dim, relu):
        super().__init__()
        self.num_filters = num_filters
        self.conv_block = ConvBlock(
            input_channels=input_dim, filter_size=1, output_channels=num_filters
        )
        # No l2_reg on the final layers
        self.fc2 = nn.Linear(self.num_filters * 8 * 8, hidden_dim, bias=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=True)
        self.relu = relu
        nn.init.xavier_normal_(self.fc_out.weight)

    def forward(self, inputs):
        flow = self.conv_block(inputs)
        flow = flow.reshape(-1, self.num_filters * 8 * 8)
        flow = self.fc2(flow)
        flow = F.relu(flow)
        flow = self.fc_out(flow)
        if self.relu:
            flow = F.relu(flow)
        return flow


class LeelaZeroNet(nn.Module):
    """
    Leela Chess Zero network architecture (inference only, matches training exactly).

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
