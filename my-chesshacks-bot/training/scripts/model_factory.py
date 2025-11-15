"""
Model factory for creating chess models from configuration.

Allows switching between different architectures without changing training code.
"""
import torch
import yaml
from pathlib import Path
from typing import Union
from models.base import ChessModelBase
from models.cnn import ChessCNN, ChessCNNLite
from models.transformer import ChessTransformer, ChessTransformerLite


def create_model_from_config(config: Union[str, dict], device: str = None) -> ChessModelBase:
    """
    Create a model from configuration.

    Args:
        config: Either path to YAML config file or config dict
        device: Device to put model on ("cuda", "cpu", or None for auto)

    Returns:
        Initialized model on specified device

    Example config:
        model:
          type: "cnn"  # or "cnn_lite", "transformer", "transformer_lite"
          params:
            num_residual_blocks: 5
            num_channels: 128
    """
    # Load config if it's a file path
    if isinstance(config, str):
        config_path = Path(config)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract model configuration
    model_config = config.get("model", {})
    model_type = model_config.get("type", "cnn_lite")
    model_params = model_config.get("params", {})

    # Create model based on type
    if model_type == "cnn":
        model = ChessCNN(**model_params)
    elif model_type == "cnn_lite":
        model = ChessCNNLite(**model_params)
    elif model_type == "transformer":
        model = ChessTransformer(**model_params)
    elif model_type == "transformer_lite":
        model = ChessTransformerLite(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Move to device
    model = model.to(device)

    # Print model info
    info = model.get_model_info()
    print(f"Created model: {info['architecture']}")
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Device: {device}")

    return model


def create_model(
    model_type: str = "cnn_lite",
    device: str = None,
    **kwargs
) -> ChessModelBase:
    """
    Simple factory for creating models without config file.

    Args:
        model_type: "cnn", "cnn_lite", "transformer", or "transformer_lite"
        device: Device to put model on
        **kwargs: Model-specific parameters

    Returns:
        Initialized model
    """
    config = {
        "model": {
            "type": model_type,
            "params": kwargs
        }
    }

    return create_model_from_config(config, device)


if __name__ == "__main__":
    # Test model creation
    print("Testing model factory...")

    print("\n1. CNN Lite:")
    model = create_model("cnn_lite")
    x = torch.randn(2, 12, 8, 8)
    policy, value = model(x)
    print(f"Input: {x.shape} → Policy: {policy.shape}, Value: {value.shape}")

    print("\n2. CNN Full:")
    model = create_model("cnn", num_residual_blocks=3, num_channels=64)
    policy, value = model(x)
    print(f"Input: {x.shape} → Policy: {policy.shape}, Value: {value.shape}")

    print("\n3. Transformer Lite:")
    model = create_model("transformer_lite")
    policy, value = model(x)
    print(f"Input: {x.shape} → Policy: {policy.shape}, Value: {value.shape}")

    print("\n✅ All models working!")
