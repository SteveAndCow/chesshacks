"""
Chess model inference with HuggingFace integration.

Handles downloading, caching, and loading models from HuggingFace Hub.
"""
import torch
import chess
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import sys
import os

# Add training scripts to path for model imports
training_scripts = Path(__file__).parent.parent.parent / "training" / "scripts"
sys.path.insert(0, str(training_scripts))


class ChessModelLoader:
    """
    Loads chess models from HuggingFace and provides inference interface.

    Handles:
    - Downloading models from HF Hub (with caching)
    - Loading model architecture
    - Converting board states to tensors
    - Running inference
    """

    def __init__(
        self,
        repo_id: str = "steveandcow/chesshacks-bot",
        model_name: str = "cnn_baseline",
        cache_dir: str = "./.model_cache",
        device: Optional[str] = None
    ):
        """
        Args:
            repo_id: HuggingFace repo ID (username/repo-name)
            model_name: Name of model file (without .pt extension)
            cache_dir: Local cache directory for downloaded models
            device: "cuda", "cpu", or None (auto-detect)
        """
        self.repo_id = repo_id
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.model_info = None

    def load_model(self) -> torch.nn.Module:
        """
        Download and load model from HuggingFace.

        Returns:
            Loaded model ready for inference
        """
        try:
            from huggingface_hub import hf_hub_download

            print(f"📥 Loading model from HuggingFace: {self.repo_id}/{self.model_name}")

            # Download model file (cached automatically by HF)
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"{self.model_name}.pt",
                cache_dir=str(self.cache_dir)
            )

            print(f"✅ Model downloaded to: {model_path}")

        except Exception as e:
            print(f"⚠️  Failed to download from HuggingFace: {e}")
            print(f"Looking for local model at {self.cache_dir}/{self.model_name}.pt")

            # Fallback to local model
            model_path = self.cache_dir / f"{self.model_name}.pt"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found on HuggingFace or locally. "
                    f"Please train a model first or check your repo_id."
                )

        # Load checkpoint
        print("Loading model weights...")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract model info
        self.model_info = {
            "model_type": checkpoint.get("model_type", "unknown"),
            "architecture": checkpoint.get("architecture", "unknown"),
            "epochs": checkpoint.get("epochs", 0),
        }

        print(f"Model info: {self.model_info}")

        # Create model architecture
        model_type = self.model_info["model_type"]

        if model_type in ["cnn", "cnn_lite"]:
            from models.cnn import ChessCNN, ChessCNNLite
            if model_type == "cnn":
                model = ChessCNN(**checkpoint.get("model_params", {}))
            else:
                model = ChessCNNLite(**checkpoint.get("model_params", {}))

        elif model_type in ["transformer", "transformer_lite"]:
            from models.transformer import ChessTransformer, ChessTransformerLite
            if model_type == "transformer":
                model = ChessTransformer(**checkpoint.get("model_params", {}))
            else:
                model = ChessTransformerLite(**checkpoint.get("model_params", {}))

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        self.model = model

        print(f"✅ Model loaded on {self.device}")
        print(f"Architecture: {model.get_architecture_name()}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """
        Convert chess board to tensor representation (16 channels).

        Args:
            board: python-chess Board object

        Returns:
            (1, 16, 8, 8) tensor ready for model input

        Channels:
            0-11: Piece positions (6 types × 2 colors)
            12-13: Castling rights (kingside/queenside)
            14: En passant file indicator
            15: Halfmove clock (normalized)
        """
        tensor = np.zeros((16, 8, 8), dtype=np.float32)

        piece_idx = {
            (chess.PAWN, chess.WHITE): 0,
            (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK, chess.WHITE): 3,
            (chess.QUEEN, chess.WHITE): 4,
            (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6,
            (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK, chess.BLACK): 9,
            (chess.QUEEN, chess.BLACK): 10,
            (chess.KING, chess.BLACK): 11,
        }

        # Fill piece positions (channels 0-11)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                idx = piece_idx[(piece.piece_type, piece.color)]
                rank = square // 8
                file = square % 8
                tensor[idx, rank, file] = 1.0

        # Castling rights (channels 12-13)
        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[12, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[13, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[13, :, :] = 1.0

        # En passant file (channel 14)
        if board.ep_square is not None:
            file = board.ep_square % 8
            tensor[14, :, file] = 1.0

        # Halfmove clock (channel 15) - normalized to [0, 1]
        tensor[15, :, :] = min(board.halfmove_clock / 100.0, 1.0)

        # Convert to torch and add batch dimension
        tensor = torch.from_numpy(tensor).unsqueeze(0)  # (1, 16, 8, 8)
        return tensor.to(self.device)

    def predict(
        self,
        board: chess.Board
    ) -> Tuple[Dict[chess.Move, float], float]:
        """
        Predict move probabilities and position value for a board.

        Args:
            board: Chess board to evaluate

        Returns:
            (move_probs, value) where:
            - move_probs: Dict mapping legal moves to probabilities
            - value: Position evaluation in [-1, 1]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert board to tensor
        board_tensor = self.board_to_tensor(board)

        # Run inference
        with torch.no_grad():
            policy_logits, value_pred, result_logits = self.model(board_tensor)

        # Convert policy logits to probabilities
        policy_probs = torch.softmax(policy_logits[0], dim=0)  # (4096,)

        # Map probabilities to legal moves only
        legal_moves = list(board.legal_moves)
        move_probs = {}

        for move in legal_moves:
            # Move encoding: from_square * 64 + to_square
            move_idx = move.from_square * 64 + move.to_square
            prob = policy_probs[move_idx].item()
            move_probs[move] = prob

        # Normalize probabilities over legal moves
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {m: p / total_prob for m, p in move_probs.items()}

        # Extract value
        value = value_pred[0, 0].item()  # Scalar in [-1, 1]

        return move_probs, value

    def get_best_move(self, board: chess.Board) -> Tuple[chess.Move, float]:
        """
        Get the best move according to the model.

        Args:
            board: Chess board

        Returns:
            (best_move, value) where best_move is the highest probability legal move
        """
        move_probs, value = self.predict(board)

        if not move_probs:
            raise ValueError("No legal moves available")

        best_move = max(move_probs.items(), key=lambda x: x[1])[0]
        return best_move, value

    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate position value (for MCTS).

        Args:
            board: Chess board

        Returns:
            Position value in [-1, 1]
        """
        _, value = self.predict(board)
        return value
