"""
LC0 (Leela Chess Zero) model inference with HuggingFace integration.

Handles the 112-channel board representation and 1858-move policy encoding.
"""
import torch
import chess
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional


def board_to_112_channels(board: chess.Board) -> np.ndarray:
    """
    Convert chess board to 112-channel LC0 format (inference version).

    For inference, we only use the current position (no history).

    Returns:
        (112, 8, 8) numpy array
    """
    channels = []

    # 104 channels: 8 board positions × 13 planes each
    # For inference without history, repeat current position 8 times
    positions = [board] * 8

    for pos in positions:
        # 12 piece planes (6 types × 2 colors)
        for piece_type in chess.PIECE_TYPES:
            # Our pieces
            plane = np.zeros(64, dtype=np.float32)
            for square in pos.pieces(piece_type, board.turn):
                plane[square] = 1.0
            channels.append(plane)

            # Opponent pieces
            plane = np.zeros(64, dtype=np.float32)
            for square in pos.pieces(piece_type, not board.turn):
                plane[square] = 1.0
            channels.append(plane)

        # Repetition counter (always 8 since same position)
        channels.append(np.full(64, 8, dtype=np.float32))

    # 5 unit planes (castling rights + side to move)
    channels.append(np.full(64, float(board.has_kingside_castling_rights(board.turn))))
    channels.append(np.full(64, float(board.has_queenside_castling_rights(board.turn))))
    channels.append(np.full(64, float(board.has_kingside_castling_rights(not board.turn))))
    channels.append(np.full(64, float(board.has_queenside_castling_rights(not board.turn))))
    channels.append(np.full(64, 1.0 if board.turn == chess.WHITE else 0.0))

    # 1 rule50 plane
    channels.append(np.full(64, board.halfmove_clock / 99.0, dtype=np.float32))

    # 2 constant planes
    channels.append(np.zeros(64, dtype=np.float32))
    channels.append(np.ones(64, dtype=np.float32))

    return np.stack(channels).reshape(112, 8, 8)


class LC0ModelLoader:
    """
    Loads LC0 models from HuggingFace and provides inference interface.

    Differences from standard chess models:
    - 112-channel input representation (vs 16 channels)
    - 1858 policy outputs (LC0 standard vs 4096 simplified)
    - WDL (win/draw/loss) value head (3 outputs vs 1)
    """

    def __init__(
        self,
        repo_id: str = "steveandcow/chesshacks-lc0",
        model_file: str = "lc0_128x6.pt",
        cache_dir: str = "./.model_cache",
        device: Optional[str] = None
    ):
        """
        Args:
            repo_id: HuggingFace repo ID (username/repo-name)
            model_file: Name of model file (with .pt extension)
            cache_dir: Local cache directory for downloaded models
            device: "cuda", "cpu", or None (auto-detect)
        """
        self.repo_id = repo_id
        self.model_file = model_file
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.policy_map = None

    def load_model(self) -> torch.nn.Module:
        """
        Download and load LC0 model from HuggingFace.

        Returns:
            Loaded model ready for inference
        """
        try:
            from huggingface_hub import hf_hub_download

            print(f"📥 Loading LC0 model from HuggingFace: {self.repo_id}/{self.model_file}")

            # Download model file (cached automatically by HF)
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.model_file,
                cache_dir=str(self.cache_dir)
            )

            print(f"✅ Model downloaded to: {model_path}")

        except Exception as e:
            print(f"⚠️  Failed to download from HuggingFace: {e}")
            print(f"Looking for local model at {self.cache_dir}/{self.model_file}")

            # Fallback to local model
            model_path = self.cache_dir / self.model_file
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found on HuggingFace or locally. "
                    f"Please train a model first or check your repo_id."
                )

        # Load checkpoint
        print("Loading model weights...")
        checkpoint = torch.load(model_path, map_location=self.device)

        print(f"Model config: {checkpoint.get('config', 'N/A')}")

        # Import LC0 model architecture (lightweight inference-only version)
        from .lc0_net import LeelaZeroNet

        # Extract config
        config = checkpoint.get('config', {})

        # Create model with config (inference only needs these 3 params)
        model = LeelaZeroNet(
            num_filters=config.get('num_filters', 128),
            num_residual_blocks=config.get('num_residual_blocks', 6),
            se_ratio=config.get('se_ratio', 8)
        )

        # Load weights - handle torch.compile() prefix
        state_dict = checkpoint['model_state_dict']

        # Strip '_orig_mod.' prefix added by torch.compile()
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        self.model = model

        # Build policy map for move decoding (UCI string -> policy index)
        from .policy_index import policy_index
        self.policy_map = {move: idx for idx, move in enumerate(policy_index)}

        print(f"✅ LC0 model loaded on {self.device}")
        print(f"Architecture: {config.get('num_filters', 128)}x{config.get('num_residual_blocks', 6)}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Val Loss: {checkpoint.get('val_loss', 'N/A')}")

        return model

    def board_to_lc0_tensor(self, board: chess.Board) -> torch.Tensor:
        """
        Convert chess board to LC0's 112-channel representation.

        Returns:
            (1, 112, 8, 8) tensor
        """
        # Convert board to 112 channels
        channels = board_to_112_channels(board)  # (112, 8, 8)

        # Convert to torch and add batch dimension
        tensor = torch.from_numpy(channels).unsqueeze(0)  # (1, 112, 8, 8)
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
            - value: Position evaluation (win probability from current player's perspective)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert board to tensor
        board_tensor = self.board_to_lc0_tensor(board)

        # Run inference
        with torch.no_grad():
            model_output = self.model(board_tensor)
            policy_logits = model_output.policy  # (1, 1858)
            wdl_logits = model_output.value      # (1, 3) - win/draw/loss

        # Convert WDL to value
        wdl_probs = torch.softmax(wdl_logits[0], dim=0)  # (3,)
        win_prob = wdl_probs[0].item()
        draw_prob = wdl_probs[1].item()
        loss_prob = wdl_probs[2].item()

        # Value = P(win) - P(loss), range [-1, 1]
        value = win_prob - loss_prob

        # Convert policy logits to probabilities
        policy_probs = torch.softmax(policy_logits[0], dim=0)  # (1858,)

        # Map probabilities to legal moves
        legal_moves = list(board.legal_moves)
        move_probs = {}

        for move in legal_moves:
            # Convert move to UCI
            move_str = move.uci()

            # Mirror for black
            if board.turn == chess.BLACK:
                from_square = chess.square_mirror(move.from_square)
                to_square = chess.square_mirror(move.to_square)
                flipped_move = chess.Move(from_square, to_square, move.promotion)
                move_str = flipped_move.uci()

            # Look up in policy map
            if move_str in self.policy_map:
                policy_idx = self.policy_map[move_str]
                prob = policy_probs[policy_idx].item()
                move_probs[move] = prob
            else:
                # Fallback for moves not in policy map (shouldn't happen)
                move_probs[move] = 1e-6

        # Normalize probabilities over legal moves
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {m: p / total_prob for m, p in move_probs.items()}
        else:
            # Uniform distribution if all probs are 0
            uniform_prob = 1.0 / len(legal_moves)
            move_probs = {m: uniform_prob for m in legal_moves}

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
