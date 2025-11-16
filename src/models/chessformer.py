import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePosition2D(nn.Module):
    """
    Shaw et al. style learned relative positional embeddings for 8x8 = 64 tokens.
    dx, dy ∈ [-7, 7], stored as [15 x 15] = 225 embeddings.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.rel_q = nn.Embedding(15 * 15, dim)
        self.rel_k = nn.Embedding(15 * 15, dim)
        self.rel_v = nn.Embedding(15 * 15, dim)

        # Precompute the 64×64 relative bias indices
        coords = torch.stack(torch.meshgrid(
            torch.arange(8), torch.arange(8), indexing='ij'
        )).reshape(2, -1)   # shape [2, 64]
        x, y = coords[0], coords[1]

        # dx[i,j], dy[i,j]
        dx = (x[:, None] - x[None, :]) + 7  # [0..14]
        dy = (y[:, None] - y[None, :]) + 7

        rel_index = dx * 15 + dy            # [64, 64]
        self.register_buffer("rel_index", rel_index)

    def forward(self):
        """
        Returns:
            a_q: [64, 64, dim]
            a_k: [64, 64, dim]
            a_v: [64, 64, dim]
        """
        idx = self.rel_index
        a_q = self.rel_q(idx)   # [64,64,dim]
        a_k = self.rel_k(idx)
        a_v = self.rel_v(idx)
        return a_q, a_k, a_v


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert dim % heads == 0

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.relpos = RelativePosition2D(self.head_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: [B, 64, dim]
        """
        B, N, D = x.shape

        # --- LayerNorm ---
        h = self.norm1(x)

        # --- Q,K,V ---
        qkv = self.qkv(h).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]   # each [B,64,H,Dh]

        # --- Relative Pos Emb ---
        a_q, a_k, a_v = self.relpos()   # [64,64,Dh]

        # Add a_ij^Q and a_ij^K:
        q_rel = q.unsqueeze(2) + a_q    # [B,H,64,64,Dh]
        k_rel = k.unsqueeze(1) + a_k    # [B,H,64,64,Dh]

        # Compute logits:
        logits = (q_rel * k_rel).sum(-1) / (self.head_dim ** 0.5)
        attn = logits.softmax(dim=-1)

        # Compute output with a_ij^V:
        v_rel = v.unsqueeze(1) + a_v    # [B,H,1-or-64,64,Dh]
        out = (attn.unsqueeze(-1) * v_rel).sum(-2)  # [B,H,64,Dh]

        out = out.transpose(1,2).reshape(B, N, D)
        x = x + self.proj(out)

        # MLP block
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerBackbone(nn.Module):
    def __init__(self, num_filters, depth=6, heads=8):
        super().__init__()
        self.depth = depth
        self.num_filters = num_filters
        self.blocks = nn.ModuleList([
            TransformerBlock(num_filters, heads=heads)
            for _ in range(depth)
        ])

    def forward(self, x):
        """
        x: [B, num_filters, 8, 8]
        -> flatten to tokens
        """
        B, C, H, W = x.shape
        assert H == 8 and W == 8

        x = x.reshape(B, C, 64).transpose(1,2)  # [B,64,C]

        for blk in self.blocks:
            x = blk(x)

        # reshape back
        x = x.transpose(1,2).reshape(B, C, 8, 8)
        return x

class LeelaZeroTransformer(nn.Module):
    def __init__(
        self,
        num_filters=256,
        num_residual_blocks=12,
        se_ratio=None,   # unused but kept for compatibility
        heads=8,
        **kwargs
    ):
        super().__init__()

        # Keep the LC0 input ConvBlock exactly
        self.input_block = ConvBlock(
            input_channels=112, filter_size=3, output_channels=num_filters
        )

        # Replace residual tower with transformers
        self.backbone = TransformerBackbone(
            num_filters=num_filters,
            depth=num_residual_blocks,
            heads=heads
        )

        # Policy/value heads unchanged
        self.policy_head = ConvolutionalPolicyHead(num_filters=num_filters)
        self.value_head = ConvolutionalValueOrMovesLeftHead(
            input_dim=num_filters, output_dim=3, num_filters=32,
            hidden_dim=128, relu=False,
        )
        self.moves_left_head = ConvolutionalValueOrMovesLeftHead(
            input_dim=num_filters, output_dim=1, num_filters=8,
            hidden_dim=128, relu=True,
        )

    def forward(self, planes):
        B = planes.shape[0]
        x = planes.reshape(B, 112, 8, 8)

        x = self.input_block(x)
        x = self.backbone(x)

        policy = self.policy_head(x)
        value = self.value_head(x)
        moves_left = self.moves_left_head(x)
        return ModelOutput(policy, value, moves_left)
