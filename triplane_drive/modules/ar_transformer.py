"""Autoregressive causal transformer for trajectory prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import TriplaneConfig


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with causal self-attention."""

    def __init__(self, d_model, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)
        x = x + attn_out

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class TrajectoryTokenEmbedding(nn.Module):
    """Embeds discretized trajectory waypoints into d_ar space."""

    def __init__(self, config: TriplaneConfig):
        super().__init__()
        self.config = config
        # Separate embeddings for x and y coordinates
        self.embed_x = nn.Embedding(config.traj_vocab_size, config.d_ar)
        self.embed_y = nn.Embedding(config.traj_vocab_size, config.d_ar)
        # Token type embedding to distinguish x from y
        self.type_embed = nn.Embedding(2, config.d_ar)

    def discretize(self, trajectory):
        """
        Convert continuous (x, y) trajectory to discrete token indices.

        Args:
            trajectory: (B, T, 2) continuous positions in meters

        Returns:
            tokens_x: (B, T) discrete x indices
            tokens_y: (B, T) discrete y indices
        """
        vocab = self.config.traj_vocab_size
        traj_range = self.config.traj_range

        # Map from [-traj_range, traj_range] to [0, vocab-1]
        x_idx = ((trajectory[..., 0] + traj_range) / (2 * traj_range) * (vocab - 1))
        y_idx = ((trajectory[..., 1] + traj_range) / (2 * traj_range) * (vocab - 1))

        x_idx = x_idx.long().clamp(0, vocab - 1)
        y_idx = y_idx.long().clamp(0, vocab - 1)

        return x_idx, y_idx

    def undiscretize(self, tokens_x, tokens_y):
        """
        Convert discrete tokens back to continuous positions.

        Args:
            tokens_x: (B, T) discrete x indices
            tokens_y: (B, T) discrete y indices

        Returns:
            trajectory: (B, T, 2) continuous positions
        """
        vocab = self.config.traj_vocab_size
        traj_range = self.config.traj_range

        x = tokens_x.float() / (vocab - 1) * (2 * traj_range) - traj_range
        y = tokens_y.float() / (vocab - 1) * (2 * traj_range) - traj_range

        return torch.stack([x, y], dim=-1)

    def forward(self, trajectory):
        """
        Embed trajectory into interleaved x,y token sequence.

        Args:
            trajectory: (B, T, 2) continuous trajectory

        Returns:
            tokens: (B, T*2, d_ar) interleaved x,y embeddings
        """
        tokens_x, tokens_y = self.discretize(trajectory)

        emb_x = self.embed_x(tokens_x) + self.type_embed(torch.zeros_like(tokens_x))
        emb_y = self.embed_y(tokens_y) + self.type_embed(torch.ones_like(tokens_y))

        # Interleave: x0, y0, x1, y1, ...
        B, T, D = emb_x.shape
        interleaved = torch.stack([emb_x, emb_y], dim=2)  # (B, T, 2, D)
        tokens = interleaved.reshape(B, T * 2, D)

        return tokens


class ARTransformer(nn.Module):
    """
    Scaled-down GPT-style causal transformer.

    Input: [sensor_tokens, past_traj_tokens, future_traj_tokens(teacher forcing)]
    Output: hidden states for trajectory prediction

    Uses causal masking: each token can only attend to itself and previous tokens.
    Sensor tokens are bidirectional (can attend to all other sensor tokens).
    """

    def __init__(self, config: TriplaneConfig):
        super().__init__()
        self.config = config

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_ar, config.ar_num_heads,
                             config.ar_ffn_dim, config.ar_dropout)
            for _ in range(config.ar_num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_ar)

        # Positional embedding for the full sequence
        max_len = config.total_sensor_tokens + config.total_past_traj_tokens + config.total_future_traj_tokens + 10
        self.pos_embed = nn.Embedding(max_len, config.d_ar)

    def _create_causal_mask(self, seq_len, num_sensor_tokens, device):
        """
        Create attention mask:
        - Sensor tokens: bidirectional attention among themselves
        - Trajectory tokens: causal (can attend to all sensor + previous traj tokens)
        """
        # Start with causal mask (upper triangle = -inf)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))

        # Allow sensor tokens to attend to each other (bidirectional)
        mask[:num_sensor_tokens, :num_sensor_tokens] = 0

        return mask

    def forward(self, sensor_tokens, traj_tokens):
        """
        Args:
            sensor_tokens: (B, L_sensor, d_ar)
            traj_tokens: (B, L_traj, d_ar) past + future trajectory tokens

        Returns:
            hidden: (B, L_total, d_ar)
        """
        B = sensor_tokens.shape[0]
        L_sensor = sensor_tokens.shape[1]

        # Concatenate
        x = torch.cat([sensor_tokens, traj_tokens], dim=1)  # (B, L, d_ar)
        L = x.shape[1]

        # Add positional embeddings
        positions = torch.arange(L, device=x.device)
        x = x + self.pos_embed(positions).unsqueeze(0)

        # Create causal mask
        mask = self._create_causal_mask(L, L_sensor, x.device)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=mask)

        x = self.final_norm(x)
        return x
