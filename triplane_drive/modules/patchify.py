"""Triplane patchification: converts triplane features into token sequences."""

import torch
import torch.nn as nn

from config import TriplaneConfig


class TriplanePatchifier(nn.Module):
    """
    Converts triplane features into a flat sequence of tokens.

    For each plane Pij of shape (B, Si, Sj, Df):
        1. Reshape into patches of size (pi, pj)
        2. Flatten each patch: (pi * pj * Df)
        3. Project via MLP to d_ar

    Concatenate tokens from all three planes.
    Optionally apply halfplane masking (for front-only cameras).
    """

    def __init__(self, config: TriplaneConfig):
        super().__init__()
        self.config = config
        D = config.feature_dim
        d_ar = config.d_ar
        px, py, pz = config.patch_x, config.patch_y, config.patch_z

        # Per-plane MLPs (input dims differ)
        self.mlp_xy = nn.Linear(px * py * D, d_ar)
        self.mlp_xz = nn.Linear(px * pz * D, d_ar)
        self.mlp_yz = nn.Linear(py * pz * D, d_ar)

        # Learnable positional embeddings for each plane's tokens
        self.pos_xy = nn.Parameter(torch.randn(1, config.num_tokens_xy, d_ar) * 0.02)
        self.pos_xz = nn.Parameter(torch.randn(1, config.num_tokens_xz, d_ar) * 0.02)
        self.pos_yz = nn.Parameter(torch.randn(1, config.num_tokens_yz, d_ar) * 0.02)

    def _patchify_plane(self, plane, patch_h, patch_w, mlp):
        """
        Patchify a single plane.

        Args:
            plane: (B, H, W, D) feature plane
            patch_h: patch height
            patch_w: patch width
            mlp: projection layer

        Returns:
            tokens: (B, num_patches, d_ar)
        """
        B, H, W, D = plane.shape
        nH = H // patch_h
        nW = W // patch_w

        # Reshape into patches
        x = plane.reshape(B, nH, patch_h, nW, patch_w, D)
        x = x.permute(0, 1, 3, 2, 4, 5)  # (B, nH, nW, pH, pW, D)
        x = x.reshape(B, nH * nW, patch_h * patch_w * D)

        # Project to d_ar
        tokens = mlp(x)

        return tokens

    def forward(self, pxy, pxz, pyz):
        """
        Args:
            pxy: (B, Sx, Sy, Df) BEV plane
            pxz: (B, Sx, Sz, Df) front plane
            pyz: (B, Sy, Sz, Df) side plane

        Returns:
            tokens: (B, L, d_ar) concatenated tokens from all planes
            token_counts: tuple of (Lxy, Lxz, Lyz) for reference
        """
        config = self.config
        px, py, pz = config.patch_x, config.patch_y, config.patch_z

        tokens_xy = self._patchify_plane(pxy, px, py, self.mlp_xy) + self.pos_xy
        tokens_xz = self._patchify_plane(pxz, px, pz, self.mlp_xz) + self.pos_xz
        tokens_yz = self._patchify_plane(pyz, py, pz, self.mlp_yz) + self.pos_yz

        if config.use_halfplane:
            # Keep only the front half of xy and xz planes
            Lxy = tokens_xy.shape[1]
            Lxz = tokens_xz.shape[1]
            tokens_xy = tokens_xy[:, :Lxy // 2]
            tokens_xz = tokens_xz[:, :Lxz // 2]

        # Concatenate
        tokens = torch.cat([tokens_xy, tokens_xz, tokens_yz], dim=1)

        Lxy = tokens_xy.shape[1]
        Lxz = tokens_xz.shape[1]
        Lyz = tokens_yz.shape[1]

        return tokens, (Lxy, Lxz, Lyz)
