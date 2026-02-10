"""3D Lifting module: simplified deformable cross-attention for 2D->3D feature lifting.

Optimized with:
  - Vectorized camera loops (batched grid_sample across all cameras)
  - Cached positional encoding as buffer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import TriplaneConfig
from utils.geometry import project_3d_to_2d, create_triplane_grid
from utils.positional_encoding import sinusoidal_positional_encoding_3d


class PerImageAttentionLayer(nn.Module):
    """
    Each 3D query attends to features sampled from a single camera image.
    Uses projected 2D locations + small offset grid as key/value positions.

    Vectorized: processes all cameras in a single batched grid_sample call.
    """

    def __init__(self, dim, num_heads, num_points=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Learnable offsets for deformable sampling
        self.offset_pred = nn.Linear(dim, num_points * 2)
        nn.init.zeros_(self.offset_pred.weight)
        nn.init.zeros_(self.offset_pred.bias)

        self.norm = nn.LayerNorm(dim)

    def forward(self, queries, image_features, pixel_coords, valid_mask):
        """
        Args:
            queries: (B, N, D) 3D query features
            image_features: (B, C, Hf, Wf, D) per-camera feature maps
            pixel_coords: (B, C, N, 2) projected normalized pixel coordinates
            valid_mask: (B, C, N) visibility mask

        Returns:
            updated_queries: (B, N, D)
        """
        B, N, D = queries.shape
        C = image_features.shape[1]
        K = self.num_points

        residual = queries
        queries = self.norm(queries)

        # Predict offsets from queries
        offsets = self.offset_pred(queries)  # (B, N, K*2)
        offsets = offsets.view(B, N, K, 2) * 0.05  # small offsets

        # === Vectorized: batch all cameras into one grid_sample call ===
        # Reshape features: (B, C, Hf, Wf, D) → (B*C, D, Hf, Wf)
        Hf, Wf = image_features.shape[2], image_features.shape[3]
        cam_feat_2d = image_features.reshape(B * C, Hf, Wf, D).permute(0, 3, 1, 2)  # (B*C, D, Hf, Wf)

        # Expand coords for offset points: (B, C, N, 2) → (B, C, N, K, 2)
        coords_expanded = pixel_coords.unsqueeze(3) + offsets.unsqueeze(1)  # (B, C, N, K, 2)
        # Reshape for grid_sample: (B*C, 1, N*K, 2)
        coords_flat = coords_expanded.reshape(B * C, N * K, 2)
        grid = coords_flat.unsqueeze(1)  # (B*C, 1, N*K, 2)

        # Single batched grid_sample for all cameras
        sampled = F.grid_sample(cam_feat_2d, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        # (B*C, D, 1, N*K) → (B, C, N, K, D)
        sampled = sampled.squeeze(2).permute(0, 2, 1)  # (B*C, N*K, D)
        sampled = sampled.view(B, C, N, K, D)

        # Average over offset points: (B, C, N, D)
        sampled_mean = sampled.mean(dim=3)

        # Compute attention per camera and aggregate
        q = self.q_proj(queries)  # (B, N, D)
        k = self.k_proj(sampled_mean)  # (B, C, N, D)
        v = self.v_proj(sampled_mean)  # (B, C, N, D)

        # Attention: (B, 1, N, D) * (B, C, N, D) → (B, C, N, 1)
        attn = (q.unsqueeze(1) * k).sum(dim=-1, keepdim=True) / math.sqrt(D)
        # Mask invalid cameras
        attn = attn * valid_mask.unsqueeze(-1).float()  # (B, C, N, 1)

        # Weighted sum across cameras
        aggregated = (attn * v).sum(dim=1)  # (B, N, D)
        total_weight = valid_mask.float().sum(dim=1).clamp(min=1.0).unsqueeze(-1)  # (B, N, 1)
        aggregated = aggregated / total_weight

        return residual + self.out_proj(aggregated)


class CrossImageAttentionLayer(nn.Module):
    """
    Each 3D query attends to features from ALL cameras jointly.
    Vectorized: single batched grid_sample for all cameras.
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, queries, image_features, pixel_coords, valid_mask):
        """
        Args:
            queries: (B, N, D) 3D query features
            image_features: (B, C, Hf, Wf, D)
            pixel_coords: (B, C, N, 2)
            valid_mask: (B, C, N)

        Returns:
            updated_queries: (B, N, D)
        """
        B, N, D = queries.shape
        C = image_features.shape[1]

        residual = queries
        queries = self.norm(queries)

        # === Vectorized grid_sample across all cameras ===
        Hf, Wf = image_features.shape[2], image_features.shape[3]
        cam_feat_2d = image_features.reshape(B * C, Hf, Wf, D).permute(0, 3, 1, 2)  # (B*C, D, Hf, Wf)
        grid = pixel_coords.reshape(B * C, N, 2).unsqueeze(1)  # (B*C, 1, N, 2)

        sampled = F.grid_sample(cam_feat_2d, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        # (B*C, D, 1, N) → (B, C, N, D)
        sampled = sampled.squeeze(2).permute(0, 2, 1).view(B, C, N, D)

        # Stack: (B, N, C, D)
        all_sampled = sampled.permute(0, 2, 1, 3)  # (B, N, C, D)
        all_valid = valid_mask.permute(0, 2, 1).float()  # (B, N, C)

        # Cross-attention: query attends to C camera features
        q = self.q_proj(queries)  # (B, N, D)
        kv = self.kv_proj(all_sampled)  # (B, N, C, 2*D)
        k, v = kv.chunk(2, dim=-1)  # each (B, N, C, D)

        # Attention scores
        attn = torch.einsum('bnd,bncd->bnc', q, k) / math.sqrt(D)

        # Mask invalid cameras
        attn = attn.masked_fill(all_valid == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = attn.masked_fill(torch.isnan(attn), 0.0)

        # Weighted sum
        out = torch.einsum('bnc,bncd->bnd', attn, v)

        return residual + self.out_proj(out)


class LiftingModule(nn.Module):
    """
    Full 3D lifting pipeline:
    1. Creates 3D query grid with sinusoidal positional encoding
    2. Projects queries to each camera's image plane
    3. Applies 2 PerImageAttention + 2 CrossImageAttention layers
    4. Averages along axes to produce triplanes

    For memory efficiency, processes queries in chunks.
    Positional encoding is cached as a buffer.
    """

    def __init__(self, config: TriplaneConfig):
        super().__init__()
        self.config = config
        D = config.feature_dim

        # Query initialization: learnable base + positional encoding
        self.query_base = nn.Parameter(torch.randn(1, 1, D) * 0.02)

        # Attention layers: 2 per-image + 2 cross-image
        self.per_image_layers = nn.ModuleList([
            PerImageAttentionLayer(D, config.num_heads_lifting, config.num_deform_points)
            for _ in range(2)
        ])
        self.cross_image_layers = nn.ModuleList([
            CrossImageAttentionLayer(D, config.num_heads_lifting)
            for _ in range(2)
        ])

        # Register the triplane grid as a buffer (not a parameter)
        grid = create_triplane_grid(config)  # (Sx, Sy, Sz, 3)
        self.register_buffer('grid_3d', grid)

        # Cache positional encoding as a buffer (computed once, not every forward)
        grid_flat = grid.reshape(-1, 3)
        pos_enc = sinusoidal_positional_encoding_3d(grid_flat, D)  # (N, D)
        self.register_buffer('pos_enc', pos_enc)

        self.chunk_size = 4096  # process this many queries at a time

    def forward(self, image_features, intrinsics, extrinsics):
        """
        Args:
            image_features: (B, C, Hf, Wf, Df) per-camera feature maps
            intrinsics: (B, C, 3, 3) camera intrinsics
            extrinsics: (B, C, 4, 4) camera extrinsics

        Returns:
            pxy: (B, Sx, Sy, Df) BEV plane
            pxz: (B, Sx, Sz, Df) front plane
            pyz: (B, Sy, Sz, Df) side plane
        """
        B = image_features.shape[0]
        config = self.config
        Sx, Sy, Sz = config.sx, config.sy, config.sz
        D = config.feature_dim

        # Flatten grid to (N, 3) where N = Sx*Sy*Sz
        grid_flat = self.grid_3d.reshape(-1, 3)  # (N, 3)
        N = grid_flat.shape[0]

        # Use cached positional encoding (expand for batch)
        pos_enc = self.pos_enc.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        # Initialize queries
        queries = self.query_base.expand(B, N, -1) + pos_enc  # (B, N, D)

        # Project all grid points to all cameras
        grid_batch = grid_flat.unsqueeze(0).expand(B, -1, -1)  # (B, N, 3)
        pixel_coords, valid_mask = project_3d_to_2d(
            grid_batch, intrinsics, extrinsics,
            image_h=config.image_height, image_w=config.image_width
        )
        # pixel_coords: (B, C, N, 2), valid_mask: (B, C, N)

        # Process in chunks for memory efficiency
        output_queries = torch.zeros_like(queries)

        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            q_chunk = queries[:, start:end]
            pc_chunk = pixel_coords[:, :, start:end]
            vm_chunk = valid_mask[:, :, start:end]

            # Per-image attention layers
            for layer in self.per_image_layers:
                q_chunk = layer(q_chunk, image_features, pc_chunk, vm_chunk)

            # Cross-image attention layers
            for layer in self.cross_image_layers:
                q_chunk = layer(q_chunk, image_features, pc_chunk, vm_chunk)

            output_queries[:, start:end] = q_chunk

        # Reshape to (B, Sx, Sy, Sz, D) and average along axes
        volume = output_queries.view(B, Sx, Sy, Sz, D)

        pxy = volume.mean(dim=3)  # (B, Sx, Sy, D) - average over z
        pxz = volume.mean(dim=2)  # (B, Sx, Sz, D) - average over y
        pyz = volume.mean(dim=1)  # (B, Sy, Sz, D) - average over x

        return pxy, pxz, pyz
