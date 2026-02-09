"""3D Lifting module: simplified deformable cross-attention for 2D->3D feature lifting."""

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
        Hf, Wf = image_features.shape[2], image_features.shape[3]

        residual = queries
        queries = self.norm(queries)

        # Predict offsets from queries
        offsets = self.offset_pred(queries)  # (B, N, num_points*2)
        offsets = offsets.view(B, N, self.num_points, 2) * 0.05  # small offsets

        # Process each camera
        aggregated = torch.zeros_like(queries)
        total_weight = torch.zeros(B, N, 1, device=queries.device)

        for c in range(C):
            cam_feat = image_features[:, c]  # (B, Hf, Wf, D)
            cam_feat_2d = cam_feat.permute(0, 3, 1, 2)  # (B, D, Hf, Wf)
            cam_coords = pixel_coords[:, c]  # (B, N, 2)
            cam_valid = valid_mask[:, c].float()  # (B, N)

            # Sample at projected location + offsets
            # Expand coords for offset points
            coords_expanded = cam_coords.unsqueeze(2) + offsets  # (B, N, num_points, 2)
            coords_flat = coords_expanded.view(B, N * self.num_points, 2)

            # grid_sample: (B, D, Hf, Wf) with grid (B, 1, N*K, 2)
            grid = coords_flat.view(B, 1, N * self.num_points, 2)
            sampled = F.grid_sample(cam_feat_2d, grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=True)
            # (B, D, 1, N*K) -> (B, N, K, D)
            sampled = sampled.squeeze(2).permute(0, 2, 1).view(B, N, self.num_points, D)

            # Average over offset points
            sampled_mean = sampled.mean(dim=2)  # (B, N, D)

            # Compute attention
            q = self.q_proj(queries)
            k = self.k_proj(sampled_mean)
            v = self.v_proj(sampled_mean)

            # Simple dot-product attention per query
            attn = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(D)
            attn = attn * cam_valid.unsqueeze(-1)

            aggregated = aggregated + attn * v
            total_weight = total_weight + cam_valid.unsqueeze(-1)

        # Normalize by number of valid cameras
        total_weight = total_weight.clamp(min=1.0)
        aggregated = aggregated / total_weight

        return residual + self.out_proj(aggregated)


class CrossImageAttentionLayer(nn.Module):
    """
    Each 3D query attends to features from ALL cameras jointly.
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

        # Sample features from all cameras at projected locations
        all_sampled = []
        all_valid = []

        for c in range(C):
            cam_feat_2d = image_features[:, c].permute(0, 3, 1, 2)  # (B, D, Hf, Wf)
            grid = pixel_coords[:, c].view(B, 1, N, 2)
            sampled = F.grid_sample(cam_feat_2d, grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=True)
            sampled = sampled.squeeze(2).permute(0, 2, 1)  # (B, N, D)
            all_sampled.append(sampled)
            all_valid.append(valid_mask[:, c])

        # Stack: (B, N, C, D)
        all_sampled = torch.stack(all_sampled, dim=2)
        all_valid = torch.stack(all_valid, dim=2).float()  # (B, N, C)

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
        Hf = image_features.shape[2]
        Wf = image_features.shape[3]

        # Flatten grid to (N, 3) where N = Sx*Sy*Sz
        grid_flat = self.grid_3d.reshape(-1, 3)  # (N, 3)
        N = grid_flat.shape[0]

        # Positional encoding for queries
        pos_enc = sinusoidal_positional_encoding_3d(grid_flat, D)  # (N, D)
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        # Initialize queries
        queries = self.query_base.expand(B, N, -1) + pos_enc  # (B, N, D)

        # Project all grid points to all cameras
        # Use full image dims for correct projection + validity, then remap
        # normalized coords to feature-map space for grid_sample
        grid_batch = grid_flat.unsqueeze(0).expand(B, -1, -1)  # (B, N, 3)
        pixel_coords, valid_mask = project_3d_to_2d(
            grid_batch, intrinsics, extrinsics,
            image_h=config.image_height, image_w=config.image_width
        )
        # pixel_coords are already normalized to [-1, 1] which works for
        # grid_sample on any spatial resolution (Hf, Wf)
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
