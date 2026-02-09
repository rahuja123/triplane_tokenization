"""Triplane representation: storage and 3D point querying."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TriplaneConfig
from utils.geometry import world_to_grid_normalized


class TriplaneRepresentation(nn.Module):
    """
    Stores three orthogonal feature planes and provides 3D point querying.

    Planes:
        Pxy: (B, Df, Sx, Sy) - bird's eye view
        Pxz: (B, Df, Sx, Sz) - front view
        Pyz: (B, Df, Sy, Sz) - side view

    Querying a 3D point:
        1. Project to each plane's 2D coordinates
        2. Bilinear sample from each plane
        3. Aggregate via element-wise product
    """

    def __init__(self, config: TriplaneConfig):
        super().__init__()
        self.config = config
        self.pxy = None
        self.pxz = None
        self.pyz = None

    def set_planes(self, pxy, pxz, pyz):
        """
        Store triplane features.

        Args:
            pxy: (B, Sx, Sy, Df)
            pxz: (B, Sx, Sz, Df)
            pyz: (B, Sy, Sz, Df)
        """
        # Convert to (B, Df, H, W) for grid_sample
        self.pxy = pxy.permute(0, 3, 1, 2).contiguous()
        self.pxz = pxz.permute(0, 3, 1, 2).contiguous()
        self.pyz = pyz.permute(0, 3, 1, 2).contiguous()

    def query_3d(self, points):
        """
        Query triplane features at 3D world-space points.

        Args:
            points: (B, N, 3) world-space coordinates

        Returns:
            features: (B, N, Df) aggregated features
        """
        assert self.pxy is not None, "Call set_planes() first"

        B, N, _ = points.shape

        # Get normalized 2D coordinates for each plane
        xy_coords, xz_coords, yz_coords = world_to_grid_normalized(points, self.config)

        # grid_sample expects (B, 1, N, 2) grid for (B, C, H, W) input
        xy_grid = xy_coords.view(B, 1, N, 2)
        xz_grid = xz_coords.view(B, 1, N, 2)
        yz_grid = yz_coords.view(B, 1, N, 2)

        # Sample from each plane
        fxy = F.grid_sample(self.pxy, xy_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        fxz = F.grid_sample(self.pxz, xz_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        fyz = F.grid_sample(self.pyz, yz_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        # Shape: (B, Df, 1, N) -> (B, Df, N) -> (B, N, Df)
        fxy = fxy.squeeze(2).permute(0, 2, 1)
        fxz = fxz.squeeze(2).permute(0, 2, 1)
        fyz = fyz.squeeze(2).permute(0, 2, 1)

        # Element-wise product aggregation
        features = fxy * fxz * fyz

        return features
