"""Volumetric renderer: NeRF-style rendering from triplane features for reconstruction loss.

Optimized with:
  - Chunk-based ray processing to reduce peak memory
  - cumsum-based transmittance (avoids cumprod, works on MPS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TriplaneConfig


class FeatureDecoder(nn.Module):
    """Lightweight MLP to decode triplane features to color + density."""

    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),  # 3 for RGB, 1 for density (sigma)
        )

    def forward(self, features):
        """
        Args:
            features: (..., Df) triplane features

        Returns:
            rgb: (..., 3) color values
            sigma: (..., 1) density values
        """
        out = self.net(features)
        rgb = torch.sigmoid(out[..., :3])
        sigma = F.softplus(out[..., 3:4])
        return rgb, sigma


class VolumetricRenderer(nn.Module):
    """
    Renders RGB images from triplane features for reconstruction loss.

    1. Cast rays from camera pixels
    2. Sample points along each ray
    3. Query triplane for features at sample points (in chunks)
    4. Decode to color + density
    5. Alpha compositing (volume rendering equation)
    """

    def __init__(self, config: TriplaneConfig):
        super().__init__()
        self.config = config
        self.decoder = FeatureDecoder(config.feature_dim)
        # Number of rays to process at once (per camera, per batch element)
        self.ray_chunk_size = getattr(config, 'ray_chunk_size', 2048)

    def _cast_rays(self, intrinsics, extrinsics, H, W, device):
        """
        Generate ray origins and directions for each pixel.

        Args:
            intrinsics: (B, C, 3, 3)
            extrinsics: (B, C, 4, 4) world-to-camera transforms
            H, W: render resolution
            device: torch device

        Returns:
            ray_origins: (B, C, H*W, 3) in world space
            ray_directions: (B, C, H*W, 3) in world space
        """
        B, C = intrinsics.shape[:2]

        # Pixel grid
        v, u = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        u = u.reshape(-1)  # (H*W,)
        v = v.reshape(-1)

        # Unproject to camera space: (u, v, 1) -> camera coords
        fx = intrinsics[:, :, 0, 0]  # (B, C)
        fy = intrinsics[:, :, 1, 1]
        cx = intrinsics[:, :, 0, 2]
        cy = intrinsics[:, :, 1, 2]

        # Direction in camera space
        dir_x = (u.unsqueeze(0).unsqueeze(0) - cx.unsqueeze(-1)) / fx.unsqueeze(-1)
        dir_y = (v.unsqueeze(0).unsqueeze(0) - cy.unsqueeze(-1)) / fy.unsqueeze(-1)
        dir_z = torch.ones_like(dir_x)

        dirs_cam = torch.stack([dir_x, dir_y, dir_z], dim=-1)  # (B, C, H*W, 3)

        # Transform to world space
        R = extrinsics[:, :, :3, :3]  # (B, C, 3, 3)
        t = extrinsics[:, :, :3, 3]   # (B, C, 3)

        R_inv = R.transpose(-1, -2)  # (B, C, 3, 3)
        cam_origin = -torch.einsum('bcij,bcj->bci', R_inv, t)  # (B, C, 3)

        # Rotate directions to world space
        dirs_world = torch.einsum('bcij,bcnj->bcni', R_inv, dirs_cam)  # (B, C, H*W, 3)
        dirs_world = F.normalize(dirs_world, dim=-1)

        ray_origins = cam_origin.unsqueeze(2).expand(-1, -1, H * W, -1)  # (B, C, H*W, 3)

        return ray_origins, dirs_world

    def _render_rays(self, triplane, ray_origins, ray_dirs, t_vals_base, training):
        """
        Render a batch of rays. This is the inner loop called per-chunk.

        Args:
            triplane: TriplaneRepresentation
            ray_origins: (B, R, 3) ray origins for R rays
            ray_dirs: (B, R, 3) ray directions
            t_vals_base: (S,) base sample distances
            training: bool

        Returns:
            colors: (B, R, 3) rendered colors
        """
        B, R, _ = ray_origins.shape
        S = t_vals_base.shape[0]
        device = ray_origins.device
        near, far = self.config.render_near, self.config.render_far

        # Stratified sampling
        if training:
            noise = torch.rand(B, R, S, device=device)
            step = (far - near) / S
            t_vals = t_vals_base.view(1, 1, -1) + noise * step  # (B, R, S)
        else:
            t_vals = t_vals_base.view(1, 1, -1).expand(B, R, -1)  # (B, R, S)

        # Compute sample points: origin + t * direction
        # (B, R, 1, 3) + (B, R, S, 1) * (B, R, 1, 3) → (B, R, S, 3)
        sample_points = ray_origins.unsqueeze(2) + t_vals.unsqueeze(-1) * ray_dirs.unsqueeze(2)

        # Query triplane — flatten to (B, R*S, 3)
        points_flat = sample_points.reshape(B, R * S, 3)
        features = triplane.query_3d(points_flat)  # (B, R*S, Df)
        features = features.reshape(B, R, S, -1)

        # Decode to color + density
        rgb, sigma = self.decoder(features)
        # rgb: (B, R, S, 3), sigma: (B, R, S, 1)

        # Volume rendering (alpha compositing)
        deltas = t_vals[..., 1:] - t_vals[..., :-1]  # (B, R, S-1)
        deltas = torch.cat([deltas, torch.full_like(deltas[..., :1], 1e10)], dim=-1)  # (B, R, S)
        deltas = deltas.unsqueeze(-1)  # (B, R, S, 1)

        # Alpha = 1 - exp(-sigma * delta)
        sigma_delta = sigma * deltas  # (B, R, S, 1)

        # Transmittance via cumsum (avoids cumprod, works on MPS)
        # T_i = exp(-sum_{j<i} sigma_j * delta_j)
        cumulative = torch.cumsum(sigma_delta, dim=2)  # (B, R, S, 1)
        # Shift right: T_0 = 1, T_i = exp(-sum_{j=0..i-1})
        transmittance = torch.exp(-(cumulative - sigma_delta))  # (B, R, S, 1)

        alpha = 1.0 - torch.exp(-sigma_delta)
        weights = alpha * transmittance  # (B, R, S, 1)

        # Final color
        colors = (weights * rgb).sum(dim=2)  # (B, R, 3)

        return colors

    def forward(self, triplane, intrinsics, extrinsics, camera_indices=None):
        """
        Render images from triplane representation.

        Args:
            triplane: TriplaneRepresentation with planes set
            intrinsics: (B, C, 3, 3)
            extrinsics: (B, C, 4, 4)
            camera_indices: optional list of camera indices to render

        Returns:
            rendered: (B, C_render, 3, H_render, W_render) rendered RGB images
            camera_indices: list of rendered camera indices
        """
        config = self.config
        B = intrinsics.shape[0]
        C_total = intrinsics.shape[1]

        H_render = int(config.image_height * config.render_image_scale)
        W_render = int(config.image_width * config.render_image_scale)

        # Select cameras to render
        if camera_indices is None:
            num_render = min(config.num_render_cameras, C_total)
            camera_indices = torch.randperm(C_total)[:num_render].tolist()

        C_render = len(camera_indices)

        # Scale intrinsics for render resolution
        scale = config.render_image_scale
        render_intrinsics = intrinsics[:, camera_indices].clone()
        render_intrinsics[:, :, 0] *= scale
        render_intrinsics[:, :, 1] *= scale
        render_intrinsics[:, :, 2, 2] = 1.0

        render_extrinsics = extrinsics[:, camera_indices]
        device = intrinsics.device

        # Cast rays
        ray_origins, ray_dirs = self._cast_rays(
            render_intrinsics, render_extrinsics, H_render, W_render, device
        )
        # ray_origins, ray_dirs: (B, C_render, H*W, 3)

        # Base t_vals (shared across all rays)
        num_samples = config.num_ray_samples
        near, far = config.render_near, config.render_far
        t_vals_base = torch.linspace(0, 1, num_samples, device=device)
        t_vals_base = near + (far - near) * t_vals_base  # (S,)

        total_rays = H_render * W_render
        chunk = self.ray_chunk_size

        # Process cameras and ray chunks
        all_rendered = []
        for c_idx in range(C_render):
            cam_origins = ray_origins[:, c_idx]  # (B, H*W, 3)
            cam_dirs = ray_dirs[:, c_idx]          # (B, H*W, 3)

            if total_rays <= chunk:
                # Small enough to process in one go
                cam_colors = self._render_rays(
                    triplane, cam_origins, cam_dirs, t_vals_base, self.training
                )  # (B, H*W, 3)
            else:
                # Process in chunks
                color_chunks = []
                for start in range(0, total_rays, chunk):
                    end = min(start + chunk, total_rays)
                    chunk_colors = self._render_rays(
                        triplane,
                        cam_origins[:, start:end],
                        cam_dirs[:, start:end],
                        t_vals_base,
                        self.training,
                    )  # (B, chunk_size, 3)
                    color_chunks.append(chunk_colors)
                cam_colors = torch.cat(color_chunks, dim=1)  # (B, H*W, 3)

            all_rendered.append(cam_colors)

        # Stack cameras and reshape to image
        rendered = torch.stack(all_rendered, dim=1)  # (B, C_render, H*W, 3)
        rendered = rendered.reshape(B, C_render, H_render, W_render, 3)
        rendered = rendered.permute(0, 1, 4, 2, 3)  # (B, C_render, 3, H, W)

        return rendered, camera_indices
