"""Volumetric renderer: NeRF-style rendering from triplane features for reconstruction loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TriplaneConfig


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
    3. Query triplane for features at sample points
    4. Decode to color + density
    5. Alpha compositing (volume rendering equation)
    """

    def __init__(self, config: TriplaneConfig):
        super().__init__()
        self.config = config
        self.decoder = FeatureDecoder(config.feature_dim)

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
        # K_inv @ [u, v, 1]^T
        fx = intrinsics[:, :, 0, 0]  # (B, C)
        fy = intrinsics[:, :, 1, 1]
        cx = intrinsics[:, :, 0, 2]
        cy = intrinsics[:, :, 1, 2]

        # Direction in camera space
        dir_x = (u.unsqueeze(0).unsqueeze(0) - cx.unsqueeze(-1)) / fx.unsqueeze(-1)  # (B, C, H*W)
        dir_y = (v.unsqueeze(0).unsqueeze(0) - cy.unsqueeze(-1)) / fy.unsqueeze(-1)
        dir_z = torch.ones_like(dir_x)

        dirs_cam = torch.stack([dir_x, dir_y, dir_z], dim=-1)  # (B, C, H*W, 3)

        # Transform to world space
        # extrinsics is world-to-camera: [R|t], so camera-to-world = [R^T | -R^T @ t]
        R = extrinsics[:, :, :3, :3]  # (B, C, 3, 3)
        t = extrinsics[:, :, :3, 3]   # (B, C, 3)

        R_inv = R.transpose(-1, -2)  # (B, C, 3, 3)
        cam_origin = -torch.einsum('bcij,bcj->bci', R_inv, t)  # (B, C, 3)

        # Rotate directions to world space
        dirs_world = torch.einsum('bcij,bcnj->bcni', R_inv, dirs_cam)  # (B, C, H*W, 3)
        dirs_world = F.normalize(dirs_world, dim=-1)

        ray_origins = cam_origin.unsqueeze(2).expand(-1, -1, H * W, -1)  # (B, C, H*W, 3)

        return ray_origins, dirs_world

    def forward(self, triplane, intrinsics, extrinsics, camera_indices=None):
        """
        Render images from triplane representation.

        Args:
            triplane: TriplaneRepresentation with planes set
            intrinsics: (B, C, 3, 3)
            extrinsics: (B, C, 4, 4)
            camera_indices: optional list of camera indices to render (for efficiency)

        Returns:
            rendered: (B, C_render, 3, H_render, W_render) rendered RGB images
        """
        config = self.config
        B = intrinsics.shape[0]
        C_total = intrinsics.shape[1]

        H_render = int(config.image_height * config.render_image_scale)
        W_render = int(config.image_width * config.render_image_scale)

        # Select cameras to render
        if camera_indices is None:
            # Randomly select a subset for efficiency
            num_render = min(config.num_render_cameras, C_total)
            camera_indices = torch.randperm(C_total)[:num_render].tolist()

        C_render = len(camera_indices)

        # Scale intrinsics for render resolution
        scale = config.render_image_scale
        render_intrinsics = intrinsics[:, camera_indices].clone()
        render_intrinsics[:, :, 0] *= scale
        render_intrinsics[:, :, 1] *= scale
        render_intrinsics[:, :, 2, 2] = 1.0  # keep homogeneous

        render_extrinsics = extrinsics[:, camera_indices]

        device = intrinsics.device

        # Cast rays
        ray_origins, ray_dirs = self._cast_rays(
            render_intrinsics, render_extrinsics, H_render, W_render, device
        )
        # ray_origins, ray_dirs: (B, C_render, H*W, 3)

        num_samples = config.num_ray_samples
        near, far = config.render_near, config.render_far

        # Stratified sampling along rays
        t_vals = torch.linspace(0, 1, num_samples, device=device)
        t_vals = near + (far - near) * t_vals  # (num_samples,)
        # Add noise for stratified sampling
        if self.training:
            noise = torch.rand(B, C_render, H_render * W_render, num_samples, device=device)
            step = (far - near) / num_samples
            t_vals_expanded = t_vals.view(1, 1, 1, -1) + noise * step
        else:
            t_vals_expanded = t_vals.view(1, 1, 1, -1).expand(B, C_render, H_render * W_render, -1)

        # Compute sample points: origin + t * direction
        # (B, C, H*W, 1, 3) + (B, C, H*W, S, 1) * (B, C, H*W, 1, 3)
        sample_points = (
            ray_origins.unsqueeze(3) +
            t_vals_expanded.unsqueeze(-1) * ray_dirs.unsqueeze(3)
        )  # (B, C, H*W, S, 3)

        # Query triplane for all sample points
        # Reshape to (B, C*H*W*S, 3) for batch querying
        all_points = sample_points.reshape(B, -1, 3)
        features = triplane.query_3d(all_points)  # (B, C*H*W*S, Df)
        features = features.reshape(B, C_render, H_render * W_render, num_samples, -1)

        # Decode to color + density
        rgb, sigma = self.decoder(features)
        # rgb: (B, C, H*W, S, 3), sigma: (B, C, H*W, S, 1)

        # Volume rendering (alpha compositing)
        # delta = distance between adjacent samples
        deltas = t_vals_expanded[..., 1:] - t_vals_expanded[..., :-1]
        deltas = torch.cat([deltas, torch.ones_like(deltas[..., :1]) * 1e10], dim=-1)
        deltas = deltas.unsqueeze(-1)  # (B, C, H*W, S, 1)

        # Alpha = 1 - exp(-sigma * delta)
        alpha = 1.0 - torch.exp(-sigma * deltas)

        # Transmittance
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1, :]), 1.0 - alpha[..., :-1, :] + 1e-10], dim=-2),
            dim=-2
        )

        # Weights
        weights = alpha * transmittance  # (B, C, H*W, S, 1)

        # Final color
        rendered_color = (weights * rgb).sum(dim=-2)  # (B, C, H*W, 3)

        # Reshape to image
        rendered = rendered_color.reshape(B, C_render, H_render, W_render, 3)
        rendered = rendered.permute(0, 1, 4, 2, 3)  # (B, C, 3, H, W)

        return rendered, camera_indices
