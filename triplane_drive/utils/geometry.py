"""Camera projection, coordinate transforms, and nonlinear grid construction."""

import torch
import torch.nn.functional as F
import math


def project_3d_to_2d(points_3d, intrinsics, extrinsics, image_h, image_w):
    """
    Project 3D world points onto camera image planes.

    Args:
        points_3d: (B, N, 3) world coordinates
        intrinsics: (B, C, 3, 3) camera intrinsic matrices
        extrinsics: (B, C, 4, 4) world-to-camera transforms
        image_h: image height in pixels
        image_w: image width in pixels

    Returns:
        pixel_coords: (B, C, N, 2) normalized pixel coordinates in [-1, 1]
        valid_mask: (B, C, N) boolean mask for points visible in each camera
    """
    B, N, _ = points_3d.shape
    C = intrinsics.shape[1]

    # Convert to homogeneous coordinates
    ones = torch.ones(B, N, 1, device=points_3d.device, dtype=points_3d.dtype)
    points_homo = torch.cat([points_3d, ones], dim=-1)  # (B, N, 4)

    # Expand for cameras: (B, 1, N, 4) and extrinsics: (B, C, 4, 4)
    points_homo = points_homo.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, N, 4)

    # Transform to camera coordinates: (B, C, N, 4) @ (B, C, 4, 4)^T -> (B, C, N, 4)
    points_cam = torch.einsum('bcnj,bcij->bcni', points_homo, extrinsics)
    points_cam = points_cam[..., :3]  # (B, C, N, 3)

    # Project to pixel coordinates
    points_pixel = torch.einsum('bcnj,bcij->bcni', points_cam, intrinsics)  # (B, C, N, 3)

    # Normalize by depth
    depth = points_pixel[..., 2:3].clamp(min=1e-5)
    uv = points_pixel[..., :2] / depth  # (B, C, N, 2)

    # Normalize to [-1, 1] for grid_sample
    pixel_coords = torch.stack([
        2.0 * uv[..., 0] / image_w - 1.0,
        2.0 * uv[..., 1] / image_h - 1.0,
    ], dim=-1)

    # Validity: in front of camera and within image bounds
    valid_mask = (
        (points_cam[..., 2] > 0.1) &
        (uv[..., 0] >= 0) & (uv[..., 0] < image_w) &
        (uv[..., 1] >= 0) & (uv[..., 1] < image_h)
    )

    return pixel_coords, valid_mask


def _nonlinear_mapping_1d(grid_idx, num_cells, inner_cells, inner_range, outer_range):
    """
    Nonlinear (bilinear resolution) mapping for one axis.

    Maps uniform grid indices [0, num_cells) to world coordinates.
    Inner region: fine resolution covering [-inner_range, inner_range].
    Outer region: coarser resolution extending to [-outer_range, outer_range].

    Args:
        grid_idx: tensor of grid indices in [0, num_cells)
        num_cells: total cells along this axis
        inner_cells: number of cells in the inner (fine) region
        inner_range: world-space extent of inner region
        outer_range: world-space extent of outer region
    """
    # Normalize to [0, 1]
    t = grid_idx.float() / num_cells

    # Center of each cell
    center = 0.5
    outer_cells = num_cells - inner_cells

    # Inner half-width in normalized space
    inner_half = inner_cells / (2.0 * num_cells)

    # Resolution per cell
    r_inner = inner_range / (inner_cells / 2.0) if inner_cells > 0 else 1.0
    r_outer = (outer_range - inner_range) / (outer_cells / 2.0) if outer_cells > 0 else 1.0

    # Piecewise mapping
    world = torch.zeros_like(grid_idx, dtype=torch.float32)

    # Inner region: [center - inner_half, center + inner_half]
    inner_mask = (t >= center - inner_half) & (t < center + inner_half)
    inner_t = (t - center) / inner_half  # [-1, 1] within inner region
    world = torch.where(inner_mask, inner_t * inner_range, world)

    # Lower outer region: [0, center - inner_half)
    lower_mask = t < center - inner_half
    lower_t = (center - inner_half - t) / (center - inner_half)  # [0, 1] going outward
    world = torch.where(lower_mask, -(inner_range + lower_t * (outer_range - inner_range)), world)

    # Upper outer region: [center + inner_half, 1]
    upper_mask = t >= center + inner_half
    upper_t = (t - center - inner_half) / (1.0 - center - inner_half + 1e-8)
    world = torch.where(upper_mask, inner_range + upper_t * (outer_range - inner_range), world)

    return world


def create_triplane_grid(config):
    """
    Create the 3D grid of query points for the triplane using nonlinear spacing.

    Returns:
        grid: (Sx, Sy, Sz, 3) tensor of world-space coordinates
    """
    sx, sy, sz = config.sx, config.sy, config.sz

    # X axis: symmetric, +-180m
    x_idx = torch.arange(sx)
    x_world = _nonlinear_mapping_1d(
        x_idx, sx, config.inner_cells_xy,
        inner_range=config.range_xy * config.inner_cells_xy / sx,
        outer_range=config.range_xy
    )

    # Y axis: symmetric, +-180m
    y_idx = torch.arange(sy)
    y_world = _nonlinear_mapping_1d(
        y_idx, sy, config.inner_cells_xy,
        inner_range=config.range_xy * config.inner_cells_xy / sy,
        outer_range=config.range_xy
    )

    # Z axis: asymmetric, [-3, 45]m
    z_idx = torch.arange(sz)
    # Map z: inner region [-3, 15]m with 36 cells, outer [15, 45]m with 12 cells
    z_world = torch.zeros(sz)
    z_inner = config.inner_cells_z_lower  # 36
    z_outer = config.outer_cells_z_upper  # 12

    # Inner region: cells [0, 36) -> [-3, 15]m
    for i in range(z_inner):
        z_world[i] = -config.range_z_below + (config.range_z_below + 15.0) * (i + 0.5) / z_inner

    # Outer region: cells [36, 48) -> [15, 45]m
    for i in range(z_outer):
        z_world[z_inner + i] = 15.0 + (config.range_z_above - 15.0) * (i + 0.5) / z_outer

    # Create meshgrid
    gx, gy, gz = torch.meshgrid(x_world, y_world, z_world, indexing='ij')
    grid = torch.stack([gx, gy, gz], dim=-1)  # (Sx, Sy, Sz, 3)

    return grid


def world_to_grid_normalized(points, config):
    """
    Map world-space coordinates to normalized [-1, 1] grid coordinates
    for triplane querying via grid_sample.

    Args:
        points: (..., 3) world-space coordinates (x, y, z)

    Returns:
        xy_coords: (..., 2) normalized coords for XY plane
        xz_coords: (..., 2) normalized coords for XZ plane
        yz_coords: (..., 2) normalized coords for YZ plane
    """
    x, y, z = points[..., 0], points[..., 1], points[..., 2]

    # Normalize x, y to [-1, 1] based on range
    x_norm = x / config.range_xy
    y_norm = y / config.range_xy

    # Normalize z to [-1, 1]: [-3, 45]m -> [-1, 1]
    z_center = (config.range_z_above - config.range_z_below) / 2.0 + config.range_z_below
    z_half = (config.range_z_above + config.range_z_below) / 2.0
    z_norm = (z - z_center) / z_half if z_half > 0 else z * 0

    # Clamp to valid range
    x_norm = x_norm.clamp(-1, 1)
    y_norm = y_norm.clamp(-1, 1)
    z_norm = z_norm.clamp(-1, 1)

    xy_coords = torch.stack([x_norm, y_norm], dim=-1)
    xz_coords = torch.stack([x_norm, z_norm], dim=-1)
    yz_coords = torch.stack([y_norm, z_norm], dim=-1)

    return xy_coords, xz_coords, yz_coords
