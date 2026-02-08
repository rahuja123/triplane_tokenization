"""Sinusoidal positional encoding for 3D coordinates."""

import torch
import math


def sinusoidal_positional_encoding_3d(coords, dim):
    """
    Generate sinusoidal positional encoding for 3D coordinates.

    Args:
        coords: (..., 3) tensor of (x, y, z) coordinates
        dim: total encoding dimension (should be divisible by 6 for clean split)

    Returns:
        encoding: (..., dim) positional encoding
    """
    assert dim % 6 == 0 or dim % 2 == 0, "dim should be divisible by 2"

    # Split dimension across 3 axes
    dim_per_axis = dim // 3
    # Make even
    dim_per_axis = dim_per_axis - (dim_per_axis % 2)
    remainder = dim - 3 * dim_per_axis

    device = coords.device
    dtype = coords.dtype

    encodings = []
    for i in range(3):
        d = dim_per_axis + (remainder if i == 2 else 0)
        half_d = d // 2

        # Log-spaced frequencies
        freq = torch.exp(
            torch.arange(half_d, device=device, dtype=dtype)
            * -(math.log(10000.0) / half_d)
        )

        # Scale coordinates and compute sin/cos
        scaled = coords[..., i:i+1] * freq  # (..., half_d)
        enc = torch.cat([scaled.sin(), scaled.cos()], dim=-1)  # (..., d)
        encodings.append(enc)

    return torch.cat(encodings, dim=-1)  # (..., dim)


def sinusoidal_positional_encoding_1d(positions, dim):
    """
    Standard 1D sinusoidal positional encoding.

    Args:
        positions: (...,) tensor of positions
        dim: encoding dimension

    Returns:
        encoding: (..., dim)
    """
    half_dim = dim // 2
    freq = torch.exp(
        torch.arange(half_dim, device=positions.device, dtype=positions.dtype)
        * -(math.log(10000.0) / half_dim)
    )
    scaled = positions.unsqueeze(-1) * freq
    return torch.cat([scaled.sin(), scaled.cos()], dim=-1)
