"""Configuration dataclass for the triplane driving model."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TriplaneConfig:
    # Image
    num_cameras: int = 7
    image_height: int = 320
    image_width: int = 512

    # DINOv2
    dino_model: str = "dinov2_vits14"
    dino_embed_dim: int = 384
    dino_patch_size: int = 14
    feature_dim: int = 192  # Df after projection

    # Triplane spatial dimensions
    sx: int = 96
    sy: int = 96
    sz: int = 48

    # Spatial coverage (meters)
    range_xy: float = 180.0
    range_z_above: float = 45.0
    range_z_below: float = 3.0

    # Nonlinear grid cells
    inner_cells_xy: int = 36
    inner_cells_z_lower: int = 36  # [-3, 15]m
    outer_cells_z_upper: int = 12  # (15, 45]m

    # 3D Lifting
    num_lifting_layers: int = 4  # 2 per-image + 2 cross-image
    num_deform_points: int = 4
    num_heads_lifting: int = 6

    # Patchification
    patch_x: int = 4
    patch_y: int = 6
    patch_z: int = 6
    use_halfplane: bool = False

    # AR Transformer (scaled down for demo)
    d_ar: int = 512
    ar_num_layers: int = 8
    ar_num_heads: int = 8
    ar_ffn_dim: int = 2048
    ar_dropout: float = 0.1

    # Trajectory
    past_frames: int = 24
    future_steps: int = 50  # 5s at 10Hz
    trajectory_dim: int = 2  # (x, y)
    num_trajectory_samples: int = 6
    traj_vocab_size: int = 1000
    traj_range: float = 180.0  # meters, for discretization

    # Volumetric rendering
    num_ray_samples: int = 64
    render_image_scale: float = 0.25
    render_near: float = 0.5
    render_far: float = 200.0

    # Loss weights
    lambda_lpips: float = 0.5
    lambda_l1: float = 0.5
    lambda_depth: float = 0.1

    # Training
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 2
    num_epochs: int = 10
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Flags
    use_pretrained_dino: bool = False  # Set True if you have internet
    use_volumetric_rendering: bool = True
    num_render_cameras: int = 2  # cameras to render per batch (for efficiency)

    @property
    def image_feat_h(self) -> int:
        """Height of DINOv2 feature map."""
        import math
        return math.ceil(self.image_height / self.dino_patch_size)

    @property
    def image_feat_w(self) -> int:
        """Width of DINOv2 feature map."""
        import math
        return math.ceil(self.image_width / self.dino_patch_size)

    @property
    def padded_image_h(self) -> int:
        return self.image_feat_h * self.dino_patch_size

    @property
    def padded_image_w(self) -> int:
        return self.image_feat_w * self.dino_patch_size

    @property
    def num_tokens_xy(self) -> int:
        return (self.sx // self.patch_x) * (self.sy // self.patch_y)

    @property
    def num_tokens_xz(self) -> int:
        return (self.sx // self.patch_x) * (self.sz // self.patch_z)

    @property
    def num_tokens_yz(self) -> int:
        return (self.sy // self.patch_y) * (self.sz // self.patch_z)

    @property
    def total_sensor_tokens(self) -> int:
        t = self.num_tokens_xy + self.num_tokens_xz + self.num_tokens_yz
        if self.use_halfplane:
            # Halve xy and xz planes
            t = self.num_tokens_xy // 2 + self.num_tokens_xz // 2 + self.num_tokens_yz
        return t

    @property
    def total_past_traj_tokens(self) -> int:
        return self.past_frames * 2  # interleaved x, y

    @property
    def total_future_traj_tokens(self) -> int:
        return self.future_steps * 2  # interleaved x, y

    @property
    def total_sequence_length(self) -> int:
        return self.total_sensor_tokens + self.total_past_traj_tokens + self.total_future_traj_tokens
