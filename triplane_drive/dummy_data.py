"""Synthetic driving dataset for testing the triplane model."""

import torch
import numpy as np
from torch.utils.data import Dataset
import math

from .config import TriplaneConfig


def generate_camera_params(config: TriplaneConfig):
    """
    Generate realistic surround-view camera intrinsics and extrinsics.

    World frame: X=forward, Y=left, Z=up (standard automotive / ego frame).
    Camera frame: X=right, Y=down, Z=forward (standard pinhole camera).

    Returns:
        intrinsics: (C, 3, 3) numpy array
        extrinsics: (C, 4, 4) numpy array (world-to-camera transforms)
    """
    C = config.num_cameras
    H, W = config.image_height, config.image_width

    # Intrinsics: same for all cameras
    fx = fy = 400.0
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    intrinsics = np.stack([K] * C, axis=0)

    # Base rotation: world (X-fwd, Y-left, Z-up) -> camera (X-right, Y-down, Z-fwd)
    # cam_X = -world_Y, cam_Y = -world_Z, cam_Z = world_X
    R_base = np.array([
        [0, -1,  0],   # cam X = -world Y
        [0,  0, -1],   # cam Y = -world Z
        [1,  0,  0],   # cam Z =  world X
    ], dtype=np.float32)

    def _rotation_z(angle_rad):
        """Rotation around world Z (up) axis for yaw."""
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    # 7-camera surround-view
    # (camera position in world frame, yaw angle in degrees)
    camera_configs = [
        (np.array([2.0,  0.0, 1.5]),    0),   # Front
        (np.array([2.0,  1.0, 1.5]),   30),   # Front-left
        (np.array([2.0, -1.0, 1.5]),  -30),   # Front-right
        (np.array([0.0,  1.5, 1.5]),   90),   # Left
        (np.array([0.0, -1.5, 1.5]),  -90),   # Right
        (np.array([-2.0, 1.0, 1.5]),  150),   # Rear-left
        (np.array([-2.0, 0.0, 1.5]),  180),   # Rear
    ]

    extrinsics = np.zeros((C, 4, 4), dtype=np.float32)
    for i, (pos_world, yaw_deg) in enumerate(camera_configs):
        # Yaw rotation in world frame, then world-to-camera base transform
        R_yaw = _rotation_z(math.radians(yaw_deg))
        # The camera looks in direction rotated by yaw from +X
        # World-to-camera rotation: R_base applied after inverse yaw
        R_yaw_inv = R_yaw.T  # rotate world points to camera-facing frame
        R_w2c = R_base @ R_yaw_inv

        # Translation: t = -R @ position_world
        t = -R_w2c @ pos_world

        extrinsics[i, :3, :3] = R_w2c
        extrinsics[i, :3, 3] = t
        extrinsics[i, 3, 3] = 1.0

    return intrinsics, extrinsics


def generate_trajectory(num_past, num_future, speed=10.0, curvature=0.01, dt=0.1):
    """
    Generate a smooth ego trajectory using a simple kinematic model.

    Args:
        num_past: number of past frames
        num_future: number of future frames
        speed: forward speed in m/s
        curvature: turning rate in rad/s
        dt: time step in seconds

    Returns:
        past_traj: (num_past, 2) past (x, y) positions
        future_traj: (num_future, 2) future (x, y) positions
    """
    total = num_past + num_future
    x, y, theta = 0.0, 0.0, 0.0
    positions = []

    # Add random curvature variation
    curvature_var = curvature + np.random.uniform(-0.005, 0.005)

    for _ in range(total):
        x += speed * dt * math.cos(theta)
        y += speed * dt * math.sin(theta)
        theta += curvature_var * dt
        positions.append([x, y])

    positions = np.array(positions, dtype=np.float32)

    # Shift so that current position (boundary between past/future) is at origin
    origin = positions[num_past - 1]
    positions -= origin

    past_traj = positions[:num_past]
    future_traj = positions[num_past:]

    return past_traj, future_traj


class DummyDrivingDataset(Dataset):
    """
    Generates synthetic driving data for testing.

    Each sample contains:
        images: (C, 3, H, W) random images with mild structure
        intrinsics: (C, 3, 3) camera intrinsic matrices
        extrinsics: (C, 4, 4) world-to-camera transforms
        past_trajectory: (num_past, 2) past ego waypoints
        future_trajectory: (num_future, 2) ground truth future waypoints
    """

    def __init__(self, config: TriplaneConfig, num_samples: int = 100, seed: int = 42):
        self.config = config
        self.num_samples = num_samples
        self.rng = np.random.RandomState(seed)

        # Generate fixed camera params (shared across all samples)
        self.intrinsics, self.extrinsics = generate_camera_params(config)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        config = self.config

        # Generate images with mild structure (gradients + noise)
        images = np.zeros((config.num_cameras, 3, config.image_height, config.image_width), dtype=np.float32)
        for c in range(config.num_cameras):
            # Horizontal gradient + noise
            grad_h = np.linspace(0.2, 0.8, config.image_width, dtype=np.float32)
            grad_v = np.linspace(0.3, 0.7, config.image_height, dtype=np.float32)
            base = grad_v[:, None] * grad_h[None, :]
            for ch in range(3):
                noise = self.rng.randn(config.image_height, config.image_width).astype(np.float32) * 0.1
                images[c, ch] = np.clip(base + noise + (c * 0.05) + (ch * 0.1), 0, 1)

        # Generate trajectory
        past_traj, future_traj = generate_trajectory(
            config.past_frames, config.future_steps,
            speed=np.random.uniform(5, 15),
            curvature=np.random.uniform(-0.02, 0.02),
        )

        return {
            'images': torch.from_numpy(images),
            'intrinsics': torch.from_numpy(self.intrinsics.copy()),
            'extrinsics': torch.from_numpy(self.extrinsics.copy()),
            'past_trajectory': torch.from_numpy(past_traj),
            'future_trajectory': torch.from_numpy(future_traj),
        }
