"""NuScenes dataset loader for the triplane driving model."""

import json
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from config import TriplaneConfig


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


class NuScenesDataset(Dataset):
    """
    NuScenes dataset loader that produces samples compatible with the triplane model.

    Each sample returns:
        images: (C, 3, H, W) - camera images resized & normalized
        intrinsics: (C, 3, 3) - camera intrinsic matrices (adjusted for resize)
        extrinsics: (C, 4, 4) - world-to-camera transforms
        past_trajectory: (T_past, 2) - past ego (x, y) in current ego frame
        future_trajectory: (T_future, 2) - future ego (x, y) in current ego frame
    """

    CAMERA_CHANNELS = [
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
    ]

    def __init__(self, config: TriplaneConfig, dataroot: str, version: str = 'v1.0-mini'):
        self.config = config
        self.dataroot = dataroot
        self.target_h = config.image_height
        self.target_w = config.image_width

        meta_dir = os.path.join(dataroot, version)

        # Load metadata tables
        self.scene = self._load_json(meta_dir, 'scene.json')
        self.sample = self._load_json(meta_dir, 'sample.json')
        self.sample_data = self._load_json(meta_dir, 'sample_data.json')
        self.calibrated_sensor = self._load_json(meta_dir, 'calibrated_sensor.json')
        self.ego_pose = self._load_json(meta_dir, 'ego_pose.json')
        self.sensor = self._load_json(meta_dir, 'sensor.json')

        # Build lookup tables
        self.sample_data_by_token = {sd['token']: sd for sd in self.sample_data}
        self.calibrated_sensor_by_token = {cs['token']: cs for cs in self.calibrated_sensor}
        self.ego_pose_by_token = {ep['token']: ep for ep in self.ego_pose}
        self.sensor_by_token = {s['token']: s for s in self.sensor}
        self.sample_by_token = {s['token']: s for s in self.sample}

        # For each sample, index the key-frame sample_data by channel
        self._sample_data_by_sample_and_channel = {}
        for sd in self.sample_data:
            if sd['is_key_frame']:
                cs = self.calibrated_sensor_by_token[sd['calibrated_sensor_token']]
                sensor = self.sensor_by_token[cs['sensor_token']]
                channel = sensor['channel']
                self._sample_data_by_sample_and_channel[(sd['sample_token'], channel)] = sd

        # Build ordered sample lists per scene for trajectory extraction
        self._scene_samples = {}
        for sc in self.scene:
            samples = []
            tok = sc['first_sample_token']
            while tok:
                s = self.sample_by_token[tok]
                samples.append(s)
                tok = s['next'] if s['next'] else None
            self._scene_samples[sc['token']] = samples

        # Build valid sample indices: need enough past and future samples
        self.valid_indices = []
        past_needed = config.past_frames  # at 2Hz keyframes, we'll interpolate
        future_needed = config.future_steps
        # We use keyframe-level trajectory (2Hz) and interpolate to 10Hz
        past_kf = max(1, config.past_frames // 5)  # ~5 keyframes for 24 past at 10Hz (2Hz kf)
        future_kf = max(1, config.future_steps // 5)

        for sc in self.scene:
            samples = self._scene_samples[sc['token']]
            for i in range(past_kf, len(samples) - future_kf):
                self.valid_indices.append((sc['token'], i))

        print(f"NuScenesDataset: {len(self.valid_indices)} valid samples from {len(self.scene)} scenes")

    def _load_json(self, meta_dir, filename):
        path = os.path.join(meta_dir, filename)
        with open(path, 'r') as f:
            return json.load(f)

    def _get_ego_pose_for_sample(self, sample_token):
        """Get ego pose (translation, rotation) for a sample via any camera sample_data."""
        sd = self._sample_data_by_sample_and_channel.get((sample_token, 'CAM_FRONT'))
        if sd is None:
            return None
        ep = self.ego_pose_by_token[sd['ego_pose_token']]
        return ep

    def _ego_pose_to_matrix(self, ego_pose):
        """Convert ego pose dict to 4x4 matrix (ego-to-global)."""
        R = quaternion_to_rotation_matrix(ego_pose['rotation'])
        t = np.array(ego_pose['translation'], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def _get_cam_extrinsic(self, calibrated_sensor, ego_pose):
        """
        Compute world-to-camera 4x4 transform.

        NuScenes provides:
          - calibrated_sensor: sensor-to-ego (translation + rotation)
          - ego_pose: ego-to-global (translation + rotation)

        We want: global-to-camera = inv(sensor-to-ego) @ inv(ego-to-global)
        But since our model uses ego frame (not global), we want:
          ego-to-camera = inv(sensor-to-ego)
        """
        # sensor-to-ego transform
        R_se = quaternion_to_rotation_matrix(calibrated_sensor['rotation'])
        t_se = np.array(calibrated_sensor['translation'], dtype=np.float32)
        T_se = np.eye(4, dtype=np.float32)
        T_se[:3, :3] = R_se
        T_se[:3, 3] = t_se

        # ego-to-sensor (camera) = inv(sensor-to-ego)
        T_es = np.linalg.inv(T_se).astype(np.float32)
        return T_es

    def _load_image(self, filename):
        """Load and resize image, return (3, H, W) float32 tensor in [0, 1]."""
        path = os.path.join(self.dataroot, filename)
        img = Image.open(path).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize((self.target_w, self.target_h), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # (3, H, W)
        return arr, orig_w, orig_h

    def _adjust_intrinsics(self, K, orig_w, orig_h):
        """Adjust intrinsics for image resize."""
        K = K.copy()
        scale_x = self.target_w / orig_w
        scale_y = self.target_h / orig_h
        K[0, 0] *= scale_x  # fx
        K[0, 2] *= scale_x  # cx
        K[1, 1] *= scale_y  # fy
        K[1, 2] *= scale_y  # cy
        return K

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        scene_token, sample_idx = self.valid_indices[idx]
        samples = self._scene_samples[scene_token]
        current_sample = samples[sample_idx]

        num_cams = len(self.CAMERA_CHANNELS)

        images = np.zeros((num_cams, 3, self.target_h, self.target_w), dtype=np.float32)
        intrinsics = np.zeros((num_cams, 3, 3), dtype=np.float32)
        extrinsics = np.zeros((num_cams, 4, 4), dtype=np.float32)

        # Load camera data
        for c, channel in enumerate(self.CAMERA_CHANNELS):
            sd = self._sample_data_by_sample_and_channel.get(
                (current_sample['token'], channel)
            )
            if sd is None:
                continue

            # Image
            img, orig_w, orig_h = self._load_image(sd['filename'])
            images[c] = img

            # Calibration
            cs = self.calibrated_sensor_by_token[sd['calibrated_sensor_token']]
            K = np.array(cs['camera_intrinsic'], dtype=np.float32)
            intrinsics[c] = self._adjust_intrinsics(K, orig_w, orig_h)

            # Extrinsic (ego-to-camera)
            ego_pose = self.ego_pose_by_token[sd['ego_pose_token']]
            extrinsics[c] = self._get_cam_extrinsic(cs, ego_pose)

        # Extract trajectory in current ego frame
        current_ego_pose = self._get_ego_pose_for_sample(current_sample['token'])
        T_global_to_ego = np.linalg.inv(
            self._ego_pose_to_matrix(current_ego_pose)
        ).astype(np.float32)

        # Past trajectory: collect from previous keyframes and interpolate
        past_kf = max(1, self.config.past_frames // 5)
        past_positions = []
        for i in range(max(0, sample_idx - past_kf), sample_idx + 1):
            s = samples[i]
            ep = self._get_ego_pose_for_sample(s['token'])
            if ep is not None:
                pos_global = np.array(ep['translation'][:2], dtype=np.float32)
                # Transform to current ego frame
                pos_h = np.array([*ep['translation'], 1.0], dtype=np.float32)
                pos_ego = T_global_to_ego @ pos_h
                past_positions.append(pos_ego[:2])

        # Interpolate past to desired number of frames
        past_traj = self._interpolate_trajectory(past_positions, self.config.past_frames)

        # Future trajectory
        future_kf = max(1, self.config.future_steps // 5)
        future_positions = []
        for i in range(sample_idx, min(len(samples), sample_idx + future_kf + 1)):
            s = samples[i]
            ep = self._get_ego_pose_for_sample(s['token'])
            if ep is not None:
                pos_h = np.array([*ep['translation'], 1.0], dtype=np.float32)
                pos_ego = T_global_to_ego @ pos_h
                future_positions.append(pos_ego[:2])

        future_traj = self._interpolate_trajectory(future_positions, self.config.future_steps)

        return {
            'images': torch.from_numpy(images),
            'intrinsics': torch.from_numpy(intrinsics),
            'extrinsics': torch.from_numpy(extrinsics),
            'past_trajectory': torch.from_numpy(past_traj),
            'future_trajectory': torch.from_numpy(future_traj),
        }

    def _interpolate_trajectory(self, positions, target_length):
        """Interpolate a list of 2D positions to target_length evenly spaced points."""
        if len(positions) < 2:
            # Not enough data, pad with zeros or repeat
            if len(positions) == 0:
                return np.zeros((target_length, 2), dtype=np.float32)
            return np.tile(positions[0], (target_length, 1)).astype(np.float32)

        positions = np.array(positions, dtype=np.float32)
        n = len(positions)
        # Source indices (0 to n-1), target indices mapped to same range
        src_indices = np.arange(n, dtype=np.float32)
        tgt_indices = np.linspace(0, n - 1, target_length)

        result = np.zeros((target_length, 2), dtype=np.float32)
        result[:, 0] = np.interp(tgt_indices, src_indices, positions[:, 0])
        result[:, 1] = np.interp(tgt_indices, src_indices, positions[:, 1])
        return result
