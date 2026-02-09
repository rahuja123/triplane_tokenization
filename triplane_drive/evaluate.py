"""Evaluation script: compute minADE6 on dummy data."""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import TriplaneConfig
from model import TriplaneDriveModel
from dummy_data import DummyDrivingDataset


def compute_min_ade(pred_trajectories, gt_trajectory, horizons_steps=None):
    """
    Compute minADE_k metric.

    Args:
        pred_trajectories: (B, K, T, 2) K sampled trajectories
        gt_trajectory: (B, T, 2) ground truth trajectory
        horizons_steps: list of timesteps to evaluate at (e.g., [10, 30, 50] for 1s, 3s, 5s)

    Returns:
        metrics: dict of minADE values at each horizon
    """
    B, K, T, _ = pred_trajectories.shape

    # Expand gt for comparison: (B, 1, T, 2)
    gt_expanded = gt_trajectory.unsqueeze(1)

    # Per-timestep displacement: (B, K, T)
    displacement = torch.norm(pred_trajectories - gt_expanded, dim=-1)

    metrics = {}

    if horizons_steps is None:
        horizons_steps = [T]

    for h in horizons_steps:
        h = min(h, T)
        # ADE up to horizon h: (B, K)
        ade = displacement[:, :, :h].mean(dim=-1)
        # Min over K samples: (B,)
        min_ade = ade.min(dim=-1).values
        # Average over batch
        avg_min_ade = min_ade.mean().item()

        horizon_sec = h / 10.0  # assuming 10Hz
        metrics[f'minADE6@{horizon_sec:.0f}s'] = avg_min_ade

    return metrics


def evaluate(config: TriplaneConfig = None, model_path: str = None,
             num_samples: int = 20, device_str: str = None):
    """
    Evaluate the triplane driving model.

    Args:
        config: TriplaneConfig
        model_path: path to saved model checkpoint
        num_samples: number of evaluation samples
        device_str: 'cuda' or 'cpu'
    """
    if config is None:
        config = TriplaneConfig()

    if device_str is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")

    # Model
    model = TriplaneDriveModel(config).to(device)

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        print("No model checkpoint provided, using random weights")

    model.eval()

    # Dataset
    dataset = DummyDrivingDataset(config, num_samples=num_samples, seed=123)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    all_metrics = {}
    count = 0

    horizons = [10, 30, 50]  # 1s, 3s, 5s at 10Hz

    for batch in dataloader:
        images = batch['images'].to(device)
        intrinsics = batch['intrinsics'].to(device)
        extrinsics = batch['extrinsics'].to(device)
        past_traj = batch['past_trajectory'].to(device)
        future_traj = batch['future_trajectory'].to(device)

        # Generate trajectory samples
        pred_trajs = model.generate_trajectory(
            images, intrinsics, extrinsics, past_traj,
            num_samples=config.num_trajectory_samples,
            temperature=1.0,
        )

        # Compute metrics
        metrics = compute_min_ade(pred_trajs, future_traj, horizons)

        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v
        count += 1

    # Average
    print("\n--- Evaluation Results ---")
    for k, v in all_metrics.items():
        avg = v / max(count, 1)
        print(f"  {k}: {avg:.4f} m")


def main():
    parser = argparse.ArgumentParser(description='Evaluate triplane driving model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of eval samples')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--small', action='store_true', help='Use smaller config')
    args = parser.parse_args()

    config = TriplaneConfig()
    if args.small:
        config.sx = 24
        config.sy = 24
        config.sz = 12
        config.inner_cells_xy = 12
        config.inner_cells_z_lower = 9
        config.outer_cells_z_upper = 3
        config.patch_x = 4
        config.patch_y = 6
        config.patch_z = 6
        config.d_ar = 128
        config.ar_num_layers = 2
        config.ar_num_heads = 4
        config.ar_ffn_dim = 256
        config.feature_dim = 64
        config.dino_embed_dim = 64
        config.num_heads_lifting = 2

    evaluate(config, model_path=args.model_path,
             num_samples=args.num_samples, device_str=args.device)


if __name__ == '__main__':
    main()
