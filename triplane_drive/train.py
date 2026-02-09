"""Training loop for the triplane driving model."""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TriplaneConfig
from model import TriplaneDriveModel
from losses import CombinedLoss
from dummy_data import DummyDrivingDataset
from nuscenes_data import NuScenesDataset


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, phase='joint'):
    """Run one training epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} ({phase})")
    for batch in pbar:
        # Move to device
        images = batch['images'].to(device)
        intrinsics = batch['intrinsics'].to(device)
        extrinsics = batch['extrinsics'].to(device)
        past_traj = batch['past_trajectory'].to(device)
        future_traj = batch['future_trajectory'].to(device)

        # Forward
        outputs = model(images, intrinsics, extrinsics, past_traj, future_traj)

        # Loss
        loss, loss_dict = loss_fn(outputs, batch={
            'images': images,
        })

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.config.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        loss_str = " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
        pbar.set_postfix_str(loss_str)

    return total_loss / max(num_batches, 1)


def train(config: TriplaneConfig = None, epochs: int = None, batch_size: int = None,
          num_samples: int = 50, device_str: str = None, nuscenes_dataroot: str = None):
    """
    Main training function.

    Args:
        config: TriplaneConfig (uses defaults if None)
        epochs: override config epochs
        batch_size: override config batch size
        num_samples: number of dummy data samples (ignored if nuscenes_dataroot is set)
        device_str: 'cuda' or 'cpu'
        nuscenes_dataroot: path to nuscenes data root (e.g. 'data/nuscenes')
    """
    if config is None:
        config = TriplaneConfig()

    if epochs is not None:
        config.num_epochs = epochs
    if batch_size is not None:
        config.batch_size = batch_size

    # Device
    if device_str is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")
    print(f"Config: {config.num_epochs} epochs, batch_size={config.batch_size}")
    print(f"Sensor tokens: {config.total_sensor_tokens}, "
          f"Past traj tokens: {config.total_past_traj_tokens}, "
          f"Future traj tokens: {config.total_future_traj_tokens}")

    # Dataset and dataloader
    if nuscenes_dataroot is not None:
        print(f"Using NuScenes data from: {nuscenes_dataroot}")
        dataset = NuScenesDataset(config, dataroot=nuscenes_dataroot)
    else:
        dataset = DummyDrivingDataset(config, num_samples=num_samples)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=0, drop_last=True
    )

    # Model
    model = TriplaneDriveModel(config).to(device)

    # Phase 1: freeze trajectory modules (AR transformer, trajectory head, trajectory embeddings)
    if config.training_phase == 'phase1':
        print("Phase 1 training: volumetric rendering only (freezing trajectory modules)")
        for param in model.ar_transformer.parameters():
            param.requires_grad = False
        for param in model.traj_head.parameters():
            param.requires_grad = False
        for param in model.traj_embed.parameters():
            param.requires_grad = False

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss
    loss_fn = CombinedLoss(config)

    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs * len(dataloader),
    )

    # Training loop
    phase_label = config.training_phase
    for epoch in range(1, config.num_epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, phase=phase_label)
        scheduler.step()
        print(f"Epoch {epoch}/{config.num_epochs} - Avg loss: {avg_loss:.4f}")

    # Save model
    save_path = f'triplane_model_{phase_label}.pt' if phase_label != 'joint' else 'triplane_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_phase': phase_label,
    }, save_path)
    print(f"Model saved to {save_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train triplane driving model')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of dummy samples')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--no_render', action='store_true', help='Disable volumetric rendering')
    parser.add_argument('--small', action='store_true', help='Use smaller config for quick testing')
    parser.add_argument('--nuscenes', type=str, default=None,
                        help='Path to NuScenes data root (e.g. data/nuscenes)')
    parser.add_argument('--phase', type=str, default='joint',
                        choices=['phase1', 'phase2', 'joint'],
                        help='Training phase: phase1 (render only), phase2 (traj only), joint')
    parser.add_argument('--dino', action='store_true',
                        help='Use pretrained DINOv2 backbone (requires timm or torch.hub)')
    args = parser.parse_args()

    config = TriplaneConfig()
    config.training_phase = args.phase

    # NuScenes has 6 cameras, not 7
    if args.nuscenes:
        config.num_cameras = 6

    # Phase 1: force rendering on, render more cameras for better signal
    if args.phase == 'phase1':
        config.use_volumetric_rendering = True
        config.num_render_cameras = config.num_cameras  # render all cameras

    if args.dino:
        config.use_pretrained_dino = True

    if args.no_render:
        config.use_volumetric_rendering = False
    if args.small:
        # Smaller config for quick smoke testing
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
        config.num_ray_samples = 16
        config.num_render_cameras = 1
        config.feature_dim = 64
        config.dino_embed_dim = 64
        config.num_heads_lifting = 2

    train(config, epochs=args.epochs, batch_size=args.batch_size,
          num_samples=args.num_samples, device_str=args.device,
          nuscenes_dataroot=args.nuscenes)


if __name__ == '__main__':
    main()
