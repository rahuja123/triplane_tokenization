"""Training loop for the triplane driving model.

Optimizations:
  - Mixed precision (AMP) with --amp flag
  - Gradient checkpointing for renderer with --grad_ckpt flag
  - Configurable dataloader workers (default 4 on CUDA)
"""

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import TriplaneConfig
from model import TriplaneDriveModel
from losses import CombinedLoss
from dummy_data import DummyDrivingDataset
from nuscenes_data import NuScenesDataset


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, phase='joint',
                grad_clip=1.0, use_amp=False, scaler=None):
    """Run one training epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} ({phase})")
    for batch in pbar:
        # Move to device (non_blocking for pinned memory)
        images = batch['images'].to(device, non_blocking=True)
        intrinsics = batch['intrinsics'].to(device, non_blocking=True)
        extrinsics = batch['extrinsics'].to(device, non_blocking=True)
        past_traj = batch['past_trajectory'].to(device, non_blocking=True)
        future_traj = batch['future_trajectory'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        # Forward + loss with optional AMP
        if use_amp and device.type == 'cuda':
            with autocast():
                outputs = model(images, intrinsics, extrinsics, past_traj, future_traj)
                loss, loss_dict = loss_fn(outputs, batch={'images': images})
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, intrinsics, extrinsics, past_traj, future_traj)
            loss, loss_dict = loss_fn(outputs, batch={'images': images})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        loss_str = " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
        pbar.set_postfix_str(loss_str)

    return total_loss / max(num_batches, 1)


def train(config: TriplaneConfig = None, epochs: int = None, batch_size: int = None,
          num_samples: int = 50, device_str: str = None, nuscenes_dataroot: str = None,
          gpu_ids: list = None, num_workers: int = None, use_amp: bool = False,
          use_grad_ckpt: bool = False):
    """
    Main training function.

    Args:
        config: TriplaneConfig (uses defaults if None)
        epochs: override config epochs
        batch_size: override config batch size
        num_samples: number of dummy data samples (ignored if nuscenes_dataroot is set)
        device_str: 'cuda', 'cuda:0', 'cuda:2', 'cpu', etc.
        nuscenes_dataroot: path to nuscenes data root (e.g. 'data/nuscenes')
        gpu_ids: list of GPU ids for DataParallel (e.g. [0,1,2,3]). Overrides device_str.
        num_workers: number of dataloader workers (default: 4 for CUDA, 0 for CPU)
        use_amp: enable mixed precision training (CUDA only)
        use_grad_ckpt: enable gradient checkpointing for renderer
    """
    if config is None:
        config = TriplaneConfig()

    if epochs is not None:
        config.num_epochs = epochs
    if batch_size is not None:
        config.batch_size = batch_size

    # Device setup
    if gpu_ids is not None and len(gpu_ids) > 0:
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
    elif device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Default num_workers
    if num_workers is None:
        num_workers = 4 if device.type == 'cuda' else 0

    # AMP only on CUDA
    if use_amp and device.type != 'cuda':
        print("Warning: AMP only works with CUDA, disabling")
        use_amp = False

    scaler = GradScaler() if use_amp else None

    print(f"Primary device: {device}")
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    print(f"Config: {config.num_epochs} epochs, batch_size={config.batch_size}")
    print(f"AMP: {use_amp}, Grad checkpointing: {use_grad_ckpt}, Workers: {num_workers}")
    print(f"Sensor tokens: {config.total_sensor_tokens}, "
          f"Past traj tokens: {config.total_past_traj_tokens}, "
          f"Future traj tokens: {config.total_future_traj_tokens}")

    # Dataset and dataloader
    if nuscenes_dataroot is not None:
        print(f"Using NuScenes data from: {nuscenes_dataroot}")
        dataset = NuScenesDataset(config, dataroot=nuscenes_dataroot)
    else:
        dataset = DummyDrivingDataset(config, num_samples=num_samples)

    dl_kwargs = dict(
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=(device.type == 'cuda'),
    )
    if num_workers > 0:
        dl_kwargs['persistent_workers'] = True
        dl_kwargs['prefetch_factor'] = 2
    dataloader = DataLoader(dataset, **dl_kwargs)

    # Model
    model = TriplaneDriveModel(config).to(device)

    # Enable gradient checkpointing on renderer
    if use_grad_ckpt and model.renderer is not None:
        model.use_renderer_checkpointing = True
        print("Gradient checkpointing enabled for renderer")

    # Phase 1: freeze trajectory modules
    if config.training_phase == 'phase1':
        print("Phase 1 training: volumetric rendering only (freezing trajectory modules)")
        for param in model.ar_transformer.parameters():
            param.requires_grad = False
        for param in model.traj_head.parameters():
            param.requires_grad = False
        for param in model.traj_embed.parameters():
            param.requires_grad = False

    # Wrap in DataParallel for multi-GPU
    if gpu_ids is not None and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"Model wrapped in DataParallel across GPUs {gpu_ids}")

    # Count parameters
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
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
    grad_clip = config.grad_clip
    for epoch in range(1, config.num_epochs + 1):
        t0 = time.time()
        avg_loss = train_epoch(model, dataloader, optimizer, loss_fn, device, epoch,
                               phase=phase_label, grad_clip=grad_clip,
                               use_amp=use_amp, scaler=scaler)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{config.num_epochs} - Avg loss: {avg_loss:.4f} - Time: {elapsed:.1f}s")

        if torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
            print(f"  Peak GPU memory: {mem_mb:.0f} MB")

    # Save model (unwrap DataParallel if needed)
    save_model = model.module if isinstance(model, nn.DataParallel) else model
    save_path = f'triplane_model_{phase_label}.pt' if phase_label != 'joint' else 'triplane_model.pt'
    torch.save({
        'model_state_dict': save_model.state_dict(),
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
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu/cuda:0/cuda:2)')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU ids for multi-GPU training (e.g. 0,1,2,3)')
    parser.add_argument('--num_workers', type=int, default=None, help='Dataloader workers')
    parser.add_argument('--no_render', action='store_true', help='Disable volumetric rendering')
    parser.add_argument('--small', action='store_true', help='Use smaller config for quick testing')
    parser.add_argument('--nuscenes', type=str, default=None,
                        help='Path to NuScenes data root (e.g. data/nuscenes)')
    parser.add_argument('--phase', type=str, default='joint',
                        choices=['phase1', 'phase2', 'joint'],
                        help='Training phase: phase1 (render only), phase2 (traj only), joint')
    parser.add_argument('--dino', action='store_true',
                        help='Use pretrained DINOv2 backbone (requires timm or torch.hub)')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision training (CUDA only)')
    parser.add_argument('--grad_ckpt', action='store_true',
                        help='Enable gradient checkpointing for renderer (saves memory)')
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

    # Parse GPU ids
    gpu_ids = None
    if args.gpus is not None:
        gpu_ids = [int(g) for g in args.gpus.split(',')]

    train(config, epochs=args.epochs, batch_size=args.batch_size,
          num_samples=args.num_samples, device_str=args.device,
          nuscenes_dataroot=args.nuscenes, gpu_ids=gpu_ids,
          num_workers=args.num_workers, use_amp=args.amp,
          use_grad_ckpt=args.grad_ckpt)


if __name__ == '__main__':
    main()
