"""Heatmap visualization for triplane driving model intermediate representations."""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import math

from .config import TriplaneConfig
from .model import TriplaneDriveModel
from .dummy_data import DummyDrivingDataset

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _to_numpy(t):
    """Convert tensor to numpy, handling detach and cpu."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return np.array(t)


def _normalize(arr):
    """Normalize array to [0, 1] for display."""
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def visualize_triplane_heatmaps(pxy, pxz, pyz, save_path='heatmaps_triplane.png', title_prefix=''):
    """
    Visualize triplane feature planes as heatmaps.

    Args:
        pxy: (Sx, Sy, Df) or (B, Sx, Sy, Df) BEV plane features
        pxz: (Sx, Sz, Df) or (B, Sx, Sz, Df) front plane features
        pyz: (Sy, Sz, Df) or (B, Sy, Sz, Df) side plane features
        save_path: output file path
        title_prefix: prefix for subplot titles
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization")
        return

    pxy = _to_numpy(pxy)
    pxz = _to_numpy(pxz)
    pyz = _to_numpy(pyz)

    # Take first batch element if batched
    if pxy.ndim == 4:
        pxy, pxz, pyz = pxy[0], pxz[0], pyz[0]

    # Compute different heatmap types
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'{title_prefix}Triplane Feature Heatmaps', fontsize=16, fontweight='bold')

    planes = [
        ('XY (BEV)', pxy, 'X (forward)', 'Y (left)'),
        ('XZ (Front)', pxz, 'X (forward)', 'Z (up)'),
        ('YZ (Side)', pyz, 'Y (left)', 'Z (up)'),
    ]

    for row, (name, plane, xlabel, ylabel) in enumerate(planes):
        # L2 norm across feature dimension
        l2_norm = np.linalg.norm(plane, axis=-1)
        im0 = axes[row, 0].imshow(_normalize(l2_norm).T, cmap='viridis', aspect='auto', origin='lower')
        axes[row, 0].set_title(f'{name} - L2 Norm')
        axes[row, 0].set_xlabel(xlabel)
        axes[row, 0].set_ylabel(ylabel)
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        # Max activation across feature dimension
        max_act = plane.max(axis=-1)
        im1 = axes[row, 1].imshow(_normalize(max_act).T, cmap='hot', aspect='auto', origin='lower')
        axes[row, 1].set_title(f'{name} - Max Activation')
        axes[row, 1].set_xlabel(xlabel)
        axes[row, 1].set_ylabel(ylabel)
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        # Variance across feature dimension (feature diversity)
        variance = plane.var(axis=-1)
        im2 = axes[row, 2].imshow(_normalize(variance).T, cmap='plasma', aspect='auto', origin='lower')
        axes[row, 2].set_title(f'{name} - Feature Variance')
        axes[row, 2].set_xlabel(xlabel)
        axes[row, 2].set_ylabel(ylabel)
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Triplane heatmaps saved to {save_path}")


def visualize_encoder_features(features, images, save_path='heatmaps_encoder.png'):
    """
    Visualize image encoder feature maps overlaid on input images.

    Args:
        features: (B, C, Hf, Wf, Df) encoder output
        images: (B, C, 3, H, W) input images
        save_path: output file path
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization")
        return

    features = _to_numpy(features[0])  # first batch element: (C, Hf, Wf, Df)
    images = _to_numpy(images[0])      # (C, 3, H, W)

    C = features.shape[0]
    cam_names = ['Front', 'Front-L', 'Front-R', 'Left', 'Right', 'Rear-L', 'Rear'][:C]

    fig, axes = plt.subplots(2, C, figsize=(4 * C, 8))
    fig.suptitle('Image Encoder: Feature Map Heatmaps', fontsize=14, fontweight='bold')

    if C == 1:
        axes = axes.reshape(2, 1)

    for c in range(C):
        # Original image
        img = np.transpose(images[c], (1, 2, 0))  # (H, W, 3)
        img = np.clip(img, 0, 1)
        axes[0, c].imshow(img)
        axes[0, c].set_title(f'{cam_names[c]} - Input')
        axes[0, c].axis('off')

        # Feature heatmap (L2 norm)
        feat_map = np.linalg.norm(features[c], axis=-1)  # (Hf, Wf)
        feat_map = _normalize(feat_map)

        # Resize to image dimensions for overlay
        H, W = images[c].shape[1], images[c].shape[2]
        feat_resized = np.array(
            torch.nn.functional.interpolate(
                torch.tensor(feat_map).unsqueeze(0).unsqueeze(0).float(),
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze().numpy()
        )

        axes[1, c].imshow(img)
        axes[1, c].imshow(feat_resized, cmap='jet', alpha=0.5)
        axes[1, c].set_title(f'{cam_names[c]} - Features')
        axes[1, c].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Encoder feature heatmaps saved to {save_path}")


def visualize_attention_weights(model, features, intrinsics, extrinsics,
                                save_path='heatmaps_attention.png'):
    """
    Visualize camera attention weights from the lifting module.
    Shows which cameras contribute most to each triplane location.

    Args:
        model: TriplaneDriveModel
        features: (B, C, Hf, Wf, Df) image features
        intrinsics: (B, C, 3, 3)
        extrinsics: (B, C, 4, 4)
        save_path: output file path
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization")
        return

    config = model.config
    lifting = model.lifting
    B = features.shape[0]
    C = features.shape[1]
    Sx, Sy, Sz = config.sx, config.sy, config.sz
    D = config.feature_dim
    Hf, Wf = features.shape[2], features.shape[3]

    from .utils.geometry import project_3d_to_2d
    from .utils.positional_encoding import sinusoidal_positional_encoding_3d

    # Compute visibility masks for each camera
    grid_flat = lifting.grid_3d.reshape(-1, 3)
    N = grid_flat.shape[0]
    grid_batch = grid_flat.unsqueeze(0).expand(B, -1, -1)

    pixel_coords, valid_mask = project_3d_to_2d(
        grid_batch, intrinsics, extrinsics,
        image_h=config.image_height, image_w=config.image_width
    )
    # valid_mask: (B, C, N)

    # Compute camera visibility per voxel
    vis = _to_numpy(valid_mask[0])  # (C, N)
    vis = vis.reshape(C, Sx, Sy, Sz)

    cam_names = ['Front', 'Front-L', 'Front-R', 'Left', 'Right', 'Rear-L', 'Rear'][:C]

    # Visualize BEV (average over z) and Front view (average over y)
    fig, axes = plt.subplots(2, C, figsize=(3.5 * C, 7))
    fig.suptitle('Camera Visibility / Attention per Triplane Location', fontsize=14, fontweight='bold')

    if C == 1:
        axes = axes.reshape(2, 1)

    for c in range(C):
        # BEV visibility
        bev_vis = vis[c].mean(axis=2)  # (Sx, Sy) average over z
        im = axes[0, c].imshow(bev_vis.T, cmap='Blues', aspect='auto', origin='lower',
                               vmin=0, vmax=1)
        axes[0, c].set_title(f'{cam_names[c]} BEV')
        axes[0, c].set_xlabel('X')
        axes[0, c].set_ylabel('Y')

        # Front visibility
        front_vis = vis[c].mean(axis=1)  # (Sx, Sz) average over y
        axes[1, c].imshow(front_vis.T, cmap='Blues', aspect='auto', origin='lower',
                          vmin=0, vmax=1)
        axes[1, c].set_title(f'{cam_names[c]} Front')
        axes[1, c].set_xlabel('X')
        axes[1, c].set_ylabel('Z')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention/visibility heatmaps saved to {save_path}")


def visualize_rendered_vs_input(rendered_images, input_images, camera_indices,
                                save_path='heatmaps_rendering.png'):
    """
    Visualize rendered images vs input images with error heatmaps.

    Args:
        rendered_images: (B, C_render, 3, H_r, W_r) rendered RGB
        input_images: (B, C_total, 3, H, W) input images
        camera_indices: list of rendered camera indices
        save_path: output file path
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization")
        return

    rendered = _to_numpy(rendered_images[0])  # (C_render, 3, H_r, W_r)
    inputs = _to_numpy(input_images[0])       # (C_total, 3, H, W)

    C_render = rendered.shape[0]
    cam_names = ['Front', 'Front-L', 'Front-R', 'Left', 'Right', 'Rear-L', 'Rear']

    fig, axes = plt.subplots(3, C_render, figsize=(5 * C_render, 12))
    fig.suptitle('Volumetric Rendering: Input vs Rendered vs Error', fontsize=14, fontweight='bold')

    if C_render == 1:
        axes = axes.reshape(3, 1)

    for i, cam_idx in enumerate(camera_indices):
        # Input image (resized to render resolution)
        input_img = np.transpose(inputs[cam_idx], (1, 2, 0))
        input_img = np.clip(input_img, 0, 1)

        H_r, W_r = rendered.shape[2], rendered.shape[3]
        input_resized = np.array(
            F.interpolate(
                torch.tensor(inputs[cam_idx]).unsqueeze(0),
                size=(H_r, W_r), mode='bilinear', align_corners=False
            ).squeeze().permute(1, 2, 0).numpy()
        )
        input_resized = np.clip(input_resized, 0, 1)

        # Rendered image
        render_img = np.transpose(rendered[i], (1, 2, 0))
        render_img = np.clip(render_img, 0, 1)

        # Error heatmap
        error = np.abs(input_resized - render_img).mean(axis=-1)

        axes[0, i].imshow(input_resized)
        axes[0, i].set_title(f'{cam_names[cam_idx]} - Input')
        axes[0, i].axis('off')

        axes[1, i].imshow(render_img)
        axes[1, i].set_title(f'{cam_names[cam_idx]} - Rendered')
        axes[1, i].axis('off')

        im = axes[2, i].imshow(error, cmap='hot', vmin=0)
        axes[2, i].set_title(f'{cam_names[cam_idx]} - Error')
        axes[2, i].axis('off')
        plt.colorbar(im, ax=axes[2, i], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Rendering comparison saved to {save_path}")


def visualize_token_norms(sensor_tokens, token_counts, save_path='heatmaps_tokens.png'):
    """
    Visualize token embedding norms arranged spatially per plane.

    Args:
        sensor_tokens: (B, L, d_ar) token embeddings
        token_counts: tuple (Lxy, Lxz, Lyz)
        save_path: output file path
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization")
        return

    tokens = _to_numpy(sensor_tokens[0])  # (L, d_ar)
    Lxy, Lxz, Lyz = token_counts

    # Compute norms
    norms = np.linalg.norm(tokens, axis=-1)  # (L,)

    # Split by plane
    xy_norms = norms[:Lxy]
    xz_norms = norms[Lxy:Lxy + Lxz]
    yz_norms = norms[Lxy + Lxz:]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Token Embedding Norms per Triplane', fontsize=14, fontweight='bold')

    # Reshape to spatial grids
    # XY plane
    nH_xy = int(math.isqrt(Lxy))
    nW_xy = Lxy // nH_xy if nH_xy > 0 else 1
    if nH_xy * nW_xy == Lxy:
        im0 = axes[0].imshow(xy_norms.reshape(nH_xy, nW_xy), cmap='viridis', aspect='auto')
    else:
        im0 = axes[0].imshow(xy_norms.reshape(1, -1), cmap='viridis', aspect='auto')
    axes[0].set_title(f'XY Tokens ({Lxy})')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # XZ plane
    nH_xz = int(math.isqrt(Lxz))
    nW_xz = Lxz // nH_xz if nH_xz > 0 else 1
    if nH_xz * nW_xz == Lxz:
        im1 = axes[1].imshow(xz_norms.reshape(nH_xz, nW_xz), cmap='viridis', aspect='auto')
    else:
        im1 = axes[1].imshow(xz_norms.reshape(1, -1), cmap='viridis', aspect='auto')
    axes[1].set_title(f'XZ Tokens ({Lxz})')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # YZ plane
    nH_yz = int(math.isqrt(Lyz))
    nW_yz = Lyz // nH_yz if nH_yz > 0 else 1
    if nH_yz * nW_yz == Lyz:
        im2 = axes[2].imshow(yz_norms.reshape(nH_yz, nW_yz), cmap='viridis', aspect='auto')
    else:
        im2 = axes[2].imshow(yz_norms.reshape(1, -1), cmap='viridis', aspect='auto')
    axes[2].set_title(f'YZ Tokens ({Lyz})')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # Combined bar chart
    axes[3].bar(range(len(norms)), norms, width=1.0, color='steelblue', alpha=0.8)
    axes[3].axvline(x=Lxy, color='r', linestyle='--', label='XY|XZ')
    axes[3].axvline(x=Lxy + Lxz, color='g', linestyle='--', label='XZ|YZ')
    axes[3].set_xlabel('Token Index')
    axes[3].set_ylabel('L2 Norm')
    axes[3].set_title('All Token Norms')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Token norm heatmaps saved to {save_path}")


def visualize_trajectory(past_traj, future_traj, pred_trajs=None,
                         save_path='heatmaps_trajectory.png'):
    """
    Visualize predicted vs ground truth trajectories in BEV.

    Args:
        past_traj: (T_past, 2) past trajectory
        future_traj: (T_future, 2) ground truth future
        pred_trajs: (K, T_future, 2) predicted trajectory samples (optional)
        save_path: output file path
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization")
        return

    past = _to_numpy(past_traj)
    future = _to_numpy(future_traj)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_title('Trajectory Prediction (BEV)', fontsize=14, fontweight='bold')

    # Past trajectory
    ax.plot(past[:, 0], past[:, 1], 'b-o', markersize=3, label='Past', linewidth=2)

    # Ground truth future
    ax.plot(future[:, 0], future[:, 1], 'g-o', markersize=3, label='GT Future', linewidth=2)

    # Predicted trajectories
    if pred_trajs is not None:
        pred = _to_numpy(pred_trajs)
        K = pred.shape[0]
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, K))
        for k in range(K):
            ax.plot(pred[k, :, 0], pred[k, :, 1], '-', color=colors[k],
                    alpha=0.6, linewidth=1.5, label=f'Pred {k+1}' if k < 3 else None)

    # Mark origin
    ax.plot(0, 0, 'k*', markersize=15, label='Ego (t=0)')

    ax.set_xlabel('X (forward, m)')
    ax.set_ylabel('Y (left, m)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Trajectory visualization saved to {save_path}")


def run_all_visualizations(config=None, output_dir='.', device_str=None):
    """
    Run forward pass and generate all heatmap visualizations.

    Args:
        config: TriplaneConfig (uses small config if None)
        output_dir: directory to save visualization files
        device_str: 'cuda' or 'cpu'
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if config is None:
        config = TriplaneConfig()
        # Use small config for visualization
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
        config.use_volumetric_rendering = True
        config.num_render_cameras = 2
        config.num_ray_samples = 16
        config.render_image_scale = 0.125
        config.past_frames = 8
        config.future_steps = 10

    if device_str is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")

    # Create model and dummy data
    model = TriplaneDriveModel(config).to(device)
    model.train()

    dataset = DummyDrivingDataset(config, num_samples=4, seed=42)
    sample = dataset[0]
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

    images = batch['images']
    intrinsics = batch['intrinsics']
    extrinsics = batch['extrinsics']
    past_traj = batch['past_trajectory']
    future_traj = batch['future_trajectory']

    print("\n--- Running forward pass ---")

    # Step 1: Image encoder
    features = model.image_encoder(images)
    print(f"Encoder features: {features.shape}")

    # Step 2: 3D Lifting -> Triplanes
    pxy, pxz, pyz = model.lifting(features, intrinsics, extrinsics)
    print(f"Triplane shapes: Pxy={pxy.shape}, Pxz={pxz.shape}, Pyz={pyz.shape}")

    # Step 3: Set planes and patchify
    model.triplane.set_planes(pxy, pxz, pyz)
    sensor_tokens, token_counts = model.patchifier(pxy, pxz, pyz)
    print(f"Sensor tokens: {sensor_tokens.shape}, counts: {token_counts}")

    # Step 4: Full model forward (for rendering)
    outputs = model(images, intrinsics, extrinsics, past_traj, future_traj)

    print("\n--- Generating visualizations ---")

    # 1. Triplane heatmaps
    visualize_triplane_heatmaps(
        pxy, pxz, pyz,
        save_path=os.path.join(output_dir, 'heatmaps_triplane.png')
    )

    # 2. Encoder feature heatmaps
    visualize_encoder_features(
        features, images,
        save_path=os.path.join(output_dir, 'heatmaps_encoder.png')
    )

    # 3. Camera attention/visibility heatmaps
    visualize_attention_weights(
        model, features, intrinsics, extrinsics,
        save_path=os.path.join(output_dir, 'heatmaps_attention.png')
    )

    # 4. Token norm heatmaps
    visualize_token_norms(
        sensor_tokens, token_counts,
        save_path=os.path.join(output_dir, 'heatmaps_tokens.png')
    )

    # 5. Rendered vs input (if rendering enabled)
    if 'rendered_images' in outputs:
        rendered, cam_indices = outputs['rendered_images']
        visualize_rendered_vs_input(
            rendered, images, cam_indices,
            save_path=os.path.join(output_dir, 'heatmaps_rendering.png')
        )

    # 6. Trajectory visualization
    print("\nGenerating trajectory samples...")
    model.eval()
    with torch.no_grad():
        pred_trajs = model.generate_trajectory(
            images, intrinsics, extrinsics, past_traj,
            num_samples=4, temperature=1.0
        )
    visualize_trajectory(
        past_traj[0], future_traj[0], pred_trajs[0],
        save_path=os.path.join(output_dir, 'heatmaps_trajectory.png')
    )

    print(f"\nAll visualizations saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Generate triplane model heatmap visualizations')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Output directory for heatmap images')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint (uses random weights if not provided)')
    args = parser.parse_args()

    run_all_visualizations(output_dir=args.output_dir, device_str=args.device)


if __name__ == '__main__':
    main()
