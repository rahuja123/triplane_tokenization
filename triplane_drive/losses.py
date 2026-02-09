"""Loss functions for training the triplane driving model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TriplaneConfig


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss as fallback for LPIPS.
    Uses early VGG layers for perceptual similarity.
    """

    def __init__(self):
        super().__init__()
        try:
            import torchvision.models as models
            vgg = models.vgg16(weights=None)
            # Use first 3 blocks
            self.features = nn.Sequential(*list(vgg.features[:16]))
            for param in self.features.parameters():
                param.requires_grad = False
            self.has_vgg = True
        except Exception:
            self.has_vgg = False

    def forward(self, pred, target):
        if not self.has_vgg:
            # Fallback to simple MSE
            return F.mse_loss(pred, target)

        # Ensure VGG features are on the same device as input
        self.features = self.features.to(pred.device)
        pred_feat = self.features(pred)
        target_feat = self.features(target)
        return F.mse_loss(pred_feat, target_feat)


class ReconstructionLoss(nn.Module):
    """
    L = lambda_lpips * LPIPS(I, I_hat) + lambda_l1 * ||I - I_hat||_1

    Uses LPIPS package if available, falls back to VGG perceptual loss.
    """

    def __init__(self, config: TriplaneConfig):
        super().__init__()
        self.lambda_lpips = config.lambda_lpips
        self.lambda_l1 = config.lambda_l1

        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='vgg')
            for param in self.lpips_fn.parameters():
                param.requires_grad = False
            self.use_lpips = True
        except (ImportError, Exception):
            self.lpips_fn = PerceptualLoss()
            self.use_lpips = False

    def forward(self, pred_images, target_images):
        """
        Args:
            pred_images: (B, C, 3, H, W) rendered images
            target_images: (B, C, 3, H, W) ground truth images

        Returns:
            loss: scalar reconstruction loss
        """
        B, C = pred_images.shape[:2]

        # Flatten batch and camera dims
        pred_flat = pred_images.reshape(B * C, 3, pred_images.shape[3], pred_images.shape[4])
        target_flat = target_images.reshape(B * C, 3, target_images.shape[3], target_images.shape[4])

        # Resize target to match pred if needed
        if pred_flat.shape[-2:] != target_flat.shape[-2:]:
            target_flat = F.interpolate(
                target_flat, size=pred_flat.shape[-2:],
                mode='bilinear', align_corners=False
            )

        # L1 loss
        l1_loss = F.l1_loss(pred_flat, target_flat)

        # Perceptual loss
        if self.use_lpips:
            # LPIPS expects [-1, 1] range
            lpips_loss = self.lpips_fn(pred_flat * 2 - 1, target_flat * 2 - 1).mean()
        else:
            lpips_loss = self.lpips_fn(pred_flat, target_flat)

        return self.lambda_lpips * lpips_loss + self.lambda_l1 * l1_loss


class TrajectoryLoss(nn.Module):
    """Cross-entropy loss on next-trajectory-token prediction."""

    def __init__(self, config: TriplaneConfig):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits_x, logits_y, target_x, target_y):
        """
        Args:
            logits_x: (B, T, vocab_size) predicted x-coordinate logits
            logits_y: (B, T, vocab_size) predicted y-coordinate logits
            target_x: (B, T) ground truth x-coordinate token indices
            target_y: (B, T) ground truth y-coordinate token indices

        Returns:
            loss: scalar cross-entropy loss
        """
        B, T, V = logits_x.shape

        loss_x = self.ce(logits_x.reshape(B * T, V), target_x.reshape(B * T))
        loss_y = self.ce(logits_y.reshape(B * T, V), target_y.reshape(B * T))

        return (loss_x + loss_y) / 2.0


class CombinedLoss(nn.Module):
    """Combines reconstruction and trajectory prediction losses."""

    def __init__(self, config: TriplaneConfig):
        super().__init__()
        self.recon_loss = ReconstructionLoss(config)
        self.traj_loss = TrajectoryLoss(config)
        self.use_recon = config.use_volumetric_rendering

    def forward(self, model_output, batch):
        """
        Args:
            model_output: dict with keys from model.forward()
            batch: dict with input data

        Returns:
            total_loss: scalar
            loss_dict: dict of individual losses for logging
        """
        losses = {}

        # Trajectory loss (always used)
        if 'logits_x' in model_output and 'target_x' in model_output:
            traj_loss = self.traj_loss(
                model_output['logits_x'], model_output['logits_y'],
                model_output['target_x'], model_output['target_y']
            )
            losses['trajectory'] = traj_loss

        # Reconstruction loss (only during tokenizer pretraining)
        if self.use_recon and 'rendered_images' in model_output:
            rendered, cam_indices = model_output['rendered_images']
            target = batch['images'][:, cam_indices]
            recon_loss = self.recon_loss(rendered, target)
            losses['reconstruction'] = recon_loss

        total = sum(losses.values())
        losses['total'] = total

        return total, losses
