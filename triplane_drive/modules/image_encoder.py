"""Image encoder: DINOv2-small wrapper with fallback ConvNet encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TriplaneConfig


class FallbackImageEncoder(nn.Module):
    """
    Simple ConvNet that mimics DINOv2's output shape for testing
    without pretrained weights.
    """

    def __init__(self, config: TriplaneConfig):
        super().__init__()
        self.config = config
        self.feat_h = config.image_feat_h
        self.feat_w = config.image_feat_w

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, config.dino_embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(config.dino_embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) images

        Returns:
            features: (B, Hf, Wf, embed_dim) feature maps
        """
        feat = self.encoder(x)  # (B, embed_dim, H', W')
        # Interpolate to target feature map size
        feat = F.interpolate(feat, size=(self.feat_h, self.feat_w), mode='bilinear', align_corners=False)
        feat = feat.permute(0, 2, 3, 1)  # (B, Hf, Wf, embed_dim)
        return feat


class ImageEncoder(nn.Module):
    """
    Wraps DINOv2-small (ViT-S/14) with projection to feature_dim.
    Falls back to a simple ConvNet if DINOv2 is unavailable.
    """

    def __init__(self, config: TriplaneConfig):
        super().__init__()
        self.config = config

        if config.use_pretrained_dino:
            try:
                self.backbone = torch.hub.load(
                    'facebookresearch/dinov2', config.dino_model, pretrained=True
                )
                # Freeze backbone
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.use_dino = True
            except Exception:
                print("Warning: Could not load DINOv2, using fallback encoder")
                self.backbone = FallbackImageEncoder(config)
                self.use_dino = False
        else:
            self.backbone = FallbackImageEncoder(config)
            self.use_dino = False

        # Project from dino_embed_dim to feature_dim
        self.projection = nn.Linear(config.dino_embed_dim, config.feature_dim)

    def forward(self, images):
        """
        Args:
            images: (B, C, 3, H, W) multi-camera images

        Returns:
            features: (B, C, Hf, Wf, Df) per-camera feature maps
        """
        B, C, _, H, W = images.shape
        config = self.config

        # Flatten batch and camera dims
        x = images.reshape(B * C, 3, H, W)

        if self.use_dino:
            # Pad to multiple of patch size
            pad_h = config.padded_image_h - H
            pad_w = config.padded_image_w - W
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))

            # DINOv2 forward: get patch tokens
            with torch.no_grad():
                features = self.backbone.forward_features(x)
                patch_tokens = features['x_norm_patchtokens']  # (B*C, N, embed_dim)

            Hf, Wf = config.image_feat_h, config.image_feat_w
            patch_tokens = patch_tokens[:, :Hf * Wf]  # trim if needed
            feat = patch_tokens.reshape(B * C, Hf, Wf, config.dino_embed_dim)
        else:
            feat = self.backbone(x)  # (B*C, Hf, Wf, embed_dim)

        # Project to feature_dim
        feat = self.projection(feat)  # (B*C, Hf, Wf, Df)

        # Reshape back
        feat = feat.reshape(B, C, config.image_feat_h, config.image_feat_w, config.feature_dim)
        return feat
