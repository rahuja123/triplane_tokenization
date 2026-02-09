"""Image encoder: DINOv2-small wrapper with fallback ConvNet encoder.

Supports three backends:
  1. timm: `pip install timm` — recommended, works on Python 3.9+
  2. torch.hub: `torch.hub.load('facebookresearch/dinov2', ...)` — requires Python 3.10+
  3. FallbackImageEncoder: simple ConvNet that mimics DINOv2 output shape
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TriplaneConfig


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


def _load_dino_timm(config: TriplaneConfig):
    """Load DINOv2 via timm (recommended). Returns (model, 'timm') or raises."""
    import timm
    # Map config.dino_model to timm model name
    timm_model_map = {
        'dinov2_vits14': 'vit_small_patch14_dinov2',
        'dinov2_vitb14': 'vit_base_patch14_dinov2',
        'dinov2_vitl14': 'vit_large_patch14_dinov2',
    }
    timm_name = timm_model_map.get(config.dino_model, 'vit_small_patch14_dinov2')
    model = timm.create_model(timm_name, pretrained=True, dynamic_img_size=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Loaded DINOv2 via timm: {timm_name} ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, 'timm'


def _load_dino_hub(config: TriplaneConfig):
    """Load DINOv2 via torch.hub (requires Python 3.10+). Returns (model, 'hub') or raises."""
    model = torch.hub.load('facebookresearch/dinov2', config.dino_model, pretrained=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Loaded DINOv2 via torch.hub: {config.dino_model}")
    return model, 'hub'


class ImageEncoder(nn.Module):
    """
    Wraps DINOv2-small (ViT-S/14) with projection to feature_dim.
    Falls back to a simple ConvNet if DINOv2 is unavailable.

    Loading priority: timm → torch.hub → FallbackImageEncoder
    """

    def __init__(self, config: TriplaneConfig):
        super().__init__()
        self.config = config
        self.dino_backend = None  # 'timm', 'hub', or None

        if config.use_pretrained_dino:
            loaded = False
            # Try timm first (works on Python 3.9+)
            try:
                self.backbone, self.dino_backend = _load_dino_timm(config)
                loaded = True
            except Exception as e:
                print(f"timm DINOv2 load failed: {e}")

            # Fall back to torch.hub (requires Python 3.10+)
            if not loaded:
                try:
                    self.backbone, self.dino_backend = _load_dino_hub(config)
                    loaded = True
                except Exception as e:
                    print(f"torch.hub DINOv2 load failed: {e}")

            if not loaded:
                print("Warning: Could not load DINOv2 from timm or torch.hub, using fallback encoder")
                self.backbone = FallbackImageEncoder(config)
        else:
            self.backbone = FallbackImageEncoder(config)

        # Project from dino_embed_dim to feature_dim
        self.projection = nn.Linear(config.dino_embed_dim, config.feature_dim)

    @property
    def use_dino(self):
        return self.dino_backend is not None

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

        if self.dino_backend == 'timm':
            # Pad to multiple of patch size (14)
            pad_h = config.padded_image_h - H
            pad_w = config.padded_image_w - W
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))

            # timm DINOv2: forward_features returns (B, 1+N, D) with CLS at index 0
            with torch.no_grad():
                tokens = self.backbone.forward_features(x)  # (B*C, 1+N, embed_dim)
                patch_tokens = tokens[:, 1:, :]  # skip CLS token → (B*C, N, embed_dim)

            Hf, Wf = config.image_feat_h, config.image_feat_w
            patch_tokens = patch_tokens[:, :Hf * Wf]  # trim padding patches if needed
            feat = patch_tokens.reshape(B * C, Hf, Wf, config.dino_embed_dim)

        elif self.dino_backend == 'hub':
            # Pad to multiple of patch size
            pad_h = config.padded_image_h - H
            pad_w = config.padded_image_w - W
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))

            # torch.hub DINOv2: forward_features returns dict with 'x_norm_patchtokens'
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
