"""End-to-end triplane driving model."""

import torch
import torch.nn as nn

from .config import TriplaneConfig
from .modules.image_encoder import ImageEncoder
from .modules.lifting import LiftingModule
from .modules.triplane import TriplaneRepresentation
from .modules.patchify import TriplanePatchifier
from .modules.ar_transformer import ARTransformer, TrajectoryTokenEmbedding
from .modules.trajectory_head import TrajectoryHead
from .modules.volumetric_renderer import VolumetricRenderer


class TriplaneDriveModel(nn.Module):
    """
    End-to-end pipeline:
    1. ImageEncoder: multi-camera images -> feature maps
    2. LiftingModule: feature maps + camera params -> 3D queries -> triplane
    3. TriplanePatchifier: triplane -> tokens
    4. ARTransformer: tokens + past trajectory -> hidden states
    5. TrajectoryHead: hidden states -> trajectory predictions

    Optional (training only):
    6. VolumetricRenderer: triplane -> reconstructed images
    """

    def __init__(self, config: TriplaneConfig):
        super().__init__()
        self.config = config

        # Core modules
        self.image_encoder = ImageEncoder(config)
        self.lifting = LiftingModule(config)
        self.triplane = TriplaneRepresentation(config)
        self.patchifier = TriplanePatchifier(config)
        self.traj_embed = TrajectoryTokenEmbedding(config)
        self.ar_transformer = ARTransformer(config)
        self.traj_head = TrajectoryHead(config)

        # Volumetric renderer (for reconstruction loss during training)
        if config.use_volumetric_rendering:
            self.renderer = VolumetricRenderer(config)
        else:
            self.renderer = None

    def forward(self, images, intrinsics, extrinsics, past_trajectory,
                future_trajectory=None):
        """
        Args:
            images: (B, C, 3, H, W) multi-camera images
            intrinsics: (B, C, 3, 3) camera intrinsics
            extrinsics: (B, C, 4, 4) camera extrinsics
            past_trajectory: (B, T_past, 2) past ego trajectory
            future_trajectory: (B, T_future, 2) ground truth future trajectory (training only)

        Returns:
            outputs: dict containing:
                - logits_x, logits_y: trajectory logits (if future_trajectory provided)
                - target_x, target_y: discretized ground truth tokens
                - rendered_images: tuple of (rendered, cam_indices) if rendering enabled
                - sensor_tokens: the sensor token sequence
        """
        outputs = {}

        # 1. Encode images
        features = self.image_encoder(images)  # (B, C, Hf, Wf, Df)

        # 2. Lift to 3D and produce triplanes
        pxy, pxz, pyz = self.lifting(features, intrinsics, extrinsics)

        # Store triplane for rendering
        self.triplane.set_planes(pxy, pxz, pyz)

        # 3. Patchify into tokens
        sensor_tokens, token_counts = self.patchifier(pxy, pxz, pyz)  # (B, L_sensor, d_ar)
        outputs['sensor_tokens'] = sensor_tokens
        outputs['token_counts'] = token_counts

        # 4. Embed past trajectory
        past_traj_tokens = self.traj_embed(past_trajectory)  # (B, T_past*2, d_ar)

        # 5. Build input sequence for transformer
        if future_trajectory is not None:
            # Teacher forcing: include future trajectory tokens (shifted by 1)
            future_traj_tokens = self.traj_embed(future_trajectory)  # (B, T_future*2, d_ar)
            traj_tokens = torch.cat([past_traj_tokens, future_traj_tokens], dim=1)
        else:
            traj_tokens = past_traj_tokens

        # 6. AR Transformer
        hidden = self.ar_transformer(sensor_tokens, traj_tokens)  # (B, L_total, d_ar)

        # 7. Trajectory prediction (on future positions)
        if future_trajectory is not None:
            L_sensor = sensor_tokens.shape[1]
            L_past = past_traj_tokens.shape[1]
            # The future trajectory hidden states start after sensor + past tokens
            # But we predict the *next* token, so we use hidden states offset by 1
            future_start = L_sensor + L_past - 1  # -1 because we predict next from current
            future_end = future_start + future_trajectory.shape[1] * 2
            future_hidden = hidden[:, future_start:future_end]

            logits_x, logits_y = self.traj_head(future_hidden)
            outputs['logits_x'] = logits_x
            outputs['logits_y'] = logits_y

            # Ground truth tokens
            target_x, target_y = self.traj_embed.discretize(future_trajectory)
            outputs['target_x'] = target_x
            outputs['target_y'] = target_y

        # 8. Volumetric rendering (training only)
        if self.training and self.renderer is not None:
            rendered, cam_indices = self.renderer(self.triplane, intrinsics, extrinsics)
            outputs['rendered_images'] = (rendered, cam_indices)

        return outputs

    @torch.no_grad()
    def generate_trajectory(self, images, intrinsics, extrinsics, past_trajectory,
                            num_samples=6, temperature=1.0):
        """
        Autoregressively generate future trajectory samples.

        Args:
            images: (B, C, 3, H, W)
            intrinsics: (B, C, 3, 3)
            extrinsics: (B, C, 4, 4)
            past_trajectory: (B, T_past, 2)
            num_samples: number of trajectory samples to generate
            temperature: sampling temperature

        Returns:
            trajectories: (B, num_samples, T_future, 2) sampled trajectories
        """
        self.eval()
        B = images.shape[0]
        config = self.config

        # Encode and get sensor tokens (shared across samples)
        features = self.image_encoder(images)
        pxy, pxz, pyz = self.lifting(features, intrinsics, extrinsics)
        sensor_tokens, _ = self.patchifier(pxy, pxz, pyz)
        past_traj_tokens = self.traj_embed(past_trajectory)

        all_trajectories = []

        for _ in range(num_samples):
            # Start with past trajectory tokens
            current_traj_tokens = past_traj_tokens.clone()

            generated_x = []
            generated_y = []

            for t in range(config.future_steps):
                # Forward through transformer
                hidden = self.ar_transformer(sensor_tokens, current_traj_tokens)

                # Get the last hidden state (predicts next token)
                last_hidden = hidden[:, -1:, :]  # (B, 1, d_ar)

                # Predict x coordinate
                x_logits = self.traj_head.head_x(last_hidden.squeeze(1))  # (B, vocab)
                if temperature <= 0:
                    x_token = x_logits.argmax(dim=-1)
                else:
                    x_probs = torch.softmax(x_logits / temperature, dim=-1)
                    x_token = torch.multinomial(x_probs, 1).squeeze(-1)

                # Embed x token and append
                x_emb = self.traj_embed.embed_x(x_token) + self.traj_embed.type_embed(
                    torch.zeros_like(x_token))
                current_traj_tokens = torch.cat([current_traj_tokens, x_emb.unsqueeze(1)], dim=1)

                # Forward again to predict y
                hidden = self.ar_transformer(sensor_tokens, current_traj_tokens)
                last_hidden = hidden[:, -1:, :]

                y_logits = self.traj_head.head_y(last_hidden.squeeze(1))
                if temperature <= 0:
                    y_token = y_logits.argmax(dim=-1)
                else:
                    y_probs = torch.softmax(y_logits / temperature, dim=-1)
                    y_token = torch.multinomial(y_probs, 1).squeeze(-1)

                # Embed y token and append
                y_emb = self.traj_embed.embed_y(y_token) + self.traj_embed.type_embed(
                    torch.ones_like(y_token))
                current_traj_tokens = torch.cat([current_traj_tokens, y_emb.unsqueeze(1)], dim=1)

                generated_x.append(x_token)
                generated_y.append(y_token)

            # Convert to continuous trajectory
            tokens_x = torch.stack(generated_x, dim=1)  # (B, T_future)
            tokens_y = torch.stack(generated_y, dim=1)
            trajectory = self.traj_embed.undiscretize(tokens_x, tokens_y)  # (B, T_future, 2)
            all_trajectories.append(trajectory)

        trajectories = torch.stack(all_trajectories, dim=1)  # (B, num_samples, T_future, 2)
        return trajectories
