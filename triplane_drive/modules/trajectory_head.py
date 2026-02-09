"""Trajectory prediction head: decodes transformer hidden states to trajectory logits."""

import torch
import torch.nn as nn

from config import TriplaneConfig


class TrajectoryHead(nn.Module):
    """
    Decodes transformer hidden states into trajectory token logits.

    Operates on interleaved x,y positions:
    - Even positions (0, 2, 4, ...): predict x-coordinate token
    - Odd positions (1, 3, 5, ...): predict y-coordinate token
    """

    def __init__(self, config: TriplaneConfig):
        super().__init__()
        self.config = config

        self.head_x = nn.Linear(config.d_ar, config.traj_vocab_size)
        self.head_y = nn.Linear(config.d_ar, config.traj_vocab_size)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (B, L_future_traj, d_ar) hidden states for future trajectory tokens

        Returns:
            logits_x: (B, T, vocab_size) logits for x-coordinates
            logits_y: (B, T, vocab_size) logits for y-coordinates
        """
        B, L, D = hidden_states.shape

        # Split into x and y positions (interleaved)
        # Even indices -> x, odd indices -> y
        x_hidden = hidden_states[:, 0::2]  # (B, T, D)
        y_hidden = hidden_states[:, 1::2]  # (B, T, D)

        logits_x = self.head_x(x_hidden)  # (B, T, vocab)
        logits_y = self.head_y(y_hidden)  # (B, T, vocab)

        return logits_x, logits_y

    def sample(self, hidden_states, temperature=1.0):
        """
        Sample trajectory tokens from the distribution.

        Args:
            hidden_states: (B, L_future_traj, d_ar)
            temperature: sampling temperature

        Returns:
            tokens_x: (B, T) sampled x-coordinate tokens
            tokens_y: (B, T) sampled y-coordinate tokens
        """
        logits_x, logits_y = self.forward(hidden_states)

        if temperature <= 0:
            tokens_x = logits_x.argmax(dim=-1)
            tokens_y = logits_y.argmax(dim=-1)
        else:
            probs_x = torch.softmax(logits_x / temperature, dim=-1)
            probs_y = torch.softmax(logits_y / temperature, dim=-1)
            tokens_x = torch.multinomial(probs_x.view(-1, probs_x.shape[-1]), 1).view(probs_x.shape[:-1])
            tokens_y = torch.multinomial(probs_y.view(-1, probs_y.shape[-1]), 1).view(probs_y.shape[:-1])

        return tokens_x, tokens_y
