"""
Encoder-only Transformer for reconstruction-based anomaly detection.

Shared model definition used by all S2 experiment scripts.
Centralised here to avoid the two scripts having diverging model code.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TransformerReconstructionAD(nn.Module):
    """
    Encoder-only Transformer: reconstructs its own input.

    Anomaly score  = mean squared reconstruction error per window.
    Segment score  = max window score across all windows in the segment.

    Input  : (batch, seq_len, 1)
    Output : (batch, seq_len, 1)
    """

    def __init__(
        self,
        seq_len:  int,
        d_model:  int,
        n_heads:  int,
        n_layers: int,
        d_ff:     int,
        dropout:  float,
    ):
        super().__init__()
        self.input_proj    = nn.Linear(1, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h   = self.input_proj(x) + self.pos_embedding(pos)
        h   = self.transformer(h)
        return self.output_proj(h)


# ── Training helpers ──────────────────────────────────────────────────────────

def build_dataloader(windows: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap a (N, T) float32 array into a DataLoader of (N, T, 1) tensors."""
    t = torch.from_numpy(windows).unsqueeze(-1)   # (N, T, 1)
    return DataLoader(TensorDataset(t), batch_size=batch_size,
                      shuffle=shuffle, pin_memory=False)


def train_epoch(
    model: TransformerReconstructionAD,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.MSELoss()
    total = 0.0
    for (x,) in loader:
        x = x.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), x)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def window_reconstruction_errors(
    model: TransformerReconstructionAD,
    windows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Return per-window MSE reconstruction error.

    Parameters
    ----------
    windows : float32 array of shape (n_windows, window_size)

    Returns
    -------
    errors : float32 array of shape (n_windows,)
    """
    model.eval()
    loader = build_dataloader(windows, batch_size=batch_size, shuffle=False)
    parts  = []
    for (x,) in loader:
        x    = x.to(device)
        xhat = model(x)
        err  = ((x - xhat) ** 2).mean(dim=(1, 2))   # per-window scalar
        parts.append(err.cpu().numpy())
    return np.concatenate(parts)
