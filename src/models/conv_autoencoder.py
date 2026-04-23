"""
Conv1D Autoencoder for temporal anomaly detection (Quesada 2026).

Used in the AE → Embedding → LPI pipeline (experiments/s3_esa_adb/run_ae_lpi.py):
  1. Train on normal windows only (unsupervised)
  2. Extract bottleneck embeddings for all windows
  3. Feed embeddings to LPINormalizingFlow as features

Architecture (window_size=256):
  Encoder:  Conv(1→32,k7,s2) → Conv(32→64,k5,s2) → Conv(64→128,k3,s2)
            → AdaptiveAvgPool1d(1) → Linear(128, embedding_dim)
  Decoder:  Linear → Upsample → Conv refine → ConvT×3

Output sizes are computed dynamically from window_size so the model is valid
for any window_size ≥ 7.  Default targets window_size=256 (ESA-Mission1).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─── Dimension helpers ────────────────────────────────────────────────────────

def _conv1d_out(L: int, k: int, s: int, p: int) -> int:
    return math.floor((L + 2 * p - k) / s + 1)


def _convT1d_out(L: int, k: int, s: int, p: int, op: int = 0) -> int:
    return (L - 1) * s - 2 * p + k + op


# ─── Model ────────────────────────────────────────────────────────────────────

class ConvAutoencoder(nn.Module):
    """
    Symmetric Conv1D autoencoder with a compact bottleneck embedding.

    Input shape : (B, 1, window_size)   — Conv1d convention (channels first)
    forward()   : returns (reconstruction, embedding)
                  reconstruction : (B, 1, window_size)
                  embedding      : (B, embedding_dim)

    The embedding is the output of the encoder bottleneck, not the
    reconstruction error — it is this embedding that feeds the LPI.

    For window_size=256, embedding_dim=32:
      Encoder sizes: 256 → 128 → 64 → 32 → pool → 128 → 32
      Decoder sizes: 32 → 128 → upsample(32) → refine → 64 → 128 → 256
      Total params:  ~127 K
    """

    def __init__(self, window_size: int = 256, embedding_dim: int = 32) -> None:
        super().__init__()
        self.window_size = window_size
        self.embedding_dim = embedding_dim

        # ── Encoder conv sizes ────────────────────────────────────────────────
        L1 = _conv1d_out(window_size, k=7, s=2, p=3)
        L2 = _conv1d_out(L1, k=5, s=2, p=2)
        L3 = _conv1d_out(L2, k=3, s=2, p=1)   # spatial size at bottleneck
        self._spatial = L3

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc_conv = nn.Sequential(
            nn.Conv1d(1,  32,  kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64,  kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.enc_pool = nn.AdaptiveAvgPool1d(1)
        self.enc_proj = nn.Linear(128, embedding_dim)

        # ── Decoder ───────────────────────────────────────────────────────────
        # output_padding computed so each ConvTranspose restores the exact
        # encoder size at each stage.
        op1 = L2 - _convT1d_out(L3, k=3, s=2, p=1)
        op2 = L1 - _convT1d_out(L2, k=5, s=2, p=2)
        op3 = window_size - _convT1d_out(L1, k=7, s=2, p=3)

        assert all(0 <= op <= 1 for op in (op1, op2, op3)), (
            f"output_padding out of range for window_size={window_size}: "
            f"op1={op1}, op2={op2}, op3={op3}"
        )

        self.dec_proj = nn.Linear(embedding_dim, 128)
        # Conv1d refine: spatial info restored by Upsample, then learned refinement
        self.dec_refine = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=op1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=op2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=op3),
        )

    # ── Forward passes ────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, T) → (B, embedding_dim)"""
        h = self.enc_conv(x)                  # (B, 128, spatial)
        h = self.enc_pool(h).squeeze(-1)       # (B, 128)
        return self.enc_proj(h)                # (B, embedding_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, embedding_dim) → (B, 1, T)"""
        h = self.dec_proj(z).unsqueeze(-1)     # (B, 128, 1)
        h = F.interpolate(                     # (B, 128, spatial)
            h, size=self._spatial, mode="linear", align_corners=False
        )
        h = self.dec_refine(h)                 # (B, 128, spatial)  [learned refine]
        return self.dec_conv(h)                # (B, 1, T)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, embedding)."""
        emb  = self.encode(x)
        recon = self.decode(emb)
        return recon, emb


# ─── Training helper ──────────────────────────────────────────────────────────

def train_autoencoder(
    model:        ConvAutoencoder,
    train_windows: np.ndarray,   # (N, T) — normal windows only
    val_windows:   np.ndarray,   # (M, T) — held-out normal windows for early stop
    epochs:       int   = 100,
    lr:           float = 1e-3,
    batch_size:   int   = 512,
    device:       str   = "cpu",
    patience:     int   = 15,
) -> ConvAutoencoder:
    """
    Train the autoencoder on normal windows with MSE reconstruction loss.

    Early stopping monitors val MSE.  Only normal windows are used — the
    model learns a "normal manifold" so anomalous windows produce higher
    reconstruction error (but reconstruction error is NOT used as the
    anomaly score here; the embedding is).

    Returns the model with the best validation weights restored.
    """
    dev = torch.device(device)
    model = model.to(dev)

    def _make_loader(windows: np.ndarray, shuffle: bool) -> DataLoader:
        t = torch.from_numpy(windows.astype(np.float32)).unsqueeze(1)  # (N, 1, T)
        return DataLoader(
            TensorDataset(t),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=min(4, batch_size // 128),
            pin_memory=(device != "cpu"),
            drop_last=False,
        )

    train_loader = _make_loader(train_windows, shuffle=True)
    val_loader   = _make_loader(val_windows,   shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-5
    )

    best_val_loss = float("inf")
    best_state    = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve    = 0
    t0            = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(dev)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(train_windows)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(dev)
                recon, _ = model(batch)
                val_loss += F.mse_loss(recon, batch).item() * len(batch)
        val_loss /= len(val_windows)

        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.perf_counter() - t0
            print(
                f"  Epoch {epoch:>4}/{epochs}  "
                f"train_mse={train_loss:.5f}  val_mse={val_loss:.5f}  "
                f"best={best_val_loss:.5f}  no_improve={no_improve}  "
                f"t={elapsed:.1f}s"
            )

        if no_improve >= patience:
            print(
                f"  Early stop at epoch {epoch}  "
                f"(val_mse={val_loss:.5f}, best={best_val_loss:.5f})"
            )
            break

    model.load_state_dict(best_state)
    model.eval()
    elapsed = time.perf_counter() - t0
    print(f"  Training done in {elapsed:.1f}s — best val_mse={best_val_loss:.5f}")
    return model


# ─── Embedding extraction ─────────────────────────────────────────────────────

def extract_embeddings(
    model:      ConvAutoencoder,
    windows:    np.ndarray,   # (N, T)
    batch_size: int = 512,
    device:     str = "cpu",
) -> np.ndarray:              # (N, embedding_dim)
    """
    Extract bottleneck embeddings for all windows in batches.

    The model must already be trained.  Runs in eval mode with torch.no_grad().
    """
    dev    = torch.device(device)
    model  = model.to(dev).eval()
    n      = len(windows)
    d      = model.embedding_dim
    result = np.empty((n, d), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end   = min(start + batch_size, n)
            batch = torch.from_numpy(
                windows[start:end].astype(np.float32)
            ).unsqueeze(1).to(dev)                  # (B, 1, T)
            result[start:end] = model.encode(batch).cpu().numpy()

    return result
