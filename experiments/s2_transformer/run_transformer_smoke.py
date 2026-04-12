"""
S2 — Transformer smoke test on OPS-SAT-AD (CPU, subset).

Goal: validate that the full pipeline runs end-to-end without errors.
NOT intended to beat baselines — that comes after Optuna tuning on GPU.

Architecture:
  - Encoder-only Transformer (reconstruction-based anomaly detection)
  - 2 layers, dim_model=64, 4 heads, feedforward=128
  - Input: fixed-length sequence (seq_len=64) padded/truncated
  - Loss: MSE reconstruction of the input sequence
  - Inference: anomaly score = mean reconstruction error over the sequence

Subset strategy:
  - Use only sampling=5 segments (shorter → faster on CPU)
  - Take ~100 train-normal + ~50 test segments (~150 total)

Usage:
    python experiments/s2_transformer/run_transformer_smoke.py
    python experiments/s2_transformer/run_transformer_smoke.py --epochs 10 --subset 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import mlflow

from src.evaluation.metrics import compute_metrics, metrics_table


# ── Config ────────────────────────────────────────────────────────────────────

MLFLOW_EXPERIMENT = "s2_transformer_opssat"

DEFAULT_CFG = dict(
    seq_len    = 64,
    d_model    = 64,
    n_heads    = 4,
    n_layers   = 2,
    d_ff       = 128,
    dropout    = 0.1,
    lr         = 1e-3,
    batch_size = 32,
    epochs     = 8,
    subset_n   = 150,   # total segments to use (train+test)
    sampling   = 5,     # only use this sampling rate for smoke test
    seed       = 42,
)


# ── Dataset ───────────────────────────────────────────────────────────────────

def pad_or_truncate(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Pad with zeros (left) or truncate (right) to fixed seq_len."""
    if len(arr) >= seq_len:
        return arr[:seq_len]
    pad = np.zeros(seq_len - len(arr), dtype=arr.dtype)
    return np.concatenate([pad, arr])


class SegmentDataset(Dataset):
    """One segment = one sample. Values are z-scored per segment."""

    def __init__(self, segments: list[np.ndarray], seq_len: int):
        self.seq_len = seq_len
        self.data = []
        for s in segments:
            s = s.astype(np.float32)
            s = (s - s.mean()) / (s.std() + 1e-8)  # z-score per segment
            s = pad_or_truncate(s, seq_len)
            self.data.append(torch.from_numpy(s).unsqueeze(-1))  # (seq_len, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class TransformerReconstructionAD(nn.Module):
    """
    Encoder-only Transformer for anomaly detection via reconstruction.

    Input  : (batch, seq_len, n_features=1)
    Output : (batch, seq_len, n_features=1)  — reconstructed sequence
    """

    def __init__(self, seq_len: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, dropout: float, n_features: int = 1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)

        # Learnable positional encoding
        self.pos_embedding = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.input_proj(x) + self.pos_embedding(positions)
        h = self.transformer(h)
        return self.output_proj(h)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_subset(cfg: dict) -> tuple:
    """
    Load a small subset of segments filtered by sampling rate.
    Returns (train_normal, test_all, test_labels).
    """
    seg = pd.read_csv(
        Path(__file__).parents[2] / "reference" / "data" / "segments.csv",
        parse_dates=["timestamp"],
    )

    # Filter by sampling rate
    seg = seg[seg.sampling == cfg["sampling"]].copy()

    seg_meta = seg.groupby("segment").agg(
        anomaly=("anomaly", "first"),
        train=("train", "first"),
        n_points=("value", "count"),
    ).reset_index()

    # Train normal segments
    train_normal_ids = seg_meta[
        (seg_meta.train == 1) & (seg_meta.anomaly == 0) & (seg_meta.n_points >= 10)
    ]["segment"].values

    # Test segments (normal + anomaly)
    test_ids = seg_meta[seg_meta.train == 0]["segment"].values
    test_labels = seg_meta[seg_meta.train == 0].set_index("segment")["anomaly"]

    # Subsample
    rng = np.random.default_rng(cfg["seed"])
    n_train = min(len(train_normal_ids), max(50, cfg["subset_n"] - len(test_ids)))
    n_test  = min(len(test_ids), cfg["subset_n"] - n_train)

    train_normal_ids = rng.choice(train_normal_ids, size=n_train, replace=False)
    test_ids         = rng.choice(test_ids, size=n_test, replace=False)

    def get_values(seg_id):
        return seg[seg.segment == seg_id].sort_values("timestamp")["value"].values

    train_segs  = [get_values(i) for i in train_normal_ids]
    test_segs   = [get_values(i) for i in test_ids]
    test_y      = test_labels.loc[test_ids].values

    print(f"Sampling rate : {cfg['sampling']}s interval")
    print(f"Train (normal): {len(train_segs)} segments")
    print(f"Test  (all)   : {len(test_segs)} segments  |  anomaly rate: {test_y.mean():.1%}")

    return train_segs, test_segs, test_y


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    criterion  = nn.MSELoss()
    for x in loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_hat = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def reconstruction_errors(model, segs: list[np.ndarray], seq_len: int, device) -> np.ndarray:
    """Return per-segment mean MSE reconstruction error."""
    model.eval()
    errors = []
    for s in segs:
        s = s.astype(np.float32)
        s = (s - s.mean()) / (s.std() + 1e-8)
        s = pad_or_truncate(s, seq_len)
        x = torch.from_numpy(s).unsqueeze(0).unsqueeze(-1).to(device)  # (1, T, 1)
        x_hat = model(x)
        err = ((x - x_hat) ** 2).mean().item()
        errors.append(err)
    return np.array(errors)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(cfg: dict):
    torch.manual_seed(cfg["seed"])
    device = torch.device("cpu")  # smoke test: CPU only
    print(f"\nDevice: {device}  |  PyTorch {torch.__version__}")

    # ── Data ──
    train_segs, test_segs, test_y = load_subset(cfg)

    train_ds = SegmentDataset(train_segs, cfg["seq_len"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    # ── Model ──
    model = TransformerReconstructionAD(
        seq_len  = cfg["seq_len"],
        d_model  = cfg["d_model"],
        n_heads  = cfg["n_heads"],
        n_layers = cfg["n_layers"],
        d_ff     = cfg["d_ff"],
        dropout  = cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params : {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    # ── MLflow ──
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="smoke_test_cpu"):
        mlflow.log_params({**cfg, "n_params": n_params, "device": str(device)})

        # ── Train ──
        print(f"\n{'Epoch':>5} {'Train MSE':>12} {'Time':>8}")
        print("-" * 30)
        losses = []
        t0 = time.time()

        for epoch in range(1, cfg["epochs"] + 1):
            t_ep = time.time()
            loss = train_epoch(model, train_dl, optimizer, device)
            scheduler.step()
            losses.append(loss)
            mlflow.log_metric("train_mse", loss, step=epoch)
            elapsed = time.time() - t_ep
            print(f"{epoch:>5}  {loss:>12.6f}  {elapsed:>7.1f}s")

        total_time = time.time() - t0

        # ── Evaluate ──
        print("\nComputing reconstruction errors on test set...")
        scores = reconstruction_errors(model, test_segs, cfg["seq_len"], device)

        # Threshold: 95th percentile of train normal errors
        train_errors = reconstruction_errors(model, train_segs, cfg["seq_len"], device)
        threshold = np.percentile(train_errors, 95)
        preds = (scores > threshold).astype(int)

        metrics = compute_metrics(test_y, preds, scores)
        mlflow.log_metrics({**metrics, "threshold": threshold, "train_time_s": total_time})

        # ── Report ──
        print(f"\n{'='*55}")
        print(f"  Threshold (p95 train normal MSE): {threshold:.6f}")
        print(f"  Train time                      : {total_time:.1f}s")
        print(f"\n  Results on test set:")
        print(f"  {'Metric':<12} {'Value':>8}")
        print(f"  {'-'*22}")
        for k, v in metrics.items():
            print(f"  {k:<12} {v:>8.3f}")
        print(f"{'='*55}")

        # Loss convergence check
        loss_drop = (losses[0] - losses[-1]) / (losses[0] + 1e-10)
        print(f"\n  Loss convergence: {losses[0]:.6f} → {losses[-1]:.6f}  (drop {loss_drop:.1%})")
        if loss_drop > 0.05:
            print("  OK — loss bajó durante el entrenamiento.")
        else:
            print("  WARNING — loss no bajó significativamente. Revisar LR o datos.")

        print(f"\n  MLflow run logged.")
        print(f"  mlflow ui --backend-store-uri mlruns/")


def parse_args():
    p = argparse.ArgumentParser()
    for k, v in DEFAULT_CFG.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
