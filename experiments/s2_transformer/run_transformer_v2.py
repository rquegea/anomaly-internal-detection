"""
S2 — Transformer v2 on OPS-SAT-AD, sampling=5 cohort, full dataset.

Changes from smoke test (run_transformer_smoke.py):
  * Uses sliding-window loader instead of pad/truncate whole segments.
    This eliminates the length-leakage confound (see src/data/loader.py).
  * Filters to sampling=5 only (homogeneous time-step cohort).
  * Full dataset: all ~830 normal train segments → ~N windows for training.
  * Segment-level evaluation: per-window scores aggregated with max()
    back to segment level before computing metrics. This matches the
    evaluation protocol of the OPS-SAT-AD benchmark (1 label per segment).
  * 30 epochs with cosine LR schedule.

Usage:
    python experiments/s2_transformer/run_transformer_v2.py
    python experiments/s2_transformer/run_transformer_v2.py --epochs 30 --d_model 128
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import torch
import torch.nn as nn
import mlflow

from src.data.loader import (
    make_sliding_windows,
    segment_scores_from_windows,
    WindowDataset,
)
from src.evaluation.metrics import compute_metrics, metrics_table
from src.models.transformer_ad import (
    TransformerReconstructionAD,
    build_dataloader,
    train_epoch,
    window_reconstruction_errors,
)


# ── Config ────────────────────────────────────────────────────────────────────

MLFLOW_EXPERIMENT = "s2_transformer_opssat"

DEFAULT_CFG = dict(
    window_size  = 64,
    stride       = 32,
    sampling     = 5,
    d_model      = 64,
    n_heads      = 4,
    n_layers     = 2,
    d_ff         = 128,
    dropout      = 0.1,
    lr           = 1e-3,
    batch_size   = 64,
    epochs       = 30,
    threshold_p  = 95,   # percentile of train-normal window errors for threshold
    score_agg    = "max",  # "max" or "mean" for segment-level aggregation
    seed         = 42,
)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    train_ds: WindowDataset,
    test_ds: WindowDataset,
    cfg: dict,
    device: torch.device,
) -> dict:
    """
    Full evaluation:
    1. Compute window-level reconstruction errors on train-normal windows
       → derive threshold at percentile p.
    2. Compute window-level errors on all test windows.
    3. Aggregate to segment level with max().
    4. Threshold → binary predictions → metrics.
    """
    # Train errors (only normal windows — same as training distribution)
    train_errors = window_reconstruction_errors(
        model, train_ds.windows, cfg["batch_size"], device
    )
    threshold = float(np.percentile(train_errors, cfg["threshold_p"]))

    # Test window errors
    test_window_errors = window_reconstruction_errors(
        model, test_ds.windows, cfg["batch_size"], device
    )

    # Aggregate to segment level
    seg_ids_test, seg_scores = segment_scores_from_windows(
        test_window_errors, test_ds.seg_ids, agg=cfg["score_agg"]
    )

    # Ground-truth segment labels (one per unique segment id)
    seg_labels = np.array([
        test_ds.labels[test_ds.seg_ids == sid][0] for sid in seg_ids_test
    ])

    preds = (seg_scores > threshold).astype(int)
    metrics = compute_metrics(seg_labels, preds, seg_scores)
    metrics["threshold"] = threshold
    metrics["n_test_segments"] = len(seg_ids_test)
    metrics["n_test_windows"]  = len(test_window_errors)
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def run(cfg: dict):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cpu")
    print(f"\nDevice: {device}  |  PyTorch {torch.__version__}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\nLoading windows  (sampling={cfg['sampling']}, "
          f"window_size={cfg['window_size']}, stride={cfg['stride']}) ...")

    train_ds = make_sliding_windows(
        window_size          = cfg["window_size"],
        stride               = cfg["stride"],
        sampling_rate_filter = cfg["sampling"],
        train_normal_only    = True,
    )
    test_ds = make_sliding_windows(
        window_size          = cfg["window_size"],
        stride               = cfg["stride"],
        sampling_rate_filter = cfg["sampling"],
        split                = "test",
    )

    n_train_segs = len(np.unique(train_ds.seg_ids))
    n_test_segs  = len(np.unique(test_ds.seg_ids))
    anomaly_rate = test_ds.labels.mean()

    print(f"Train  : {len(train_ds):>6} windows  from {n_train_segs} normal segments")
    print(f"Test   : {len(test_ds):>6} windows  from {n_test_segs} segments  "
          f"(anomaly rate: {anomaly_rate:.1%})")

    train_dl = build_dataloader(train_ds.windows, cfg["batch_size"], shuffle=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = TransformerReconstructionAD(
        seq_len  = cfg["window_size"],
        d_model  = cfg["d_model"],
        n_heads  = cfg["n_heads"],
        n_layers = cfg["n_layers"],
        d_ff     = cfg["d_ff"],
        dropout  = cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params : {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01
    )

    # ── MLflow ─────────────────────────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="v2_sliding_windows_full"):
        mlflow.log_params({
            **cfg,
            "n_params":        n_params,
            "n_train_windows": len(train_ds),
            "n_train_segments": n_train_segs,
            "n_test_segments":  n_test_segs,
            "loader":          "sliding_windows_v2",
        })

        # ── Train ──────────────────────────────────────────────────────────────
        print(f"\n{'Epoch':>5} {'Train MSE':>12} {'LR':>10} {'Time':>8}")
        print("─" * 40)
        losses = []
        t0 = time.time()

        for epoch in range(1, cfg["epochs"] + 1):
            t_ep = time.time()
            loss = train_epoch(model, train_dl, optimizer, device)
            scheduler.step()
            losses.append(loss)
            lr_now = scheduler.get_last_lr()[0]
            mlflow.log_metrics({"train_mse": loss, "lr": lr_now}, step=epoch)
            print(f"{epoch:>5}  {loss:>12.6f}  {lr_now:>10.2e}  {time.time()-t_ep:>7.2f}s")

        total_time = time.time() - t0

        # ── Evaluate ───────────────────────────────────────────────────────────
        print("\nEvaluating (segment-level, score_agg=max) ...")
        metrics = evaluate(model, train_ds, test_ds, cfg, device)
        mlflow.log_metrics({**metrics, "train_time_s": total_time})

        # ── Report ─────────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Pipeline : sliding windows  (no length leakage)")
        print(f"  Sampling : {cfg['sampling']}s  |  window_size={cfg['window_size']}  stride={cfg['stride']}")
        print(f"  Train    : {len(train_ds)} windows / {n_train_segs} segs  ({total_time:.1f}s)")
        print(f"  Eval     : {metrics['n_test_windows']} windows → {metrics['n_test_segments']} segments")
        print(f"  Threshold: p{cfg['threshold_p']} of train-normal errors = {metrics['threshold']:.6f}")
        print()
        print(f"  {'Metric':<14} {'This run':>10}  {'OCSVM baseline':>16}")
        print(f"  {'-'*44}")
        baselines = {"precision":0.656, "recall":0.726, "f1":0.689, "f05":0.669, "auc_roc":0.800}
        for k in ["precision", "recall", "f1", "f05", "auc_roc"]:
            v    = metrics[k]
            ref  = baselines.get(k, float("nan"))
            diff = v - ref
            mark = "▲" if diff > 0 else "▼"
            print(f"  {k:<14} {v:>10.3f}  {ref:>10.3f}  {mark}{abs(diff):.3f}")
        print(f"{'='*60}")

        loss_drop = (losses[0] - losses[-1]) / (losses[0] + 1e-10)
        print(f"\n  Loss: {losses[0]:.6f} → {losses[-1]:.6f}  (drop {loss_drop:.1%})")
        print(f"  MLflow: mlflow ui --backend-store-uri mlruns/")


def parse_args():
    p = argparse.ArgumentParser()
    for k, v in DEFAULT_CFG.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
