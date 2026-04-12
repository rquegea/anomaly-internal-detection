"""
S2 — Threshold percentile sweep on OPS-SAT-AD, sampling=5 cohort.

Goal: find the optimal reconstruction-error threshold without data snooping.

Data split (3-way, NO test-set contact during threshold selection):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  segments.csv  train=1  (1594 segs, cohort sampling=5)              │
  │    normal (193 segs)  ──80/20──▶  train_normal  (154 segs)          │
  │                                   val_normal    ( 39 segs)          │
  │    anomaly (N segs)  ──sample──▶  val_anomaly   (rebalanced)        │
  │                       to match test anomaly rate (~15%)             │
  ├─────────────────────────────────────────────────────────────────────┤
  │  segments.csv  train=0  (529 segs, cohort sampling=5)               │
  │    test (72 segs) ─────────────────── NEVER TOUCHED                │
  │                                       until final step              │
  └─────────────────────────────────────────────────────────────────────┘

Val rebalancing rationale:
  Previous sweep used ALL train anomalies in val, producing anomaly_rate≈47%
  in val vs ≈15% in test.  A threshold calibrated on a 47%-anomaly population
  is too permissive for the real 15% population — it selects a low threshold
  that generates many FPs at test time.  Fix: subsample val anomaly segments
  so that val_anomaly_rate ≈ test_anomaly_rate.  The test split is loaded
  only to count its segment-level anomaly rate; its time-series values are
  never seen until the final one-shot evaluation.

  n_val_anomaly = round(n_val_normal * test_rate / (1 - test_rate))
  seed=42 for reproducibility.

Procedure:
  1. Load test split → compute target_anomaly_rate (segment-level, read-only).
  2. Build train_normal / rebalanced val from train split.
  3. Train reconstruction Transformer on train_normal windows.
  4. Compute train_normal reconstruction errors → error distribution.
  5. For each p in [70, 75, 80, 85, 90, 92, 95]:
       threshold = percentile(p, train_normal_errors)
       score val set with that threshold → Precision/Recall/F1/F0.5/AUC
       log as MLflow child run with tag phase=threshold_sweep
  6. Select best_p = argmax F0.5 on VALIDATION.
  7. Evaluate with best_p threshold on TEST (once, never again).
  8. Compare vs OCSVM baseline (F0.5=0.669) and print decision.

MLflow tags logged:
  val_anomaly_rate, test_anomaly_rate, n_val_anomaly_used, n_val_anomaly_discarded

Usage:
    python experiments/s2_transformer/run_threshold_sweep.py
    python experiments/s2_transformer/run_threshold_sweep.py --epochs 30 --val_frac 0.2
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
from src.evaluation.metrics import compute_metrics
from src.models.transformer_ad import (
    TransformerReconstructionAD,
    build_dataloader,
    train_epoch,
    window_reconstruction_errors,
)


# ── Config ────────────────────────────────────────────────────────────────────

MLFLOW_EXPERIMENT = "s2_transformer_opssat"
OCSVM_F05         = 0.669     # OCSVM baseline (S1)
OCSVM_AUC         = 0.800

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
    val_frac     = 0.2,    # fraction of normal train segments held out as val
    score_agg    = "max",
    seed         = 42,
)

SWEEP_PERCENTILES = [70, 75, 80, 85, 90, 92, 95]


# ── Data preparation ──────────────────────────────────────────────────────────

def build_splits(cfg: dict):
    """
    Build train_normal / val / test WindowDataset splits.

    Train split from the dataset is divided:
      - 80% of normal segments → train_normal (for model training)
      - 20% of normal segments + REBALANCED anomaly segments → val

    Val anomaly segments are subsampled so that val_anomaly_rate matches
    test_anomaly_rate (segment-level). This prevents threshold miscalibration
    caused by the large rate mismatch observed in the previous sweep
    (val≈47% anomaly vs test≈15% anomaly).

    The test split is loaded before building val solely to read its
    segment-level anomaly rate; its time-series values remain unseen
    until the final one-shot evaluation.

    Returns
    -------
    train_normal_windows : np.ndarray  (n_windows, window_size)
    val_ds               : WindowDataset
    test_ds              : WindowDataset
    split_info           : dict with rebalancing diagnostics for MLflow
    """
    rng = np.random.default_rng(cfg["seed"])

    # ── Step 1: load test to get target anomaly rate (segment-level) ──────────
    test_ds = make_sliding_windows(
        window_size          = cfg["window_size"],
        stride               = cfg["stride"],
        sampling_rate_filter = cfg["sampling"],
        split                = "test",
    )
    n_test_normal  = int(len(np.unique(test_ds.seg_ids[test_ds.labels == 0])))
    n_test_anomaly = int(len(np.unique(test_ds.seg_ids[test_ds.labels == 1])))
    target_anomaly_rate = n_test_anomaly / (n_test_anomaly + n_test_normal)

    # ── Step 2: load all train windows (normal + anomaly) ─────────────────────
    all_train = make_sliding_windows(
        window_size          = cfg["window_size"],
        stride               = cfg["stride"],
        sampling_rate_filter = cfg["sampling"],
        split                = "train",
        train_normal_only    = False,
    )

    # Identify unique segment ids by class
    normal_seg_ids  = np.unique(all_train.seg_ids[all_train.labels == 0])
    anomaly_seg_ids = np.unique(all_train.seg_ids[all_train.labels == 1])

    # 80/20 split of normal segments
    shuffled = rng.permutation(normal_seg_ids)
    n_train  = int(len(shuffled) * (1 - cfg["val_frac"]))
    train_normal_ids = shuffled[:n_train]
    val_normal_ids   = shuffled[n_train:]

    # Mask for training windows: only from train_normal_ids
    mask_train = np.isin(all_train.seg_ids, train_normal_ids)
    train_normal_windows = all_train.windows[mask_train]  # pure numpy, no label needed

    # ── Step 3: rebalance val anomaly segments to match test_anomaly_rate ─────
    n_val_normal         = len(val_normal_ids)
    n_val_anomaly_target = round(n_val_normal * target_anomaly_rate / (1 - target_anomaly_rate))
    n_available          = len(anomaly_seg_ids)
    n_val_anomaly_used   = min(n_val_anomaly_target, n_available)
    n_val_anomaly_discarded = n_available - n_val_anomaly_used

    val_anomaly_ids = rng.choice(anomaly_seg_ids, size=n_val_anomaly_used, replace=False)
    val_anomaly_rate = n_val_anomaly_used / (n_val_normal + n_val_anomaly_used)

    # Validation: val-normal + rebalanced train-anomaly
    val_seg_ids = np.concatenate([val_normal_ids, val_anomaly_ids])
    mask_val    = np.isin(all_train.seg_ids, val_seg_ids)
    val_ds = WindowDataset(
        windows     = all_train.windows[mask_val],
        labels      = all_train.labels[mask_val],
        seg_ids     = all_train.seg_ids[mask_val],
        window_size = cfg["window_size"],
    )

    split_info = {
        "target_anomaly_rate":    target_anomaly_rate,
        "val_anomaly_rate":       val_anomaly_rate,
        "n_val_normal_segs":      n_val_normal,
        "n_val_anomaly_used":     n_val_anomaly_used,
        "n_val_anomaly_discarded": n_val_anomaly_discarded,
        "n_test_normal_segs":     n_test_normal,
        "n_test_anomaly_segs":    n_test_anomaly,
    }

    return train_normal_windows, val_ds, test_ds, split_info


# ── Segment-level evaluation helper ──────────────────────────────────────────

def eval_at_threshold(
    window_errors: np.ndarray,
    ds: WindowDataset,
    threshold: float,
    score_agg: str,
) -> dict:
    """Score a WindowDataset at a given threshold → segment-level metrics."""
    unique_ids, seg_scores = segment_scores_from_windows(
        window_errors, ds.seg_ids, agg=score_agg
    )
    seg_labels = np.array([
        ds.labels[ds.seg_ids == sid][0] for sid in unique_ids
    ])
    preds = (seg_scores > threshold).astype(int)
    return compute_metrics(seg_labels, preds, seg_scores)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(cfg: dict):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cpu")
    print(f"\nDevice: {device}  |  PyTorch {torch.__version__}")

    # ── Build splits ──────────────────────────────────────────────────────────
    print("\nBuilding train/val/test splits ...")
    train_normal_windows, val_ds, test_ds, split_info = build_splits(cfg)

    si              = split_info
    n_val_normal_w  = int((val_ds.labels == 0).sum())
    n_val_anomaly_w = int((val_ds.labels == 1).sum())
    n_val_segs      = len(np.unique(val_ds.seg_ids))
    n_test_segs     = len(np.unique(test_ds.seg_ids))

    print(f"  train_normal : {len(train_normal_windows)} windows  "
          f"({si['n_val_normal_segs']} normal segs held out for val)")
    print(f"  val          : {len(val_ds)} windows  "
          f"({n_val_normal_w} normal + {n_val_anomaly_w} anomaly windows)  "
          f"→ {n_val_segs} segments")
    print(f"    val anomaly rate : {si['val_anomaly_rate']:.3f}  "
          f"(target from test: {si['target_anomaly_rate']:.3f})")
    print(f"    val anomaly segs : {si['n_val_anomaly_used']} used  "
          f"/ {si['n_val_anomaly_used'] + si['n_val_anomaly_discarded']} available  "
          f"({si['n_val_anomaly_discarded']} discarded)")
    print(f"  test         : {len(test_ds)} windows  → {n_test_segs} segments  "
          f"[LOCKED until final eval]")

    train_dl = build_dataloader(train_normal_windows, cfg["batch_size"], shuffle=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = TransformerReconstructionAD(
        seq_len  = cfg["window_size"],
        d_model  = cfg["d_model"],
        n_heads  = cfg["n_heads"],
        n_layers = cfg["n_layers"],
        d_ff     = cfg["d_ff"],
        dropout  = cfg["dropout"],
    ).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01
    )

    print(f"\nTraining ({n_params:,} params, {cfg['epochs']} epochs) ...")
    t0 = time.time()
    losses = []
    for epoch in range(1, cfg["epochs"] + 1):
        loss = train_epoch(model, train_dl, optimizer, device)
        scheduler.step()
        losses.append(loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}  MSE={loss:.6f}")
    train_time = time.time() - t0
    print(f"  done in {train_time:.1f}s  "
          f"(loss {losses[0]:.4f} → {losses[-1]:.4f}, "
          f"drop {(losses[0]-losses[-1])/losses[0]:.1%})")

    # ── Pre-compute all window errors (do this once, not per threshold) ───────
    print("\nComputing reconstruction errors ...")
    train_errors = window_reconstruction_errors(
        model, train_normal_windows, cfg["batch_size"], device
    )
    val_errors  = window_reconstruction_errors(
        model, val_ds.windows, cfg["batch_size"], device
    )
    test_errors = window_reconstruction_errors(
        model, test_ds.windows, cfg["batch_size"], device
    )
    print(f"  train-normal error stats: "
          f"min={train_errors.min():.4f}  "
          f"p50={np.median(train_errors):.4f}  "
          f"p95={np.percentile(train_errors,95):.4f}  "
          f"max={train_errors.max():.4f}")

    # ── Threshold sweep on VALIDATION ─────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print(f"\n{'─'*72}")
    print(f"{'p':>4}  {'Thresh':>10}  {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'F0.5':>7} {'AUC':>7}  val-set")
    print(f"{'─'*72}")

    sweep_results: list[dict] = []

    with mlflow.start_run(run_name="threshold_sweep") as parent_run:
        mlflow.set_tag("phase", "threshold_sweep")
        mlflow.log_params({**cfg, "n_params": n_params,
                           "n_train_windows": len(train_normal_windows),
                           "n_val_segments": n_val_segs,
                           "n_test_segments": n_test_segs})
        mlflow.log_metrics({
            "val_anomaly_rate":       si["val_anomaly_rate"],
            "test_anomaly_rate":      si["target_anomaly_rate"],
            "n_val_anomaly_used":     float(si["n_val_anomaly_used"]),
            "n_val_anomaly_discarded": float(si["n_val_anomaly_discarded"]),
        })

        for p in SWEEP_PERCENTILES:
            threshold = float(np.percentile(train_errors, p))
            vm = eval_at_threshold(val_errors, val_ds, threshold, cfg["score_agg"])

            row = {"percentile": p, "threshold": threshold, **vm}
            sweep_results.append(row)

            with mlflow.start_run(run_name=f"sweep_p{p}", nested=True):
                mlflow.set_tag("phase", "threshold_sweep")
                mlflow.log_params({"percentile": p, "threshold": threshold})
                mlflow.log_metrics({k: v for k, v in vm.items()
                                    if isinstance(v, float)})

            print(f"  p{p:<2}  {threshold:>10.6f}  "
                  f"{vm['precision']:>7.3f} {vm['recall']:>7.3f} "
                  f"{vm['f1']:>7.3f} {vm['f05']:>7.3f} {vm['auc_roc']:>7.3f}")

    print(f"{'─'*72}")

    # ── Select best threshold by val F0.5 ─────────────────────────────────────
    best = max(sweep_results, key=lambda r: r["f05"])
    best_p   = best["percentile"]
    best_thr = best["threshold"]

    print(f"\nBest on validation: p{best_p}  "
          f"F0.5={best['f05']:.3f}  threshold={best_thr:.6f}")

    # ── Final evaluation on TEST (one-shot) ───────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION (first and only contact with test set)")
    print(f"{'='*60}")

    test_m = eval_at_threshold(test_errors, test_ds, best_thr, cfg["score_agg"])

    # Log final to MLflow
    with mlflow.start_run(run_name=f"final_p{best_p}_test"):
        mlflow.set_tag("phase", "final_test")
        mlflow.log_params({**cfg, "best_percentile": best_p,
                           "best_threshold": best_thr, "n_params": n_params})
        mlflow.log_metrics(test_m)

    baselines = {"precision": 0.656, "recall": 0.726, "f1": 0.689,
                 "f05": OCSVM_F05, "auc_roc": OCSVM_AUC}

    print(f"\n  Threshold : p{best_p} of train-normal errors = {best_thr:.6f}")
    print(f"\n  {'Metric':<14} {'Test':>8}  {'OCSVM':>8}  {'Δ':>8}")
    print(f"  {'─'*44}")
    for k in ["precision", "recall", "f1", "f05", "auc_roc"]:
        v    = test_m[k]
        ref  = baselines[k]
        diff = v - ref
        sym  = "▲" if diff >= 0 else "▼"
        print(f"  {k:<14} {v:>8.3f}  {ref:>8.3f}  {sym}{abs(diff):.3f}")

    print(f"\n{'='*60}")

    # ── Decision ──────────────────────────────────────────────────────────────
    beats_baseline = test_m["f05"] >= OCSVM_F05

    if beats_baseline:
        print(f"\n  ✓  F0.5 = {test_m['f05']:.3f} ≥ OCSVM {OCSVM_F05}  —  BASELINE BEATEN")
        print(f"  Threshold sweep sufficient. No Optuna needed for sampling=5 cohort.")
        print(f"  Recommended next step: cohort sampling=1 (window_size=256).")
    else:
        gap = OCSVM_F05 - test_m["f05"]
        print(f"\n  ✗  F0.5 = {test_m['f05']:.3f} < OCSVM {OCSVM_F05}  (gap {gap:.3f})")
        print(f"  Optuna needed — current architecture/training insufficient.")
        print(f"  Suggested search space: d_model, n_layers, epochs, lr, stride.")

    print(f"\n  MLflow: mlflow ui --backend-store-uri mlruns/")
    return test_m


def parse_args():
    p = argparse.ArgumentParser()
    for k, v in DEFAULT_CFG.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
