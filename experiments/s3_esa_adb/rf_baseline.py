"""
S3 — Random Forest supervised baseline on ESA-AD Mission1  (Quesada 2026)

Trains on train=1, evaluates on train=0. Uses the same 16-feature set as
the LPINormalizingFlow ensemble (n_peaks and gaps_squared excluded).

Outputs:
  • Feature importance ranking (MDI + permutation)
  • AUC-ROC and F0.5 on test set (supervised ceiling for the problem)
  • Comparison table vs S2 published claims

Usage:
  python experiments/s3_esa_adb/rf_baseline.py \\
      --data_path reference/data/esa_mission1_ch47/dataset.csv

  # Single-channel run (if dataset.csv covers multiple channels):
  python experiments/s3_esa_adb/rf_baseline.py \\
      --data_path reference/data/esa_mission1/dataset.csv \\
      --channel   channel_47
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ─── Constants (mirror run_nf_ensemble_s3.py) ─────────────────────────────────

EXCLUDE_FEATURES: set[str] = {"n_peaks", "gaps_squared"}
EXPECTED_N_FEATURES = 16

RF_PARAMS = dict(
    n_estimators=500,
    max_features="sqrt",
    class_weight="balanced",   # handles anomaly imbalance without resampling
    n_jobs=-1,
    random_state=42,
)

# Threshold sweep over predict_proba scores (same percentiles as NF ensemble)
SWEEP_PERCENTILES = [70, 75, 80, 85, 88, 90, 92, 95]

S2_BASELINES = {
    "OCSVM (S1, OPS-SAT-AD)":                      {"f05": 0.669, "auc": 0.800},
    "LPI v1 sin n_peaks (S2, OPS-SAT-AD)":          {"f05": 0.670, "auc": 0.920},
    "NF ensemble median 5 seeds (S2, OPS-SAT-AD)":  {"f05": 0.871, "auc": 0.997},
}


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_data(
    data_path: Path,
    channel_filter: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(data_path, index_col="segment")

    if channel_filter:
        df = df[df["channel"] == channel_filter]
        if df.empty:
            available = pd.read_csv(data_path, index_col="segment")["channel"].unique()
            raise ValueError(
                f"--channel '{channel_filter}' not found.\n"
                f"Available: {sorted(available)}"
            )
        print(f"  Channel filter: {channel_filter}  ({len(df)} rows)")

    meta_cols = {"anomaly", "train", "channel", "sampling"}
    feature_cols = [
        c for c in df.columns if c not in meta_cols and c not in EXCLUDE_FEATURES
    ]
    if len(feature_cols) != EXPECTED_N_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_N_FEATURES} features, got {len(feature_cols)}.\n"
            f"  Excluded : {sorted(EXCLUDE_FEATURES)}\n"
            f"  Got      : {feature_cols}"
        )

    X_train = df.loc[df["train"] == 1, feature_cols].values.astype(float)
    y_train = df.loc[df["train"] == 1, "anomaly"].values.astype(int)
    X_test  = df.loc[df["train"] == 0, feature_cols].values.astype(float)
    y_test  = df.loc[df["train"] == 0, "anomaly"].values.astype(int)

    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test  = np.where(np.isinf(X_test),  np.nan, X_test)
    train_medians = np.nanmedian(X_train, axis=0)
    for j in range(X_train.shape[1]):
        X_train[np.isnan(X_train[:, j]), j] = train_medians[j]
        X_test [np.isnan(X_test [:, j]), j] = train_medians[j]

    print(f"  Features : {len(feature_cols)}")
    print(f"  Train    : {len(X_train):,} segs  anomaly rate {y_train.mean():.2%}")
    print(f"  Test     : {len(X_test):,}  segs  anomaly rate {y_test.mean():.2%}")

    if y_test.sum() == 0:
        raise RuntimeError("Test set has 0 anomalies — AUC undefined.")

    return X_train, y_train, X_test, y_test, feature_cols


# ─── Threshold sweep ──────────────────────────────────────────────────────────

def best_threshold(scores_train: np.ndarray, y_train: np.ndarray) -> tuple[int, float]:
    """Pick percentile that maximises F0.5 on the training scores."""
    best_p, best_f05 = SWEEP_PERCENTILES[-1], -1.0
    for p in SWEEP_PERCENTILES:
        thr = float(np.percentile(scores_train, p))
        f05 = fbeta_score(y_train, (scores_train >= thr).astype(int),
                          beta=0.5, zero_division=0)
        if f05 > best_f05:
            best_f05, best_p = f05, p
    return best_p, best_f05


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(data_path: Path, channel_filter: str | None) -> None:
    print(f"\n{'='*60}")
    print("Random Forest Supervised Baseline — S3 ESA-Mission1")
    print(f"{'='*60}")
    print(f"  data_path : {data_path}")
    print(f"  RF params : {RF_PARAMS}\n")

    X_train, y_train, X_test, y_test, feature_cols = load_data(data_path, channel_filter)

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    print(f"\n  Fit time: {elapsed:.1f}s")

    # ── Scores & threshold ────────────────────────────────────────────────────
    # Use anomaly-class probability as the anomaly score
    train_scores = rf.predict_proba(X_train)[:, 1]
    test_scores  = rf.predict_proba(X_test) [:, 1]

    best_p, train_f05 = best_threshold(train_scores, y_train)
    threshold = float(np.percentile(train_scores, best_p))
    test_preds = (test_scores >= threshold).astype(int)

    test_f05  = fbeta_score(y_test, test_preds, beta=0.5, zero_division=0)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec  = recall_score(y_test, test_preds, zero_division=0)
    test_auc  = roc_auc_score(y_test, test_scores)
    test_ap   = average_precision_score(y_test, test_scores)

    # ── Feature importance — MDI (fast) ───────────────────────────────────────
    mdi = rf.feature_importances_
    mdi_order = np.argsort(mdi)[::-1]

    # ── Feature importance — permutation (slower but unbiased) ────────────────
    print("\n  Computing permutation importance on test set (n_repeats=20)...")
    t1 = time.perf_counter()
    perm = permutation_importance(
        rf, X_test, y_test,
        scoring="roc_auc",
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )
    print(f"  Permutation done in {time.perf_counter()-t1:.1f}s")

    perm_mean  = perm.importances_mean
    perm_std   = perm.importances_std
    perm_order = np.argsort(perm_mean)[::-1]

    # ── Tables ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TABLE 1 — Feature Importance (MDI + Permutation AUC-ROC drop)")
    print(f"{'='*60}")
    print(f"\n  {'Rank':>4}  {'Feature':<22}  {'MDI':>7}  {'Perm mean':>10}  {'Perm std':>9}  {'Rank Δ':>7}")
    print(f"  {'─'*4}  {'─'*22}  {'─'*7}  {'─'*10}  {'─'*9}  {'─'*7}")

    mdi_rank  = {f: i for i, f in enumerate([feature_cols[i] for i in mdi_order],  1)}
    perm_rank = {f: i for i, f in enumerate([feature_cols[i] for i in perm_order], 1)}

    for rank, idx in enumerate(perm_order, 1):
        feat  = feature_cols[idx]
        delta = mdi_rank[feat] - perm_rank[feat]
        delta_str = f"{delta:+d}" if delta != 0 else "  ="
        print(
            f"  {rank:>4}  {feat:<22}  {mdi[idx]:>7.4f}  "
            f"{perm_mean[idx]:>10.4f}  {perm_std[idx]:>9.4f}  {delta_str:>7}"
        )

    print(f"\n  (Rank Δ = MDI rank − Permutation rank; positive = MDI overestimates)")

    print(f"\n{'='*60}")
    print("TABLE 2 — RF Test Metrics vs S2 Baselines")
    print(f"{'='*60}")
    print(f"\n  {'Model':<42}  {'F0.5':>6}  {'AUC':>6}")
    print(f"  {'─'*42}  {'─'*6}  {'─'*6}")
    for name, m in S2_BASELINES.items():
        print(f"  {name:<42}  {m['f05']:>6.3f}  {m['auc']:>6.3f}")

    run_tag = channel_filter or "all_channels"
    rf_label = f"RandomForest supervised (ESA-M1 {run_tag})"
    print(f"  {rf_label:<42}  {test_f05:>6.3f}  {test_auc:>6.3f}  ← supervised ceiling")

    print(f"\n{'='*60}")
    print("RF TEST METRICS (full detail)")
    print(f"{'='*60}")
    print(f"  Threshold percentile : p{best_p}  (selected on train scores)")
    print(f"  Threshold value      : {threshold:.4f}")
    print(f"  Precision            : {test_prec:.3f}")
    print(f"  Recall               : {test_rec:.3f}")
    print(f"  F0.5                 : {test_f05:.3f}")
    print(f"  AUC-ROC              : {test_auc:.3f}")
    print(f"  AP (avg precision)   : {test_ap:.3f}")
    print(f"  Anomalies in test    : {int(y_test.sum())} / {len(y_test)}")

    gap_f05 = test_f05 - 0.871
    gap_auc = test_auc - 0.997
    print(f"\n  Gap vs NF ensemble S2:  F0.5 {gap_f05:+.3f}   AUC {gap_auc:+.3f}")
    print(f"  (negative gap = LPI unsupervised is below the supervised ceiling, expected)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RF supervised baseline on ESA-AD Mission1 dataset.csv",
    )
    parser.add_argument(
        "--data_path", required=True,
        help="Path to dataset.csv produced by prepare_mission1.py",
    )
    parser.add_argument(
        "--channel", default=None, metavar="CHANNEL_NAME",
        help="Restrict to a single channel (e.g. 'channel_47'). Default: all.",
    )
    args = parser.parse_args()
    run(Path(args.data_path), args.channel)


if __name__ == "__main__":
    main()
