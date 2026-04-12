"""
S2 — LPI (Latent Propensity Index) experiment on OPS-SAT-AD features.

Protocol (mirrors the Transformer threshold-sweep protocol):
  1. Filter dataset.csv to sampling=5 cohort (consistent comparison basis).
  2. 5-fold CV on train split → out-of-fold (OOF) LPI scores.
     Labels in train are used for enrichment; test is never touched.
  3. Threshold selection: sweep percentiles over OOF scores, pick p that
     maximises F0.5 on OOF. No snooping into test.
  4. Fit final LPI on full train split.
  5. Apply same percentile to final model's train scores → absolute threshold.
  6. Evaluate on test set (one-shot). Log to MLflow.

Semi-supervised note:
  LPI is not purely unsupervised. It requires anomaly labels to compute
  cluster enrichment (f_k = rare-class rate in cluster k). Labels come
  from the train split only. Test labels are sealed until step 6.

Baselines for comparison (from CLAUDE.md):
  - OneClassSVM  : F0.5=0.669, AUC=0.800  (S1, all sampling, full train)
  - Transformer v2 rebalanced: F0.5=0.641, AUC=0.766  (S2, sampling=5)

Usage:
    python experiments/s2_lpi/run_lpi_opssat.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import fbeta_score, roc_auc_score

from src.data.loader import REFERENCE_DATA_DIR
from src.evaluation.metrics import compute_metrics, metrics_table
from src.models.lpi import LPIDetector


MLFLOW_EXPERIMENT   = "s2_lpi_opssat"
SAMPLING_FILTER     = 5
CV_FOLDS            = 5
SWEEP_PERCENTILES   = [70, 75, 80, 85, 90, 92, 95]
RANDOM_STATE        = 42

# Published S2 baselines for comparison table
BASELINES = {
    "OneClassSVM (S1, all sampling)": {
        "precision": 0.656, "recall": 0.726, "f1": 0.689,
        "f05": 0.669, "auc_roc": 0.800,
    },
    "Transformer v2 rebalanced (S2, sampling=5)": {
        "precision": None, "recall": None, "f1": None,
        "f05": 0.641, "auc_roc": 0.766,
    },
}


def load_sampling5_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-computed features from dataset.csv, filtered to sampling=5 cohort.

    Returns
    -------
    X_train, y_train, X_test, y_test : numpy arrays
        Train/test split by the dataset's built-in `train` column.
        X shape: (n_segments, n_features); y shape: (n_segments,)
    """
    csv = REFERENCE_DATA_DIR / "dataset.csv"
    df  = pd.read_csv(csv, index_col="segment")

    meta_cols    = {"anomaly", "train", "channel", "sampling"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    df5 = df[df["sampling"] == SAMPLING_FILTER]

    X_train = df5.loc[df5["train"] == 1, feature_cols].values.astype(float)
    y_train = df5.loc[df5["train"] == 1, "anomaly"].values.astype(int)
    X_test  = df5.loc[df5["train"] == 0, feature_cols].values.astype(float)
    y_test  = df5.loc[df5["train"] == 0, "anomaly"].values.astype(int)

    return X_train, y_train, X_test, y_test


def select_threshold(oof_scores: np.ndarray, y_train: np.ndarray) -> tuple[int, float]:
    """
    Sweep percentile thresholds over OOF scores; return the percentile and
    absolute threshold that maximise F0.5 on training labels.
    """
    best_p, best_f05, best_threshold = 95, -1.0, 0.0

    print("\n── Threshold sweep on OOF scores ────────────────────────────")
    print(f"  {'p':>4}  {'F0.5':>6}  {'Precision':>10}  {'Recall':>8}")

    for p in SWEEP_PERCENTILES:
        threshold = float(np.percentile(oof_scores, p))
        preds     = (oof_scores >= threshold).astype(int)
        f05       = fbeta_score(y_train, preds, beta=0.5, zero_division=0)
        prec      = (preds * y_train).sum() / max(preds.sum(), 1)
        rec       = (preds * y_train).sum() / max(y_train.sum(), 1)
        print(f"  p{p:>2}  {f05:>6.3f}  {prec:>10.3f}  {rec:>8.3f}")

        if f05 > best_f05:
            best_f05      = f05
            best_p        = p
            best_threshold = threshold

    print(f"\n  → Best: p{best_p}  F0.5={best_f05:.3f}  threshold={best_threshold:.4f}")
    return best_p, best_threshold


def run() -> None:
    X_train, y_train, X_test, y_test = load_sampling5_features()

    print("=" * 62)
    print(f"LPI — OPS-SAT-AD  (sampling={SAMPLING_FILTER})")
    print("=" * 62)
    print(f"Train: {len(X_train)} segments  |  anomaly rate: {y_train.mean():.1%}")
    print(f"Test : {len(X_test)} segments  |  anomaly rate: {y_test.mean():.1%}")
    print(f"Features: {X_train.shape[1]}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="lpi_opssat_cv") as run:
        mlflow.set_tag("model", "lpi_opssat")
        mlflow.set_tag("phase", "s2")
        mlflow.log_params({
            "sampling_filter": SAMPLING_FILTER,
            "n_train": len(X_train),
            "n_test":  len(X_test),
            "n_features": X_train.shape[1],
            "cv_folds": CV_FOLDS,
            "n_components_min": 2,
            "n_components_max": 15,
            "n_bootstrap": 20,
            "scaler": "robust",
            "random_state": RANDOM_STATE,
        })

        # ── Step 1: 5-fold CV → OOF scores ───────────────────────────────────
        print("\n── 5-fold CV on train split ─────────────────────────────────")
        detector = LPIDetector(
            n_components_range=(2, 15),
            n_bootstrap=20,
            scaler="robust",
            random_state=RANDOM_STATE,
        )
        oof_scores = detector.fit_predict_cv(X_train, y_train, cv=CV_FOLDS)

        oof_auc = roc_auc_score(y_train, oof_scores)
        print(f"\n  OOF AUC-ROC: {oof_auc:.3f}")
        mlflow.log_metric("oof_auc_roc", oof_auc)

        # ── Step 2: threshold selection on OOF ───────────────────────────────
        best_p, oof_threshold = select_threshold(oof_scores, y_train)
        oof_preds = (oof_scores >= oof_threshold).astype(int)
        oof_metrics = compute_metrics(y_train, oof_preds, oof_scores)

        mlflow.log_metric("oof_best_percentile", best_p)
        mlflow.log_metrics({f"oof_{k}": v for k, v in oof_metrics.items()})

        print(f"\n  OOF metrics (p{best_p}):")
        for k, v in oof_metrics.items():
            print(f"    {k:>10}: {v:.3f}")

        # ── Step 3: fit final model on all train data ─────────────────────────
        print("\n── Fitting final model on full train split ──────────────────")
        final_detector = LPIDetector(
            n_components_range=(2, 15),
            n_bootstrap=20,
            scaler="robust",
            random_state=RANDOM_STATE,
        )
        final_detector.fit(X_train, y_train)
        mlflow.log_metric("final_k", final_detector.best_k)
        mlflow.log_param("final_enrichments", str(final_detector.enrichments.round(3).tolist()))

        # Derive threshold from final model's train scores at same percentile
        train_scores_final = final_detector.score(X_train)
        final_threshold = float(np.percentile(train_scores_final, best_p))
        mlflow.log_metric("final_threshold", final_threshold)

        # ── Step 4: ONE-SHOT test evaluation ─────────────────────────────────
        print("\n── Test evaluation (one-shot) ───────────────────────────────")
        test_scores = final_detector.score(X_test)
        test_preds  = (test_scores >= final_threshold).astype(int)
        test_metrics = compute_metrics(y_test, test_preds, test_scores)

        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("test_threshold_percentile", best_p)
        mlflow.log_metric("test_n_flagged", int(test_preds.sum()))
        mlflow.log_metric("test_n_anomaly", int(y_test.sum()))

        # ── Report ────────────────────────────────────────────────────────────
        print("\n" + "=" * 62)
        print("RESULTS — TEST SET")
        print("=" * 62)
        for k, v in test_metrics.items():
            print(f"  {k:>10}: {v:.3f}")

        print(f"\n  Threshold: p{best_p} = {final_threshold:.4f}")
        print(f"  Flagged {int(test_preds.sum())} / {len(y_test)} segments as anomalous")
        print(f"  True anomalies in test: {int(y_test.sum())}")
        print(f"  Final GMM K: {final_detector.best_k}")
        print(f"  Cluster enrichments: {final_detector.enrichments.round(3).tolist()}")

        print("\n" + "=" * 62)
        print("COMPARISON TABLE")
        print("=" * 62)

        all_results = {
            "OneClassSVM (S1)": BASELINES["OneClassSVM (S1, all sampling)"],
            "Transformer v2 rebal (S2)": BASELINES["Transformer v2 rebalanced (S2, sampling=5)"],
            "LPI (S2, sampling=5)": test_metrics,
        }

        # Build comparison DataFrame
        rows = []
        for model_name, m in all_results.items():
            row = {
                "Model": model_name,
                "Precision": f"{m['precision']:.3f}" if m["precision"] is not None else "—",
                "Recall":    f"{m['recall']:.3f}"    if m["recall"]    is not None else "—",
                "F1":        f"{m['f1']:.3f}"        if m["f1"]        is not None else "—",
                "F0.5":      f"{m['f05']:.3f}",
                "AUC-ROC":   f"{m['auc_roc']:.3f}",
            }
            rows.append(row)

        comparison_df = pd.DataFrame(rows).set_index("Model")
        print(comparison_df.to_string())

        print("\n── Diagnostic ───────────────────────────────────────────────")
        ocsvm_f05 = BASELINES["OneClassSVM (S1, all sampling)"]["f05"]
        ocsvm_auc = BASELINES["OneClassSVM (S1, all sampling)"]["auc_roc"]
        lpi_f05   = test_metrics["f05"]
        lpi_auc   = test_metrics["auc_roc"]
        lpi_prec  = test_metrics["precision"]
        lpi_rec   = test_metrics["recall"]

        print(f"  F0.5 vs OCSVM:       {lpi_f05:.3f} vs {ocsvm_f05:.3f}"
              f"  ({'▲' if lpi_f05 > ocsvm_f05 else '▼'} {abs(lpi_f05-ocsvm_f05):.3f})")
        print(f"  AUC  vs OCSVM:       {lpi_auc:.3f} vs {ocsvm_auc:.3f}"
              f"  ({'▲' if lpi_auc > ocsvm_auc else '▼'} {abs(lpi_auc-ocsvm_auc):.3f})")
        print(f"  Precision/Recall:    {lpi_prec:.3f} / {lpi_rec:.3f}")
        print(f"  Error profile:       "
              f"FP={int(test_preds.sum()) - int((test_preds * y_test).sum())}  "
              f"FN={int(y_test.sum()) - int((test_preds * y_test).sum())}")

        if lpi_f05 > ocsvm_f05:
            positioning = "modelo principal"
        elif lpi_f05 >= ocsvm_f05 - 0.05:
            positioning = "modelo complementario (gap < 0.05)"
        else:
            positioning = "descartar"

        print(f"\n  Recomendación:       {positioning}")
        print(f"\n  MLflow run:  {run.info.run_id}")
        print(f"  MLflow UI:   mlflow ui --backend-store-uri mlruns/")
        print("=" * 62)


if __name__ == "__main__":
    run()
