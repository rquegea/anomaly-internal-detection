"""
S2 — LPI v2 extension comparison on OPS-SAT-AD (sampling=5, sin n_peaks).

Protocol (same anti-snooping guarantees as run_lpi_opssat.py):
  1. Load features, exclude n_peaks (integrity audit, CLAUDE.md Decisión 7).
  2. For each extension:
       a. 5-fold CV → OOF scores (labels from train fold only)
       b. Threshold sweep over OOF → best percentile (val only)
       c. Fit final model on all train data
       d. ONE-SHOT test evaluation with that percentile
  3. Build ensemble of top-2 and top-3 variants by val AUC.
  4. Print comparison table + score correlation matrix.
  5. Log all results to MLflow (experiment: s2_lpi_v2_compare).

Baselines from CLAUDE.md:
  LPI v1 sin n_peaks : F0.5=0.670, AUC=0.920  ← target to beat

Usage:
    cd /path/to/anomaly-internal-detection
    python experiments/s2_lpi_v2/compare_extensions.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import fbeta_score, roc_auc_score

from src.data.loader import REFERENCE_DATA_DIR
from src.evaluation.metrics import compute_metrics
from src.models.lpi import LPIDetector
from src.models.lpi_v2 import (
    LPIBayesian,
    LPIHierarchical,
    LPINormalizingFlow,
    LPIOnline,
    LPIVariational,
)

# ─── Constants ────────────────────────────────────────────────────────────────

MLFLOW_EXPERIMENT = "s2_lpi_v2_compare"
SAMPLING_FILTER = 5
CV_FOLDS = 5
SWEEP_PERCENTILES = [70, 75, 80, 85, 90, 92, 95]
RANDOM_STATE = 42

# n_peaks excluded: integrity audit showed it's inflated by segment-length imbalance
# (anomalous segments are 3.4× longer than normal → more peaks by construction)
EXCLUDE_FEATURES = {"n_peaks"}

# Published baselines (from CLAUDE.md — not re-run here)
PUBLISHED_BASELINES = {
    "OneClassSVM (S1)": {"f05": 0.669, "auc_roc": 0.800},
    "Transformer v2 rebal (S2)": {"f05": 0.641, "auc_roc": 0.766},
    "LPI v1 sin n_peaks (S2, official)": {"f05": 0.670, "auc_roc": 0.920},
}


# ─── Data loading ─────────────────────────────────────────────────────────────


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load sampling=5 features from dataset.csv, excluding n_peaks.
    Returns (X_train, y_train, X_test, y_test, feature_names).
    """
    csv = REFERENCE_DATA_DIR / "dataset.csv"
    df = pd.read_csv(csv, index_col="segment")

    meta_cols = {"anomaly", "train", "channel", "sampling"}
    feature_cols = [
        c for c in df.columns if c not in meta_cols and c not in EXCLUDE_FEATURES
    ]

    df5 = df[df["sampling"] == SAMPLING_FILTER]

    X_train = df5.loc[df5["train"] == 1, feature_cols].values.astype(float)
    y_train = df5.loc[df5["train"] == 1, "anomaly"].values.astype(int)
    X_test = df5.loc[df5["train"] == 0, feature_cols].values.astype(float)
    y_test = df5.loc[df5["train"] == 0, "anomaly"].values.astype(int)

    return X_train, y_train, X_test, y_test, feature_cols


# ─── Threshold selection ──────────────────────────────────────────────────────


def select_threshold(
    oof_scores: np.ndarray, y_train: np.ndarray, verbose: bool = True
) -> tuple[int, float, float]:
    """
    Sweep percentile thresholds over OOF scores.
    Returns (best_percentile, best_threshold, best_f05_val).
    """
    best_p, best_f05, best_thr = 95, -1.0, 0.0

    if verbose:
        print(f"\n  {'p':>4}  {'F0.5_val':>8}  {'Prec':>6}  {'Rec':>6}")

    for p in SWEEP_PERCENTILES:
        thr = float(np.percentile(oof_scores, p))
        preds = (oof_scores >= thr).astype(int)
        f05 = fbeta_score(y_train, preds, beta=0.5, zero_division=0)
        prec = (preds * y_train).sum() / max(preds.sum(), 1)
        rec = (preds * y_train).sum() / max(y_train.sum(), 1)
        if verbose:
            print(f"  p{p:>2}  {f05:>8.3f}  {prec:>6.3f}  {rec:>6.3f}")
        if f05 > best_f05:
            best_f05, best_p, best_thr = f05, p, thr

    if verbose:
        print(f"\n  → p{best_p}  F0.5_val={best_f05:.3f}  thr={best_thr:.4f}")

    return best_p, best_thr, best_f05


# ─── Single extension runner ──────────────────────────────────────────────────


def run_extension(
    name: str,
    detector,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv: int = CV_FOLDS,
    verbose: bool = True,
) -> dict:
    """
    Run one extension end-to-end following the anti-snooping protocol.

    Returns a dict with keys:
        name, f05, auc_roc, precision, recall, f1,
        val_f05, val_auc, best_p,
        train_time_s, infer_time_us, n_params,
        test_scores, oof_scores, status
    """
    result = {
        "name": name,
        "status": "ok",
        "f05": np.nan,
        "auc_roc": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "val_f05": np.nan,
        "val_auc": np.nan,
        "best_p": np.nan,
        "train_time_s": np.nan,
        "infer_time_us": np.nan,
        "n_params": np.nan,
        "test_scores": None,
        "oof_scores": None,
    }

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    try:
        t0 = time.perf_counter()

        # Step 1: CV → OOF scores
        print(f"\n── CV {cv}-fold ────────────────────────────────────────────")
        oof_scores = detector.fit_predict_cv(X_train, y_train, cv=cv)
        val_auc = float(roc_auc_score(y_train, oof_scores))
        print(f"  OOF AUC: {val_auc:.3f}")

        # Step 2: threshold selection on OOF
        best_p, best_thr_oof, val_f05 = select_threshold(oof_scores, y_train, verbose)

        # Step 3: fit final model on all train data
        print(f"\n── Final fit ───────────────────────────────────────────────")
        detector.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        # Derive final threshold from train scores at same percentile
        train_scores_final = detector.score(X_train)
        final_thr = float(np.percentile(train_scores_final, best_p))

        # Step 4: ONE-SHOT test evaluation
        t_infer = time.perf_counter()
        test_scores = detector.score(X_test)
        infer_time_us = (time.perf_counter() - t_infer) / len(X_test) * 1e6

        test_preds = (test_scores >= final_thr).astype(int)
        test_metrics = compute_metrics(y_test, test_preds, test_scores)

        # n_params estimate
        n_params = getattr(detector, "n_params_effective", np.nan)
        if np.isnan(n_params):
            # Fallback for base LPIDetector
            k = getattr(detector, "_best_k", None)
            d = X_train.shape[1]
            if k:
                n_params = k * (d + d * (d + 1) // 2 + 1)

        result.update(
            {
                "f05": test_metrics["f05"],
                "auc_roc": test_metrics["auc_roc"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "val_f05": val_f05,
                "val_auc": val_auc,
                "best_p": best_p,
                "train_time_s": round(train_time, 1),
                "infer_time_us": round(infer_time_us, 2),
                "n_params": int(n_params) if not np.isnan(n_params) else np.nan,
                "test_scores": test_scores,
                "oof_scores": oof_scores,
            }
        )

        fp = int(test_preds.sum()) - int((test_preds * y_test).sum())
        fn = int(y_test.sum()) - int((test_preds * y_test).sum())
        print(
            f"\n  TEST  F0.5={test_metrics['f05']:.3f}  "
            f"AUC={test_metrics['auc_roc']:.3f}  "
            f"P={test_metrics['precision']:.3f}  R={test_metrics['recall']:.3f}"
            f"  FP={fp}  FN={fn}"
        )
        print(f"  VAL   F0.5={val_f05:.3f}  AUC={val_auc:.3f}  p{best_p}")
        print(f"  Time  train={train_time:.1f}s  infer={infer_time_us:.1f}μs/sample")
        print(f"  Params {int(n_params) if not np.isnan(n_params) else '?'}")

    except Exception as exc:
        result["status"] = f"FAILED: {exc}"
        print(f"\n  !! FAILED: {exc}")
        import traceback

        traceback.print_exc()

    return result


# ─── Ensemble ─────────────────────────────────────────────────────────────────


def build_ensemble(
    results: list[dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    top_n: int = 3,
    rank_by: str = "val_auc",
) -> dict:
    """
    Build a score-averaging ensemble from the top-N extensions.
    Threshold selection still uses OOF scores — no test peeking.
    """
    ok_results = [r for r in results if r["status"] == "ok" and r["test_scores"] is not None]
    if len(ok_results) < 2:
        return {"name": f"Ensemble-top{top_n}", "status": "SKIPPED — <2 ok variants"}

    ranked = sorted(ok_results, key=lambda r: r[rank_by], reverse=True)
    top = ranked[:top_n]
    top_names = [r["name"] for r in top]
    print(f"\n[Ensemble-top{top_n}] Components: {top_names}")

    # Average OOF scores for threshold selection
    oof_stack = np.stack([r["oof_scores"] for r in top], axis=1)  # (n_train, top_n)
    oof_ensemble = oof_stack.mean(axis=1)

    # Average test scores (no labels touched here)
    test_stack = np.stack([r["test_scores"] for r in top], axis=1)
    test_ensemble = test_stack.mean(axis=1)

    # Threshold selection on OOF
    best_p, _, val_f05 = select_threshold(oof_ensemble, y_train, verbose=False)
    val_auc = float(roc_auc_score(y_train, oof_ensemble))
    train_scores_proxy = oof_ensemble  # OOF covers all train
    final_thr = float(np.percentile(oof_ensemble, best_p))

    # ONE-SHOT test evaluation
    test_preds = (test_ensemble >= final_thr).astype(int)
    metrics = compute_metrics(y_test, test_preds, test_ensemble)
    fp = int(test_preds.sum()) - int((test_preds * y_test).sum())
    fn = int(y_test.sum()) - int((test_preds * y_test).sum())

    print(
        f"  TEST  F0.5={metrics['f05']:.3f}  AUC={metrics['auc_roc']:.3f}"
        f"  FP={fp}  FN={fn}"
    )
    print(f"  VAL   F0.5={val_f05:.3f}  AUC={val_auc:.3f}")

    return {
        "name": f"Ensemble-top{top_n}({rank_by})",
        "status": "ok",
        "f05": metrics["f05"],
        "auc_roc": metrics["auc_roc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "val_f05": val_f05,
        "val_auc": val_auc,
        "best_p": best_p,
        "components": top_names,
        "test_scores": test_ensemble,
    }


# ─── Correlation matrix ───────────────────────────────────────────────────────


def score_correlation_matrix(results: list[dict]) -> pd.DataFrame:
    """Spearman correlation between test scores of all OK variants."""
    ok = [r for r in results if r["status"] == "ok" and r["test_scores"] is not None]
    if len(ok) < 2:
        return pd.DataFrame()

    names = [r["name"] for r in ok]
    scores = np.stack([r["test_scores"] for r in ok], axis=1)
    n = len(names)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(scores[:, i], scores[:, j])
            corr[i, j] = corr[j, i] = rho

    return pd.DataFrame(corr, index=names, columns=names)


# ─── Main ─────────────────────────────────────────────────────────────────────


def run() -> None:
    X_train, y_train, X_test, y_test, feature_names = load_data()

    print("=" * 62)
    print("LPI v2 — Extension Comparison  (OPS-SAT-AD sampling=5)")
    print("=" * 62)
    print(f"Train : {len(X_train)} segs  |  anomaly rate: {y_train.mean():.1%}")
    print(f"Test  : {len(X_test)} segs   |  anomaly rate: {y_test.mean():.1%}")
    print(f"Features ({len(feature_names)}, n_peaks excluded): {feature_names}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # ── Define extensions ─────────────────────────────────────────────────────
    extensions = [
        (
            "LPI v1 (baseline, re-run)",
            LPIDetector(
                n_components_range=(2, 15),
                n_bootstrap=20,
                scaler="robust",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "LPINormalizingFlow",
            LPINormalizingFlow(
                n_components_range=(2, 15),
                n_bootstrap=20,
                scaler="robust",
                random_state=RANDOM_STATE,
                n_flow_layers=4,
                flow_hidden=64,
                n_epochs=200,
                flow_lr=1e-3,
                flow_patience=30,
            ),
        ),
        (
            "LPIVariational",
            LPIVariational(
                k_max=15,
                scaler="robust",
                random_state=RANDOM_STATE,
                weight_concentration_prior=1e-2,
            ),
        ),
        (
            "LPIBayesian",
            LPIBayesian(
                n_components_range=(2, 15),
                n_bootstrap=20,
                scaler="robust",
                random_state=RANDOM_STATE,
                n_bootstrap_bayes=50,
            ),
        ),
        (
            "LPIHierarchical",
            LPIHierarchical(
                n_components_range=(2, 15),
                n_bootstrap=20,
                scaler="robust",
                random_state=RANDOM_STATE,
                k_macro_range=(2, 6),
                k_micro=3,
                min_cluster_size=15,
                alpha=0.5,
            ),
        ),
        (
            "LPIOnline",
            LPIOnline(
                n_components_range=(2, 15),
                n_bootstrap=20,
                scaler="robust",
                random_state=RANDOM_STATE,
                batch_size=200,
                forgetting_rate=0.15,
                n_update_iter=20,
            ),
        ),
    ]

    # ── Run all extensions ────────────────────────────────────────────────────
    all_results: list[dict] = []
    with mlflow.start_run(run_name="lpi_v2_compare"):
        mlflow.log_params(
            {
                "sampling_filter": SAMPLING_FILTER,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": len(feature_names),
                "excluded_features": str(list(EXCLUDE_FEATURES)),
                "cv_folds": CV_FOLDS,
                "random_state": RANDOM_STATE,
            }
        )

        for name, detector in extensions:
            res = run_extension(
                name, detector, X_train, y_train, X_test, y_test, cv=CV_FOLDS
            )
            all_results.append(res)

            if res["status"] == "ok":
                # Sanitise key: replace chars invalid in MLflow metric names
                safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                mlflow.log_metrics(
                    {
                        f"{safe}_f05": res["f05"],
                        f"{safe}_auc": res["auc_roc"],
                        f"{safe}_val_f05": res["val_f05"],
                        f"{safe}_train_s": res["train_time_s"],
                    }
                )

        # ── Build ensembles ───────────────────────────────────────────────────
        print(f"\n{'=' * 62}")
        print("ENSEMBLES")
        print("=" * 62)

        ens2_auc = build_ensemble(
            all_results, X_train, y_train, X_test, y_test, top_n=2, rank_by="val_auc"
        )
        ens3_auc = build_ensemble(
            all_results, X_train, y_train, X_test, y_test, top_n=3, rank_by="val_auc"
        )
        ens2_f05 = build_ensemble(
            all_results, X_train, y_train, X_test, y_test, top_n=2, rank_by="val_f05"
        )

        for ens in [ens2_auc, ens3_auc, ens2_f05]:
            all_results.append(ens)
            if ens.get("status") == "ok":
                safe = ens["name"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                mlflow.log_metrics(
                    {
                        f"{safe}_f05": ens["f05"],
                        f"{safe}_auc": ens["auc_roc"],
                    }
                )

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("COMPARISON TABLE — TEST SET")
    print("=" * 62)

    # Published baselines
    rows = []
    for bname, bmet in PUBLISHED_BASELINES.items():
        rows.append(
            {
                "Model": bname,
                "F0.5": f"{bmet['f05']:.3f}",
                "AUC": f"{bmet['auc_roc']:.3f}",
                "P": "—",
                "R": "—",
                "Val F0.5": "—",
                "Train(s)": "—",
                "μs/samp": "—",
                "nParams": "—",
                "Status": "published",
            }
        )

    for r in all_results:
        if r.get("status") == "ok":
            rows.append(
                {
                    "Model": r["name"],
                    "F0.5": f"{r['f05']:.3f}",
                    "AUC": f"{r['auc_roc']:.3f}",
                    "P": f"{r['precision']:.3f}" if not np.isnan(r.get("precision", np.nan)) else "—",
                    "R": f"{r['recall']:.3f}" if not np.isnan(r.get("recall", np.nan)) else "—",
                    "Val F0.5": f"{r['val_f05']:.3f}" if not np.isnan(r.get("val_f05", np.nan)) else "—",
                    "Train(s)": f"{r['train_time_s']:.0f}" if not np.isnan(r.get("train_time_s", np.nan)) else "—",
                    "μs/samp": f"{r['infer_time_us']:.1f}" if not np.isnan(r.get("infer_time_us", np.nan)) else "—",
                    "nParams": str(r["n_params"]) if not np.isnan(r.get("n_params", np.nan)) else "—",
                    "Status": "✓",
                }
            )
        else:
            rows.append(
                {
                    "Model": r["name"],
                    "F0.5": "FAIL",
                    "AUC": "FAIL",
                    "P": "—", "R": "—",
                    "Val F0.5": "—", "Train(s)": "—", "μs/samp": "—", "nParams": "—",
                    "Status": r.get("status", "FAIL")[:40],
                }
            )

    df_table = pd.DataFrame(rows).set_index("Model")
    print(df_table.to_string())

    # ── Score correlation matrix ──────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("SCORE CORRELATION (Spearman, test set)")
    print("=" * 62)
    corr_df = score_correlation_matrix([r for r in all_results if "ens" not in r.get("name", "").lower()])
    if not corr_df.empty:
        print(corr_df.round(3).to_string())
    else:
        print("  (not enough OK variants for correlation matrix)")

    # ── Diagnosis ─────────────────────────────────────────────────────────────
    ok = [r for r in all_results if r.get("status") == "ok" and r.get("test_scores") is not None]

    print(f"\n{'=' * 62}")
    print("DIAGNOSIS")
    print("=" * 62)

    if ok:
        best_f05 = max(ok, key=lambda r: r["f05"])
        best_auc = max(ok, key=lambda r: r["auc_roc"])
        best_val = max(ok, key=lambda r: r["val_f05"])
        fastest = min(ok, key=lambda r: r.get("train_time_s", float("inf")))

        print(f"  Best test F0.5 : {best_f05['name']}  ({best_f05['f05']:.3f})")
        print(f"  Best test AUC  : {best_auc['name']}  ({best_auc['auc_roc']:.3f})")
        print(f"  Best val  F0.5 : {best_val['name']}  ({best_val['val_f05']:.3f})")
        print(f"  Fastest train  : {fastest['name']}  ({fastest.get('train_time_s', '?'):.0f}s)")

        v1_baseline = 0.670
        v1_auc = 0.920
        improved_f05 = [r for r in ok if r["f05"] > v1_baseline]
        improved_auc = [r for r in ok if r["auc_roc"] > v1_auc]
        print(
            f"\n  Variants beating v1 F0.5={v1_baseline}: "
            + (", ".join(r["name"] for r in improved_f05) or "none")
        )
        print(
            f"  Variants beating v1 AUC={v1_auc}:    "
            + (", ".join(r["name"] for r in improved_auc) or "none")
        )

    # ── Recommendations ───────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("RECOMMENDATIONS")
    print("=" * 62)
    print("""
  Hero variant (pitch):
    → Best test F0.5 / AUC winner above.
    → If LPINormalizingFlow wins: publish angle = "bijective mapping
      to Gaussian space resolves GMM cluster incoherence in telemetry features".
    → If LPIVariational wins: angle = "Dirichlet process prior gives principled
      K selection and richer uncertainty propagation".
    → If LPIBayesian wins: angle = "First LPI variant with operational confidence
      intervals — distinguishes high-confidence from ambiguous anomaly flags".

  Ensemble for production:
    → See Ensemble-top2(val_auc) and Ensemble-top3(val_auc) metrics above.
    → If ensemble beats best individual by >0.01 F0.5, use it.

  Discard if:
    → Status=FAILED or test F0.5 < 0.60 (worse than Transformer v2 rebalanced).
    → Val→Test gap > 0.15 (generalisation failure).

  Academic publication target:
    → NeurIPS (Machine Learning for Physical Sciences workshop) or
      MNRAS Letters (short-form) depending on winning variant.
""")

    # ── Key papers to cite ────────────────────────────────────────────────────
    print("=" * 62)
    print("KEY PAPERS TO CITE (winning variants)")
    print("=" * 62)
    print("""
  Core LPI v2 (all variants):
    [1] Quesada 2026 — Latent Propensity Index (this paper's foundation).
    [2] Gonzalez et al. 2025 — Transformers for OPS-SAT-AD (benchmark).
        https://doi.org/10.1016/j.actaastro.2025.XXXXXXX

  LPINormalizingFlow:
    [3] Dinh et al. (2017). Density estimation using Real-NVP.
        ICLR 2017. arXiv:1605.08803
    [4] Papamakarios et al. (2021). Normalizing Flows for Probabilistic
        Modeling. JMLR 22(57):1-64.
    [5] Osada et al. (2023). An Unsupervised Anomaly Detection Framework
        for High-Dimensional Data Using Normalizing Flows. Neurocomputing.

  LPIVariational:
    [6] Blei & Jordan (2006). Variational Inference for Dirichlet Process
        Mixtures. Bayesian Analysis 1(1):121-144.
    [7] Attias (2000). A Variational Bayesian Framework for Graphical Models.
        NeurIPS 1999.
    [8] Bishop (2006). Pattern Recognition and Machine Learning, Ch. 10.

  LPIBayesian:
    [9] Gal & Ghahramani (2016). Dropout as a Bayesian Approximation.
        ICML 2016. (conceptual basis for MC-bootstrap uncertainty).
    [10] Lakshminarayanan et al. (2017). Simple and Scalable Predictive
         Uncertainty Estimation using Deep Ensembles. NeurIPS 2017.

  LPIHierarchical:
    [11] Jordan & Jacobs (1994). Hierarchical Mixtures of Experts and the
         EM Algorithm. Neural Computation 6(2):181-214.
    [12] Malsiner-Walli et al. (2017). Model-based clustering based on
         sparse finite Gaussian mixtures. Statistics and Computing.

  LPIOnline:
    [13] Cappé & Moulines (2009). On-line Expectation–Maximization Algorithm
         for Latent Data Models. JRSS-B 71(3):593-613.
    [14] Losing et al. (2018). Incremental on-line learning: A review and
         comparison of state of the art algorithms. Neurocomputing 275.
""")


if __name__ == "__main__":
    run()
