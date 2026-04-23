"""
S3 — LPINormalizingFlow Seed Ensemble on ESA-AD  (experiment tag: s3_nf_ensemble_v1)

Adapts the S2 NF ensemble pipeline (run_nf_seed_ensemble.py) for ESA-Mission1.
Key differences from S2:
  • --data_path CLI argument (no hardcoded path)
  • No SAMPLING_FILTER — ESA-M1 channels have ~90s/sample (not 5s)
  • MLflow experiment: s3_nf_ensemble_v1
  • Baselines table updated with S2 published claims

Protocol (identical to S2 — anti-snooping):
  1. Per seed: 5-fold CV on train → OOF scores, final fit, test scores
  2. Three ensemble strategies: mean / median / rank
  3. Threshold sweep on OOF ensemble only → ONE-SHOT test evaluation
  4. Bootstrap CI95 (B=1000)

Usage
─────
    # Step 1 — prepare dataset (single channel):
    python experiments/s3_esa_adb/prepare_mission1.py \\
        --data_dir /workspace/ESA-Mission1/ESA-Mission1 \\
        --out_dir  reference/data/esa_mission1_ch47 \\
        --channel  channel_47

    # Step 2 — run LPI ensemble:
    python experiments/s3_esa_adb/run_nf_ensemble_s3.py \\
        --data_path reference/data/esa_mission1_ch47/dataset.csv \\
        --channel   channel_47

    # Full Mission1 (all channels):
    python experiments/s3_esa_adb/prepare_mission1.py \\
        --data_dir /workspace/ESA-Mission1/ESA-Mission1 \\
        --out_dir  reference/data/esa_mission1
    python experiments/s3_esa_adb/run_nf_ensemble_s3.py \\
        --data_path reference/data/esa_mission1/dataset.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import fbeta_score, roc_auc_score

from src.evaluation.metrics import compute_metrics
from src.models.lpi_v2 import LPINormalizingFlow

# ─── Constants ────────────────────────────────────────────────────────────────

SEEDS = [0, 1, 42, 123, 999]

EXCLUDE_FEATURES: set[str] = {"n_peaks", "gaps_squared"}
EXPECTED_N_FEATURES = 16

CV_FOLDS = 5
SWEEP_PERCENTILES = [85, 88, 90, 92, 95]

BOOTSTRAP_N = 1000
BOOTSTRAP_MASTER_SEED = 42

MLFLOW_EXPERIMENT = "s3_nf_ensemble_v1"

NF_PARAMS: dict = dict(
    n_components_range=(2, 15),
    n_bootstrap=20,
    scaler="robust",
    n_flow_layers=4,
    flow_hidden=64,
    n_epochs=200,
    flow_lr=1e-3,
    flow_patience=30,
)

# S2 published baselines for cross-dataset comparison
S2_BASELINES = {
    "OCSVM (S1, OPS-SAT-AD)": {"f05": 0.669, "auc": 0.800},
    "LPI v1 sin n_peaks (S2, OPS-SAT-AD)": {"f05": 0.670, "auc": 0.920},
    "NF ensemble median 5 seeds (S2, OPS-SAT-AD)": {"f05": 0.871, "auc": 0.997},
}


# ─── Data ─────────────────────────────────────────────────────────────────────


def load_data(
    data_path: Path,
    channel_filter: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load ESA-Mission1 dataset.csv.

    No sampling filter — ESA-M1 has ~90s/sample, not 5s.
    Optional channel_filter restricts to a single channel for single-channel runs.
    """
    df = pd.read_csv(data_path, index_col="segment")

    if channel_filter:
        df = df[df["channel"] == channel_filter]
        if df.empty:
            available = pd.read_csv(data_path, index_col="segment")["channel"].unique()
            raise ValueError(
                f"--channel '{channel_filter}' not found. "
                f"Available: {sorted(available)}"
            )
        print(f"  Channel filter: {channel_filter}  ({len(df)} rows)")

    meta_cols = {"anomaly", "train", "channel", "sampling"}
    feature_cols = [
        c for c in df.columns if c not in meta_cols and c not in EXCLUDE_FEATURES
    ]

    n_feats = len(feature_cols)
    print(f"\n{'─'*60}")
    print(f"Feature set: {n_feats} features  (expected {EXPECTED_N_FEATURES})")
    if n_feats != EXPECTED_N_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_N_FEATURES} features, got {n_feats}.\n"
            f"  Excluded: {sorted(EXCLUDE_FEATURES)}\n"
            f"  Got: {feature_cols}"
        )
    print(f"  Excluded : {sorted(EXCLUDE_FEATURES)}")
    print(f"  Retained : {feature_cols}")

    X_train = df.loc[df["train"] == 1, feature_cols].values.astype(float)
    y_train = df.loc[df["train"] == 1, "anomaly"].values.astype(int)
    X_test  = df.loc[df["train"] == 0, feature_cols].values.astype(float)
    y_test  = df.loc[df["train"] == 0, "anomaly"].values.astype(int)

    # Replace inf/-inf, then impute NaN with train-column medians (no test leakage)
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test  = np.where(np.isinf(X_test),  np.nan, X_test)
    nan_counts = np.isnan(X_train).sum(axis=0)
    if nan_counts.any():
        print(f"\n  NaN/inf detected — imputing with train median:")
        for i, (feat, cnt) in enumerate(zip(feature_cols, nan_counts)):
            if cnt:
                print(f"    {feat}: {cnt} NaN in train")
    train_medians = np.nanmedian(X_train, axis=0)
    for j in range(X_train.shape[1]):
        mask_tr = np.isnan(X_train[:, j])
        if mask_tr.any():
            X_train[mask_tr, j] = train_medians[j]
        mask_te = np.isnan(X_test[:, j])
        if mask_te.any():
            X_test[mask_te, j] = train_medians[j]

    sampling_rates = sorted(df["sampling"].unique())
    print(f"  Sampling rates in dataset: {sampling_rates} s")
    print(f"  Train: {len(X_train)} segs  anomaly rate {y_train.mean():.1%}")
    print(f"  Test : {len(X_test)} segs   anomaly rate {y_test.mean():.1%}")

    if y_test.sum() == 0:
        raise RuntimeError(
            "Test set has 0 anomalies — evaluation is undefined. "
            "Choose a channel with anomalies in both H1 and H2 (check --scan output)."
        )

    return X_train, y_train, X_test, y_test, feature_cols


# ─── Threshold sweep ──────────────────────────────────────────────────────────


def select_threshold_oof(
    oof_scores: np.ndarray, y_train: np.ndarray
) -> tuple[int, float, list[tuple[int, float]]]:
    best_p, best_f05 = SWEEP_PERCENTILES[-1], -1.0
    rows: list[tuple[int, float]] = []
    for p in SWEEP_PERCENTILES:
        thr = float(np.percentile(oof_scores, p))
        preds = (oof_scores >= thr).astype(int)
        f05 = fbeta_score(y_train, preds, beta=0.5, zero_division=0)
        rows.append((p, f05))
        if f05 > best_f05:
            best_f05, best_p = f05, p
    return best_p, best_f05, rows


# ─── Single-seed training ─────────────────────────────────────────────────────


def train_single_seed(
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    print(f"\n{'─'*60}")
    print(f"Seed {seed}")
    print(f"{'─'*60}")

    t0 = time.perf_counter()

    detector = LPINormalizingFlow(**NF_PARAMS, random_state=seed)

    oof_scores = detector.fit_predict_cv(X_train, y_train, cv=CV_FOLDS)
    val_auc = float(roc_auc_score(y_train, oof_scores))

    best_p, val_f05, sweep = select_threshold_oof(oof_scores, y_train)

    detector.fit(X_train, y_train)
    test_scores = detector.score(X_test)

    oof_thr = float(np.percentile(oof_scores, best_p))
    test_preds = (test_scores >= oof_thr).astype(int)
    met = compute_metrics(y_test, test_preds, test_scores)

    elapsed = time.perf_counter() - t0

    fp = int(test_preds.sum()) - int((test_preds * y_test).sum())
    fn = int(y_test.sum()) - int((test_preds * y_test).sum())

    print(f"  OOF AUC={val_auc:.3f}  Val F0.5={val_f05:.3f}  threshold=p{best_p}")
    print(
        f"  Test F0.5={met['f05']:.3f}  AUC={met['auc_roc']:.3f}"
        f"  P={met['precision']:.3f}  R={met['recall']:.3f}"
        f"  FP={fp}  FN={fn}"
    )
    print(f"  Time: {elapsed:.1f}s")

    return {
        "seed": seed,
        "oof_scores": oof_scores,
        "test_scores": test_scores,
        "val_auc": val_auc,
        "val_f05": val_f05,
        "best_p": best_p,
        "test_f05": met["f05"],
        "test_auc": met["auc_roc"],
        "test_precision": met["precision"],
        "test_recall": met["recall"],
        "elapsed": elapsed,
    }


# ─── Normalisation helpers ────────────────────────────────────────────────────


def normalize_minmax(ref_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    lo, hi = float(ref_scores.min()), float(ref_scores.max())
    if hi <= lo:
        return np.zeros_like(scores, dtype=float)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)


def to_fractional_ranks(scores: np.ndarray) -> np.ndarray:
    n = len(scores)
    if n <= 1:
        return np.zeros(n, dtype=float)
    ranks = rankdata(scores, method="average")
    return (ranks - 1.0) / (n - 1.0)


# ─── Ensemble builder ─────────────────────────────────────────────────────────


def build_ensembles(
    seed_results: list[dict],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict]:
    oof_list  = [r["oof_scores"]  for r in seed_results]
    test_list = [r["test_scores"] for r in seed_results]

    strategies: dict[str, dict] = {}

    for strategy in ("mean", "median", "rank"):
        print(f"\n{'─'*60}")
        print(f"Ensemble strategy: {strategy}")
        print(f"{'─'*60}")

        if strategy in ("mean", "median"):
            norm_oof  = np.stack(
                [normalize_minmax(oof, oof) for oof in oof_list], axis=1
            )
            norm_test = np.stack(
                [normalize_minmax(oof, test) for oof, test in zip(oof_list, test_list)],
                axis=1,
            )
            if strategy == "mean":
                ens_oof  = norm_oof.mean(axis=1)
                ens_test = norm_test.mean(axis=1)
            else:
                ens_oof  = np.median(norm_oof, axis=1)
                ens_test = np.median(norm_test, axis=1)
        else:
            rank_oof  = np.stack([to_fractional_ranks(oof)  for oof in oof_list],  axis=1)
            rank_test = np.stack([to_fractional_ranks(test) for test in test_list], axis=1)
            ens_oof  = rank_oof.mean(axis=1)
            ens_test = rank_test.mean(axis=1)

        best_p, val_f05, sweep_rows = select_threshold_oof(ens_oof, y_train)
        val_auc = float(roc_auc_score(y_train, ens_oof))

        print(f"  Threshold sweep (OOF):")
        for p, f in sweep_rows:
            marker = " ←" if p == best_p else ""
            print(f"    p{p:>2}  F0.5={f:.3f}{marker}")

        ens_thr = float(np.percentile(ens_oof, best_p))
        test_preds = (ens_test >= ens_thr).astype(int)
        met = compute_metrics(y_test, test_preds, ens_test)

        fp = int(test_preds.sum()) - int((test_preds * y_test).sum())
        fn = int(y_test.sum()) - int((test_preds * y_test).sum())

        print(
            f"\n  TEST  F0.5={met['f05']:.3f}  AUC={met['auc_roc']:.3f}"
            f"  P={met['precision']:.3f}  R={met['recall']:.3f}"
            f"  FP={fp}  FN={fn}"
        )
        print(f"  VAL   F0.5={val_f05:.3f}  AUC={val_auc:.3f}  p{best_p}")

        strategies[strategy] = {
            "strategy": strategy,
            "ens_oof": ens_oof,
            "ens_test": ens_test,
            "threshold": ens_thr,
            "best_p": best_p,
            "val_f05": val_f05,
            "val_auc": val_auc,
            "test_f05": met["f05"],
            "test_auc": met["auc_roc"],
            "test_precision": met["precision"],
            "test_recall": met["recall"],
        }

    return strategies


# ─── Bootstrap CI ─────────────────────────────────────────────────────────────


def bootstrap_ci_metrics(
    ensemble_scores: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_MASTER_SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_test)

    f05_boots: list[float] = []
    auc_boots: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores_b = ensemble_scores[idx]
        y_b = y_test[idx]

        if len(np.unique(y_b)) < 2:
            continue

        preds_b = (scores_b >= threshold).astype(int)
        f05_boots.append(fbeta_score(y_b, preds_b, beta=0.5, zero_division=0))
        auc_boots.append(float(roc_auc_score(y_b, scores_b)))

    f05_arr = np.array(f05_boots)
    auc_arr = np.array(auc_boots)

    point_preds = (ensemble_scores >= threshold).astype(int)
    point_f05 = float(fbeta_score(y_test, point_preds, beta=0.5, zero_division=0))
    point_auc = float(roc_auc_score(y_test, ensemble_scores))

    return {
        "f05_point": point_f05,
        "f05_ci_lower": float(np.percentile(f05_arr, 2.5)),
        "f05_ci_upper": float(np.percentile(f05_arr, 97.5)),
        "f05_std": float(f05_arr.std()),
        "auc_point": point_auc,
        "auc_ci_lower": float(np.percentile(auc_arr, 2.5)),
        "auc_ci_upper": float(np.percentile(auc_arr, 97.5)),
        "auc_std": float(auc_arr.std()),
        "n_valid_boots": len(f05_boots),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def run(data_path: Path, channel_filter: str | None) -> None:
    X_train, y_train, X_test, y_test, feature_cols = load_data(data_path, channel_filter)

    run_tag = channel_filter or "all_channels"
    print(f"\n{'='*60}")
    print(f"LPINormalizingFlow Seed Ensemble — S3 ESA-Mission1  ({run_tag})")
    print(f"{'='*60}")
    print(f"Seeds: {SEEDS}")
    print(f"NF config: {NF_PARAMS}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"nf_ensemble_{run_tag}"):
        mlflow.log_params(
            {
                "seeds": str(SEEDS),
                "channel": run_tag,
                "data_path": str(data_path),
                "sampling_filter": "none (ESA-M1 ~90s/sample)",
                "n_train": len(X_train),
                "n_test": len(X_test),
                "anomaly_rate_train": f"{y_train.mean():.3f}",
                "anomaly_rate_test": f"{y_test.mean():.3f}",
                "n_features": len(feature_cols),
                "excluded_features": str(sorted(EXCLUDE_FEATURES)),
                "cv_folds": CV_FOLDS,
                "n_flow_layers": NF_PARAMS["n_flow_layers"],
                "flow_hidden": NF_PARAMS["flow_hidden"],
                "n_epochs": NF_PARAMS["n_epochs"],
                "bootstrap_n": BOOTSTRAP_N,
            }
        )

        # ── Step 1: Train all seeds ───────────────────────────────────────────
        print(f"\n{'='*60}")
        print("STEP 1 — Individual seed training")
        print(f"{'='*60}")

        seed_results: list[dict] = []
        for seed in SEEDS:
            res = train_single_seed(seed, X_train, y_train, X_test, y_test)
            seed_results.append(res)
            mlflow.log_metrics(
                {
                    f"seed{seed}_val_auc": res["val_auc"],
                    f"seed{seed}_val_f05": res["val_f05"],
                    f"seed{seed}_test_f05": res["test_f05"],
                    f"seed{seed}_test_auc": res["test_auc"],
                }
            )

        all_oof = np.stack([r["oof_scores"] for r in seed_results], axis=1)
        cross_seed_std = float(all_oof.std(axis=1).mean())
        print(f"\n  Cross-seed OOF score std: {cross_seed_std:.4f}")
        if cross_seed_std < 1e-6:
            print("  !! WARNING: all seeds produced identical OOF scores")
        mlflow.log_metric("cross_seed_oof_std", cross_seed_std)

        # ── Step 2: Build ensembles ───────────────────────────────────────────
        print(f"\n{'='*60}")
        print("STEP 2 — Ensemble strategies")
        print(f"{'='*60}")

        strategies = build_ensembles(seed_results, y_train, y_test)

        for name, strat in strategies.items():
            mlflow.log_metrics(
                {
                    f"ens_{name}_val_f05": strat["val_f05"],
                    f"ens_{name}_val_auc": strat["val_auc"],
                    f"ens_{name}_test_f05": strat["test_f05"],
                    f"ens_{name}_test_auc": strat["test_auc"],
                }
            )

        # ── Step 3: Bootstrap CI ──────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"STEP 3 — Bootstrap CI95  (B={BOOTSTRAP_N})")
        print(f"{'='*60}")

        ci_results: dict[str, dict] = {}
        for name, strat in strategies.items():
            print(f"\n  {name} ensemble ...")
            ci = bootstrap_ci_metrics(
                strat["ens_test"], y_test, strat["threshold"],
                n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_MASTER_SEED,
            )
            ci_results[name] = ci
            print(
                f"  F0.5  point={ci['f05_point']:.3f}  "
                f"CI95=[{ci['f05_ci_lower']:.3f}, {ci['f05_ci_upper']:.3f}]"
            )
            print(
                f"  AUC   point={ci['auc_point']:.3f}  "
                f"CI95=[{ci['auc_ci_lower']:.3f}, {ci['auc_ci_upper']:.3f}]"
            )
            mlflow.log_metrics(
                {
                    f"ens_{name}_f05_point": ci["f05_point"],
                    f"ens_{name}_f05_ci_lower": ci["f05_ci_lower"],
                    f"ens_{name}_f05_ci_upper": ci["f05_ci_upper"],
                    f"ens_{name}_auc_point": ci["auc_point"],
                    f"ens_{name}_auc_ci_lower": ci["auc_ci_lower"],
                    f"ens_{name}_auc_ci_upper": ci["auc_ci_upper"],
                }
            )

    # ── Tables ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TABLE 1 — Individual seeds (test set)")
    print(f"{'='*60}")

    seed_rows = []
    for r in seed_results:
        seed_rows.append(
            {
                "Seed": r["seed"],
                "Val F0.5": f"{r['val_f05']:.3f}",
                "Val AUC": f"{r['val_auc']:.3f}",
                "Test F0.5": f"{r['test_f05']:.3f}",
                "Test AUC": f"{r['test_auc']:.3f}",
                "Prec": f"{r['test_precision']:.3f}",
                "Rec": f"{r['test_recall']:.3f}",
                "p": r["best_p"],
                "Time(s)": f"{r['elapsed']:.0f}",
            }
        )

    seed_f05 = [r["test_f05"] for r in seed_results]
    seed_auc = [r["test_auc"] for r in seed_results]
    seed_rows.append(
        {
            "Seed": "mean±std",
            "Val F0.5": "—",
            "Val AUC": "—",
            "Test F0.5": f"{np.mean(seed_f05):.3f}±{np.std(seed_f05):.3f}",
            "Test AUC": f"{np.mean(seed_auc):.3f}±{np.std(seed_auc):.3f}",
            "Prec": "—",
            "Rec": "—",
            "p": "—",
            "Time(s)": "—",
        }
    )
    df_seeds = pd.DataFrame(seed_rows).set_index("Seed")
    print(df_seeds.to_string())

    print(f"\n{'='*60}")
    print("TABLE 2 — Ensemble strategies + S2 cross-dataset comparison")
    print(f"{'='*60}")

    ens_rows = []
    for bname, bmet in S2_BASELINES.items():
        ens_rows.append(
            {
                "Model": bname,
                "F0.5": f"{bmet['f05']:.3f}",
                "F0.5 CI95": "— (S2 dataset)",
                "AUC": f"{bmet['auc']:.3f}",
                "Val F0.5": "—",
                "Note": "S2 published",
            }
        )

    strategy_display = {
        "mean":   "NF ensemble (mean, S3)",
        "median": "NF ensemble (median, S3)",
        "rank":   "NF ensemble (rank, S3)",
    }
    for sname, display in strategy_display.items():
        strat = strategies[sname]
        ci = ci_results[sname]
        ens_rows.append(
            {
                "Model": display,
                "F0.5": f"{ci['f05_point']:.3f}",
                "F0.5 CI95": f"[{ci['f05_ci_lower']:.3f}, {ci['f05_ci_upper']:.3f}]",
                "AUC": f"{ci['auc_point']:.3f}",
                "Val F0.5": f"{strat['val_f05']:.3f}",
                "Note": f"p{strat['best_p']}",
            }
        )
    df_ens = pd.DataFrame(ens_rows).set_index("Model")
    print(df_ens.to_string())

    # ── Winner selection ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("WINNER SELECTION")
    print(f"{'='*60}")

    best_strat = max(
        strategies.keys(),
        key=lambda s: (strategies[s]["val_f05"], -ci_results[s]["f05_std"]),
    )
    winner_strat = strategies[best_strat]
    winner_ci = ci_results[best_strat]
    winner_name = strategy_display[best_strat]

    ensemble_f05 = winner_ci["f05_point"]

    print(f"\n  Best strategy (val F0.5 + stability): {winner_name}")
    print(f"  Val  F0.5: {winner_strat['val_f05']:.3f}  AUC: {winner_strat['val_auc']:.3f}")
    print(
        f"  Test F0.5: {ensemble_f05:.3f}  CI95=[{winner_ci['f05_ci_lower']:.3f}, {winner_ci['f05_ci_upper']:.3f}]"
    )
    print(
        f"  Test AUC:  {winner_ci['auc_point']:.3f}  CI95=[{winner_ci['auc_ci_lower']:.3f}, {winner_ci['auc_ci_upper']:.3f}]"
    )

    # ── Generalization check vs S2 ────────────────────────────────────────────
    s2_claim_f05 = 0.871
    print(f"\n{'='*60}")
    print("GENERALIZATION CHECK (S3 vs S2 OPS-SAT-AD)")
    print(f"{'='*60}")
    delta = ensemble_f05 - s2_claim_f05
    print(f"\n  S2 claim (OPS-SAT-AD median): F0.5={s2_claim_f05:.3f}")
    print(f"  S3 result ({run_tag}):          F0.5={ensemble_f05:.3f}  (Δ{delta:+.3f})")
    if delta >= -0.05:
        print("  ✓ S3 F0.5 within 0.05 of S2 — cross-mission generalization supported")
    elif delta >= -0.15:
        print("  ~ S3 F0.5 degraded by >0.05 but <0.15 — partial generalization")
    else:
        print("  ✗ S3 F0.5 degraded by >0.15 — generalization NOT supported for this channel")

    # ── Final claim line ──────────────────────────────────────────────────────
    ci_lower = winner_ci["f05_ci_lower"]
    print(f"\n{'='*60}")
    print(f"CLAIM S3 ({run_tag})")
    print(f"{'='*60}")
    print(
        f"\n  LPINormalizingFlow ensemble ({best_strat}, 5 seeds), 16 features auditadas,"
        f"\n  F0.5={ensemble_f05:.3f} (CI95=[{ci_lower:.3f}, {winner_ci['f05_ci_upper']:.3f}]),"
        f"\n  AUC={winner_ci['auc_point']:.3f} (CI95=[{winner_ci['auc_ci_lower']:.3f}, {winner_ci['auc_ci_upper']:.3f}]),"
        f"\n  ESA-Mission1 {run_tag}, one-shot test, GroupKFold threshold selection"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S3 — LPINormalizingFlow ensemble on ESA-AD Mission1",
    )
    parser.add_argument(
        "--data_path", required=True,
        help="Path to dataset.csv produced by prepare_mission1.py",
    )
    parser.add_argument(
        "--channel", default=None, metavar="CHANNEL_NAME",
        help="Restrict to a single channel (e.g. 'channel_47'). Default: all channels.",
    )
    args = parser.parse_args()

    run(Path(args.data_path), args.channel)


if __name__ == "__main__":
    main()
