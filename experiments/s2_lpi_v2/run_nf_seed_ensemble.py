"""
S2 — LPINormalizingFlow Seed Ensemble  (experiment tag: s2_nf_ensemble_v7)

Objective
---------
Estabilizar el claim de LPINormalizingFlow mediante un ensemble de 5 seeds
y obtener un intervalo de confianza publicable (CI95 bootstrap).

Feature set: 16 features  (18 originales − n_peaks − gaps_squared)
─────────────────────────────────────────────────────────────────
  Excluidas:
  • n_peaks       — length leakage confirmado: segmentos anómalos son 3.4× más
                    largos que los normales (EDA notebook 01, audit notebook 03).
                    Más puntos → más picos crudos por construcción, no por física.
  • gaps_squared  — audit notebook 04 mostró que su AUC individual en test es
                    elevado pero correlacionado con duración del segmento. Con el
                    feature set limpio de 16 la media de F0.5 sobre 5 seeds es
                    ~0.820, sin la ambigüedad de señal fisica vs artefacto de
                    longitud. Necesaria para defensabilidad cross-dataset (SpainSat NG).
  Retenida:
  • diff2_var     — jerk de la varianza (cambio en frecuencia de oscilación).
                    Audit notebook 04 confirma señal física real. ΔF0.5=−0.342
                    si se elimina.

Protocol (anti-snooping estricto)
──────────────────────────────────
  1. Para cada seed ∈ [0, 1, 42, 123, 999]:
       a. 5-fold CV sobre train → scores OOF (sin ver test)
       b. Final fit sobre todo el train
       c. Generar test scores (sin umbral aún)
  2. Tres estrategias de ensemble sobre los 5 seeds:
       • Mean   — normalizar [0,1] por seed (ref=OOF), promediar
       • Median — igual pero con mediana
       • Rank   — convertir a ranks fraccionarios, promediar ranks
  3. Para cada estrategia: threshold sweep sobre OOF ensemble → best_p
     → ONE-SHOT test evaluation con ese umbral
  4. Bootstrap CI95 (B=1000, master_seed=42) para F0.5 y AUC del mejor ensemble
  5. MLflow tracking (experiment: s2_nf_ensemble_v7)

Usage
─────
    cd /path/to/anomaly-internal-detection
    python experiments/s2_lpi_v2/run_nf_seed_ensemble.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import fbeta_score, roc_auc_score

from src.data.loader import REFERENCE_DATA_DIR
from src.evaluation.metrics import compute_metrics
from src.models.lpi_v2 import LPINormalizingFlow

# ─── Constants ────────────────────────────────────────────────────────────────

SEEDS = [0, 1, 42, 123, 999]

# Features excluded from the original 18:
#   n_peaks      → length leakage (audit 03)
#   gaps_squared → length-confounded dominant feature (audit 04, Decision 8)
EXCLUDE_FEATURES: set[str] = {"n_peaks", "gaps_squared"}
EXPECTED_N_FEATURES = 16  # 18 original − 2 excluded

SAMPLING_FILTER = 5
CV_FOLDS = 5
SWEEP_PERCENTILES = [85, 88, 90, 92, 95]

BOOTSTRAP_N = 1000
BOOTSTRAP_MASTER_SEED = 42  # master seed for CI resampling only

MLFLOW_EXPERIMENT = "s2_nf_ensemble_v7"

# NF hyperparameters — same as winning config in compare_extensions.py
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

# Published baselines for comparison table
PUBLISHED_BASELINES = {
    "OCSVM (S1)": {"f05": 0.669, "auc": 0.800},
    "LPI v1 sin n_peaks (S2, official)": {"f05": 0.670, "auc": 0.920},
}


# ─── Data ─────────────────────────────────────────────────────────────────────


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load sampling=5 features with 16-feature set. Confirms count at runtime."""
    csv = REFERENCE_DATA_DIR / "dataset.csv"
    df = pd.read_csv(csv, index_col="segment")

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

    df5 = df[df["sampling"] == SAMPLING_FILTER]
    X_train = df5.loc[df5["train"] == 1, feature_cols].values.astype(float)
    y_train = df5.loc[df5["train"] == 1, "anomaly"].values.astype(int)
    X_test  = df5.loc[df5["train"] == 0, feature_cols].values.astype(float)
    y_test  = df5.loc[df5["train"] == 0, "anomaly"].values.astype(int)

    print(f"  Train : {len(X_train)} segs  anomaly rate {y_train.mean():.1%}")
    print(f"  Test  : {len(X_test)} segs   anomaly rate {y_test.mean():.1%}")

    return X_train, y_train, X_test, y_test, feature_cols


# ─── Threshold sweep ──────────────────────────────────────────────────────────


def select_threshold_oof(
    oof_scores: np.ndarray, y_train: np.ndarray
) -> tuple[int, float, list[tuple[int, float]]]:
    """
    Sweep percentile thresholds over OOF scores (no test data used).
    Returns (best_p, best_val_f05, sweep_table).
    """
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
    """
    Train one LPINormalizingFlow with a given seed.

    Returns dict with:
      seed, oof_scores, test_scores (raw, no threshold),
      val_auc, val_f05, best_p,
      test_f05, test_auc, test_precision, test_recall, elapsed
    """
    print(f"\n{'─'*60}")
    print(f"Seed {seed}")
    print(f"{'─'*60}")

    t0 = time.perf_counter()

    detector = LPINormalizingFlow(**NF_PARAMS, random_state=seed)

    # Step 1: 5-fold CV → OOF scores (threshold selection will use these)
    oof_scores = detector.fit_predict_cv(X_train, y_train, cv=CV_FOLDS)
    val_auc = float(roc_auc_score(y_train, oof_scores))

    # Step 2: threshold from OOF only (for per-seed test reporting)
    best_p, val_f05, sweep = select_threshold_oof(oof_scores, y_train)

    # Step 3: final model on all train data
    detector.fit(X_train, y_train)

    # Step 4: test scores (raw; threshold applied separately for per-seed report)
    test_scores = detector.score(X_test)

    # Per-seed test metrics (OOF threshold applied to test — valid for reporting)
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
    """
    Normalize `scores` to [0,1] using min/max derived from `ref_scores`.
    Using OOF as reference for both OOF and test prevents any leakage of
    test distribution into the normalization.
    """
    lo, hi = float(ref_scores.min()), float(ref_scores.max())
    if hi <= lo:
        return np.zeros_like(scores, dtype=float)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)


def to_fractional_ranks(scores: np.ndarray) -> np.ndarray:
    """
    Convert scores to fractional ranks in [0, 1] (ties get average rank).
    Higher score → higher rank (closer to 1). Rank 0 = lowest score.
    """
    n = len(scores)
    if n <= 1:
        return np.zeros(n, dtype=float)
    ranks = rankdata(scores, method="average")  # 1-based
    return (ranks - 1.0) / (n - 1.0)


# ─── Ensemble builder ─────────────────────────────────────────────────────────


def build_ensembles(
    seed_results: list[dict],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict]:
    """
    Build the 3 ensemble strategies from 5 seed results.

    For each strategy:
      1. Aggregate OOF scores (normalized/ranked)
      2. Threshold sweep on OOF ensemble → best_p (NO test labels used)
      3. Derive threshold from OOF ensemble at best_p
      4. Apply to aggregated test ensemble → ONE-SHOT predictions

    Returns dict keyed by strategy name.
    """
    oof_list  = [r["oof_scores"]  for r in seed_results]
    test_list = [r["test_scores"] for r in seed_results]

    strategies: dict[str, dict] = {}

    for strategy in ("mean", "median", "rank"):
        print(f"\n{'─'*60}")
        print(f"Ensemble strategy: {strategy}")
        print(f"{'─'*60}")

        if strategy in ("mean", "median"):
            # Normalize each seed to [0,1] using its own OOF as reference
            norm_oof  = np.stack(
                [normalize_minmax(oof, oof) for oof in oof_list], axis=1
            )  # (n_train, 5)
            norm_test = np.stack(
                [normalize_minmax(oof, test) for oof, test in zip(oof_list, test_list)],
                axis=1,
            )  # (n_test, 5)

            if strategy == "mean":
                ens_oof  = norm_oof.mean(axis=1)
                ens_test = norm_test.mean(axis=1)
            else:
                ens_oof  = np.median(norm_oof, axis=1)
                ens_test = np.median(norm_test, axis=1)

        else:  # rank
            # Fractional ranks within each seed's score distribution
            rank_oof  = np.stack(
                [to_fractional_ranks(oof) for oof in oof_list], axis=1
            )
            rank_test = np.stack(
                [to_fractional_ranks(test) for test in test_list], axis=1
            )
            ens_oof  = rank_oof.mean(axis=1)
            ens_test = rank_test.mean(axis=1)

        # Threshold selection on OOF ensemble only
        best_p, val_f05, sweep_rows = select_threshold_oof(ens_oof, y_train)
        val_auc = float(roc_auc_score(y_train, ens_oof))

        print(f"  Threshold sweep (OOF):")
        for p, f in sweep_rows:
            marker = " ←" if p == best_p else ""
            print(f"    p{p:>2}  F0.5={f:.3f}{marker}")

        # Threshold value derived from OOF ensemble
        ens_thr = float(np.percentile(ens_oof, best_p))

        # ONE-SHOT test evaluation
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
    """
    Bootstrap CI95 for F0.5 and AUC on the test set.
    Threshold is derived from OOF (not from test) — stays fixed during bootstrap.

    Returns dict with point estimates and CI95 bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(y_test)

    f05_boots: list[float] = []
    auc_boots: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores_b = ensemble_scores[idx]
        y_b = y_test[idx]

        if len(np.unique(y_b)) < 2:
            # Degenerate resample — skip
            continue

        preds_b = (scores_b >= threshold).astype(int)
        f05_boots.append(fbeta_score(y_b, preds_b, beta=0.5, zero_division=0))
        auc_boots.append(float(roc_auc_score(y_b, scores_b)))

    f05_arr = np.array(f05_boots)
    auc_arr = np.array(auc_boots)

    # Point estimates (full test set, no resampling)
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


def run() -> None:
    X_train, y_train, X_test, y_test, feature_cols = load_data()

    print(f"\n{'='*60}")
    print("LPINormalizingFlow Seed Ensemble  (s2_nf_ensemble_v7)")
    print(f"{'='*60}")
    print(f"Seeds: {SEEDS}")
    print(f"NF config: {NF_PARAMS}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="nf_seed_ensemble"):
        mlflow.log_params(
            {
                "seeds": str(SEEDS),
                "sampling_filter": SAMPLING_FILTER,
                "n_train": len(X_train),
                "n_test": len(X_test),
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

        # Sanity check: models are distinct
        all_oof = np.stack([r["oof_scores"] for r in seed_results], axis=1)
        cross_seed_std = float(all_oof.std(axis=1).mean())
        print(f"\n  Cross-seed OOF score std (mean over samples): {cross_seed_std:.4f}")
        if cross_seed_std < 1e-6:
            print("  !! WARNING: all seeds produced identical OOF scores — models may not differ")
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

        # ── Step 3: Bootstrap CI for each ensemble ────────────────────────────
        print(f"\n{'='*60}")
        print(f"STEP 3 — Bootstrap CI95  (B={BOOTSTRAP_N}, master_seed={BOOTSTRAP_MASTER_SEED})")
        print(f"{'='*60}")

        ci_results: dict[str, dict] = {}
        for name, strat in strategies.items():
            print(f"\n  {name} ensemble ...")
            ci = bootstrap_ci_metrics(
                strat["ens_test"],
                y_test,
                strat["threshold"],
                n_bootstrap=BOOTSTRAP_N,
                seed=BOOTSTRAP_MASTER_SEED,
            )
            ci_results[name] = ci

            print(
                f"  F0.5  point={ci['f05_point']:.3f}  "
                f"CI95=[{ci['f05_ci_lower']:.3f}, {ci['f05_ci_upper']:.3f}]  "
                f"std={ci['f05_std']:.3f}"
            )
            print(
                f"  AUC   point={ci['auc_point']:.3f}  "
                f"CI95=[{ci['auc_ci_lower']:.3f}, {ci['auc_ci_upper']:.3f}]  "
                f"std={ci['auc_std']:.3f}"
            )
            print(f"  Valid bootstrap resamples: {ci['n_valid_boots']}/{BOOTSTRAP_N}")

            mlflow.log_metrics(
                {
                    f"ens_{name}_f05_point": ci["f05_point"],
                    f"ens_{name}_f05_ci_lower": ci["f05_ci_lower"],
                    f"ens_{name}_f05_ci_upper": ci["f05_ci_upper"],
                    f"ens_{name}_f05_std": ci["f05_std"],
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
    print("TABLE 2 — Ensemble strategies vs baselines (test set)")
    print(f"{'='*60}")

    ens_rows = []
    for bname, bmet in PUBLISHED_BASELINES.items():
        ens_rows.append(
            {
                "Model": bname,
                "F0.5": f"{bmet['f05']:.3f}",
                "F0.5 CI95": "—",
                "AUC": f"{bmet['auc']:.3f}",
                "AUC CI95": "—",
                "Val F0.5": "—",
                "Note": "published",
            }
        )

    # NF single seed summary row
    ens_rows.append(
        {
            "Model": "NF single seed (mean±std, 5 seeds)",
            "F0.5": f"{np.mean(seed_f05):.3f}",
            "F0.5 CI95": f"±{np.std(seed_f05):.3f} (seed std)",
            "AUC": f"{np.mean(seed_auc):.3f}",
            "AUC CI95": f"±{np.std(seed_auc):.3f} (seed std)",
            "Val F0.5": "—",
            "Note": "not ensemble",
        }
    )

    strategy_display = {"mean": "NF ensemble (mean)", "median": "NF ensemble (median)", "rank": "NF ensemble (rank)"}
    for sname, display in strategy_display.items():
        strat = strategies[sname]
        ci = ci_results[sname]
        ens_rows.append(
            {
                "Model": display,
                "F0.5": f"{ci['f05_point']:.3f}",
                "F0.5 CI95": f"[{ci['f05_ci_lower']:.3f}, {ci['f05_ci_upper']:.3f}]",
                "AUC": f"{ci['auc_point']:.3f}",
                "AUC CI95": f"[{ci['auc_ci_lower']:.3f}, {ci['auc_ci_upper']:.3f}]",
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

    # Primary criterion: val F0.5 (anti-snooping consistent)
    # Secondary: CI95 width (defensibility)
    best_strat = max(
        strategies.keys(),
        key=lambda s: (strategies[s]["val_f05"], -ci_results[s]["f05_std"]),
    )
    winner_strat = strategies[best_strat]
    winner_ci = ci_results[best_strat]
    winner_name = strategy_display[best_strat]

    best_individual_f05 = max(seed_f05)
    ensemble_f05 = winner_ci["f05_point"]

    print(f"\n  Best strategy (val F0.5 + stability): {winner_name}")
    print(f"  Val F0.5: {winner_strat['val_f05']:.3f}  |  Val AUC: {winner_strat['val_auc']:.3f}")
    print(
        f"  Test F0.5: {ensemble_f05:.3f}  CI95=[{winner_ci['f05_ci_lower']:.3f}, {winner_ci['f05_ci_upper']:.3f}]"
    )
    print(
        f"  Test AUC:  {winner_ci['auc_point']:.3f}  CI95=[{winner_ci['auc_ci_lower']:.3f}, {winner_ci['auc_ci_upper']:.3f}]"
    )

    if ensemble_f05 >= best_individual_f05 - 0.001:
        print(
            f"\n  ✓ Ensemble ({ensemble_f05:.3f}) ≥ best individual ({best_individual_f05:.3f}) — ensemble justified"
        )
    else:
        delta = best_individual_f05 - ensemble_f05
        print(
            f"\n  !! Ensemble ({ensemble_f05:.3f}) < best individual ({best_individual_f05:.3f}) "
            f"by {delta:.3f} — consider regularisation instead of ensemble"
        )

    # ── Comparison vs baselines ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPARISON VS BASELINES")
    print(f"{'='*60}")

    for bname, bmet in PUBLISHED_BASELINES.items():
        delta_f05 = ensemble_f05 - bmet["f05"]
        delta_auc = winner_ci["auc_point"] - bmet["auc"]
        rel_f05 = delta_f05 / bmet["f05"] * 100
        print(f"\n  vs {bname}:")
        print(f"    F0.5: {bmet['f05']:.3f} → {ensemble_f05:.3f}  (Δ{delta_f05:+.3f}, {rel_f05:+.1f}%)")
        print(f"    AUC : {bmet['auc']:.3f} → {winner_ci['auc_point']:.3f}  (Δ{delta_auc:+.3f})")

    # ── Publishability assessment ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PUBLISHABILITY ASSESSMENT — NeurIPS ML4PS / MNRAS Letters")
    print(f"{'='*60}")

    ci_lower = winner_ci["f05_ci_lower"]
    ci_upper = winner_ci["f05_ci_upper"]
    f05_std_boots = winner_ci["f05_std"]

    print(f"\n  Claim candidato:")
    print(
        f"    LPINormalizingFlow ensemble ({best_strat}), 16 features auditadas,"
    )
    print(
        f"    F0.5={ensemble_f05:.3f} (CI95=[{ci_lower:.3f}, {ci_upper:.3f}]), AUC={winner_ci['auc_point']:.3f}"
    )
    print(f"    (bootstrap std sobre test: {f05_std_boots:.3f})")

    issues = []
    if ci_lower < 0.75:
        issues.append(f"  ⚠ CI95 lower bound {ci_lower:.3f} < 0.75 — claim debe ser conservador")
    if f05_std_boots > 0.07:
        issues.append(f"  ⚠ Bootstrap std {f05_std_boots:.3f} > 0.07 — alta varianza en test")
    if ensemble_f05 < best_individual_f05 - 0.001:
        issues.append("  ⚠ Ensemble no supera mejor seed individual — revisar estrategia")

    if issues:
        print("\n  Issues:")
        for iss in issues:
            print(f"    {iss}")
        print(
            "\n  Recomendación: esperar a S3 (ESA-AD multi-misión) antes de publicar."
            "\n  Con generalización cross-misión validada, el claim se vuelve mucho más sólido."
        )
    else:
        print(
            "\n  ✓ Claim defendible para NeurIPS ML4PS workshop o MNRAS Letters."
            "\n  Argumentos de fortaleza:"
            "\n    1. CI95 lower bound ≥ 0.75 → supera todos los baselines publicados con certeza estadística"
            "\n    2. Features auditadas (sin n_peaks, sin gaps_squared)"
            "\n    3. Anti-snooping: test evaluado ONE-SHOT, threshold de OOF"
            "\n    4. Ensemble de 5 seeds independientes — no cherry-picking de seed=42"
            "\n  Próximo paso recomendado: S3 (ESA-AD) para validar generalización antes de envío."
        )

    # ── Final claim line ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CLAIM FINAL PUBLICABLE")
    print(f"{'='*60}")
    print(
        f"\n  LPINormalizingFlow ensemble ({best_strat}, 5 seeds), 16 features auditadas,"
        f"\n  F0.5={ensemble_f05:.3f} (CI95=[{ci_lower:.3f}, {ci_upper:.3f}]),"
        f"\n  AUC={winner_ci['auc_point']:.3f} (CI95=[{winner_ci['auc_ci_lower']:.3f}, {winner_ci['auc_ci_upper']:.3f}]),"
        f"\n  OPS-SAT-AD sampling=5, one-shot test, GroupKFold threshold selection"
    )


if __name__ == "__main__":
    run()
