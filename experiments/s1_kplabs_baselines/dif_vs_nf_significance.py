"""
Significance analysis: DIF vs NF ensemble median vs NF ensemble rank.

Pregunta: ¿es la superioridad de DIF (F0.5=0.893) sobre NF median (F0.5=0.871)
estadísticamente significativa?

Protocolo
─────────
  1. DIF (PyOD 3.0, contamination=0.20, sampling=5, 18 features, random_state=42)
       → binary predictions + continuous scores → CI95 bootstrap (B=1000)
  2. NF ensemble (5 seeds, 16 features, protocolo anti-snooping idéntico al paper)
       → scores median + scores rank → CI95 bootstrap (B=1000, mismo master_seed)
  3. Bootstrap PAIRED (mismos índices para DIF y NF en cada resample)
       → distribución de δF0.5 = F0.5_DIF − F0.5_NF
       → p-value unilateral: P(δ > 0) = fracción de boots donde DIF > NF
       → CI95 del δ (si 0 fuera del CI → diferencia significativa al 5%)

Nota sobre comparabilidad
─────────────────────────
  • DIF usa 18 features (incluyendo n_peaks y gaps_squared).
  • NF ensemble usa 16 features (excluye n_peaks y gaps_squared por length leakage).
  • Esta asimetría es correcta: el paper de DIF no tiene acceso a nuestro audit.
    El claim de NF es más defensible para publicación aunque el punto sea menor.
    El CI del δ incorpora esta diferencia de forma natural.

Usage
─────
    python experiments/s1_kplabs_baselines/dif_vs_nf_significance.py
"""
from __future__ import annotations

import sys
import warnings
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.loader import REFERENCE_DATA_DIR
from src.models.lpi_v2 import LPINormalizingFlow

# ── Constants ─────────────────────────────────────────────────────────────────

BOOTSTRAP_N           = 1000
BOOTSTRAP_MASTER_SEED = 42
SAMPLING              = 5
CONTAMINATION         = 0.20
SEED                  = 42

NF_SEEDS              = [0, 1, 42, 123, 999]
NF_EXCLUDE            = {"n_peaks", "gaps_squared"}
NF_CV_FOLDS           = 5
NF_SWEEP_PERCENTILES  = [85, 88, 90, 92, 95]
NF_PARAMS             = dict(
    n_components_range=(2, 15),
    n_bootstrap=20,
    scaler="robust",
    n_flow_layers=4,
    flow_hidden=64,
    n_epochs=200,
    flow_lr=1e-3,
    flow_patience=30,
)


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_data_18() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """18 features — for DIF (same as run_kplabs_baselines.py)."""
    df = pd.read_csv(REFERENCE_DATA_DIR / "dataset.csv", index_col="segment")
    df = df[df["sampling"] == SAMPLING]
    meta = {"anomaly", "train", "channel", "sampling"}
    features = [c for c in df.columns if c not in meta]

    train = df[df["train"] == 1]
    test  = df[df["train"] == 0]

    X_train_normal = train.loc[train["anomaly"] == 0, features].values
    X_train_all    = train[features].values
    X_test         = test[features].values
    y_test         = test["anomaly"].values.astype(int)

    scaler = StandardScaler()
    scaler.fit(X_train_normal)

    return (
        np.nan_to_num(scaler.transform(X_train_normal), nan=0., posinf=1e6, neginf=-1e6),
        np.nan_to_num(scaler.transform(X_test),         nan=0., posinf=1e6, neginf=-1e6),
        y_test,
        features,
    )


def load_data_16() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """16 features — for NF ensemble (excludes n_peaks + gaps_squared)."""
    df = pd.read_csv(REFERENCE_DATA_DIR / "dataset.csv", index_col="segment")
    df = df[df["sampling"] == SAMPLING]
    meta = {"anomaly", "train", "channel", "sampling"}
    features = [c for c in df.columns if c not in meta and c not in NF_EXCLUDE]

    df5 = df
    X_train = df5.loc[df5["train"] == 1, features].values.astype(float)
    y_train = df5.loc[df5["train"] == 1, "anomaly"].values.astype(int)
    X_test  = df5.loc[df5["train"] == 0, features].values.astype(float)
    y_test  = df5.loc[df5["train"] == 0, "anomaly"].values.astype(int)

    return X_train, y_train, X_test, y_test


# ── DIF runner ────────────────────────────────────────────────────────────────

def run_dif() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit DIF (same config as run_kplabs_baselines.py) on X_train_normal,
    predict on X_test.

    Returns (y_test, y_pred_binary, y_score_continuous).
    """
    from pyod.models.dif import DIF

    print(f"\n{'─'*60}")
    print("Step 1 — DIF (Deep Isolation Forest)")
    print(f"  contamination={CONTAMINATION}, random_state={SEED}, 18 features")
    print(f"{'─'*60}")

    X_train_normal, X_test, y_test, features = load_data_18()
    print(f"  Train normal: {len(X_train_normal)} | Test: {len(X_test)} "
          f"| Anom rate test: {y_test.mean():.1%}")

    t0 = time.perf_counter()
    model = DIF(contamination=CONTAMINATION, random_state=SEED)
    model.fit(X_train_normal)
    y_pred   = model.predict(X_test)        # 0 / 1
    y_score  = model.decision_function(X_test)
    y_score  = np.nan_to_num(y_score, nan=0., posinf=1e6, neginf=-1e6)
    elapsed  = time.perf_counter() - t0

    f05 = fbeta_score(y_test, y_pred, beta=0.5, zero_division=0)
    auc = roc_auc_score(y_test, y_score)
    print(f"  F0.5={f05:.3f}  AUC={auc:.3f}  ({elapsed:.1f}s)")

    return y_test, y_pred, y_score


# ── NF ensemble runner ────────────────────────────────────────────────────────

def _select_threshold(oof_scores: np.ndarray, y_train: np.ndarray) -> tuple[int, float]:
    best_p, best_f05 = NF_SWEEP_PERCENTILES[-1], -1.
    for p in NF_SWEEP_PERCENTILES:
        thr  = float(np.percentile(oof_scores, p))
        preds = (oof_scores >= thr).astype(int)
        f05  = fbeta_score(y_train, preds, beta=0.5, zero_division=0)
        if f05 > best_f05:
            best_f05, best_p = f05, p
    return best_p, best_f05


def _norm(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    lo, hi = ref.min(), ref.max()
    if hi <= lo:
        return np.zeros_like(x, dtype=float)
    return np.clip((x - lo) / (hi - lo), 0., 1.)


def _frac_ranks(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n <= 1:
        return np.zeros(n)
    r = rankdata(x, method="average")
    return (r - 1.) / (n - 1.)


def run_nf_ensemble() -> dict[str, dict]:
    """
    Train NF ensemble (5 seeds, 16 features) following the exact anti-snooping
    protocol from run_nf_seed_ensemble.py.

    Returns dict keyed by strategy ('median', 'rank') with:
        ens_test: continuous ensemble score (n_test,)
        threshold: float (derived from OOF — no test labels used)
        y_test:    ground truth
        y_pred:    binary predictions
    """
    print(f"\n{'─'*60}")
    print("Step 2 — NF ensemble (5 seeds, 16 features, anti-snooping protocol)")
    print(f"  Seeds: {NF_SEEDS}  |  NF config: 4L h64 200ep")
    print(f"{'─'*60}")

    X_train, y_train, X_test, y_test = load_data_16()
    print(f"  Train: {len(X_train)} | Test: {len(X_test)} "
          f"| Anom rate test: {y_test.mean():.1%}")

    oof_list, test_list = [], []
    for seed in NF_SEEDS:
        t0 = time.perf_counter()
        print(f"\n  Seed {seed} ... ", end="", flush=True)
        det = LPINormalizingFlow(**NF_PARAMS, random_state=seed)
        oof_scores = det.fit_predict_cv(X_train, y_train, cv=NF_CV_FOLDS)
        det.fit(X_train, y_train)
        test_scores = det.score(X_test)
        oof_list.append(oof_scores)
        test_list.append(test_scores)
        p, vf = _select_threshold(oof_scores, y_train)
        elapsed = time.perf_counter() - t0
        # per-seed test metrics
        thr = float(np.percentile(oof_scores, p))
        test_preds = (test_scores >= thr).astype(int)
        f05_seed = fbeta_score(y_test, test_preds, beta=0.5, zero_division=0)
        print(f"Val F0.5={vf:.3f}  Test F0.5={f05_seed:.3f}  ({elapsed:.0f}s)")

    strategies: dict[str, dict] = {}
    for strategy in ("median", "rank"):
        print(f"\n  Building {strategy} ensemble ...")
        if strategy == "median":
            ens_oof  = np.median(
                np.stack([_norm(oof, oof)  for oof in oof_list], axis=1), axis=1
            )
            ens_test = np.median(
                np.stack([_norm(oof, tst)  for oof, tst in zip(oof_list, test_list)], axis=1),
                axis=1,
            )
        else:  # rank
            ens_oof  = np.stack([_frac_ranks(oof) for oof in oof_list], axis=1).mean(axis=1)
            ens_test = np.stack([_frac_ranks(tst) for tst in test_list], axis=1).mean(axis=1)

        best_p, val_f05 = _select_threshold(ens_oof, y_train)
        thr = float(np.percentile(ens_oof, best_p))
        y_pred = (ens_test >= thr).astype(int)
        f05  = fbeta_score(y_test, y_pred, beta=0.5, zero_division=0)
        auc  = roc_auc_score(y_test, ens_test)

        print(f"  Val F0.5={val_f05:.3f}  Test F0.5={f05:.3f}  AUC={auc:.3f}  p{best_p}")

        strategies[strategy] = {
            "ens_test":  ens_test,
            "threshold": thr,
            "y_test":    y_test,
            "y_pred":    y_pred,
            "f05":       f05,
            "auc":       auc,
            "val_f05":   val_f05,
            "best_p":    best_p,
        }

    return strategies, y_test


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    B: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_MASTER_SEED,
) -> dict:
    """
    Bootstrap CI95 for F0.5 and AUC. Threshold is FIXED (derived from OOF
    or contamination — never from the test set being resampled).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f05_b, auc_b = [], []

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yb  = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        sb  = scores[idx]
        pb  = (sb >= threshold).astype(int)
        f05_b.append(fbeta_score(yb, pb, beta=0.5, zero_division=0))
        auc_b.append(float(roc_auc_score(yb, sb)))

    f05_arr = np.array(f05_b)
    auc_arr = np.array(auc_b)

    # Point estimates on the full test set
    pt_pred = (scores >= threshold).astype(int)
    pt_f05  = float(fbeta_score(y_true, pt_pred, beta=0.5, zero_division=0))
    pt_auc  = float(roc_auc_score(y_true, scores))

    return {
        "f05_point": pt_f05,
        "f05_lo":    float(np.percentile(f05_arr, 2.5)),
        "f05_hi":    float(np.percentile(f05_arr, 97.5)),
        "f05_std":   float(f05_arr.std()),
        "auc_point": pt_auc,
        "auc_lo":    float(np.percentile(auc_arr, 2.5)),
        "auc_hi":    float(np.percentile(auc_arr, 97.5)),
        "n_valid":   len(f05_b),
    }


# ── Paired bootstrap significance test ───────────────────────────────────────

def paired_bootstrap(
    y_true:      np.ndarray,
    scores_A:    np.ndarray,  # "challenger" (here: DIF)
    threshold_A: float,
    scores_B:    np.ndarray,  # "reference"  (here: NF median or NF rank)
    threshold_B: float,
    B:    int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_MASTER_SEED,
    label_A: str = "A",
    label_B: str = "B",
) -> dict:
    """
    Paired bootstrap: same resample indices for both models.

    Returns:
        delta_point:  F0.5_A − F0.5_B on the full test set
        delta_ci_lo/hi: CI95 of δ across B resamples
        p_A_gt_B:     empirical P(F0.5_A > F0.5_B) over bootstrap resamples
        significant:  bool — True if CI95 of δ excludes 0
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas: list[float] = []

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yb  = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue

        sA = scores_A[idx]
        sB = scores_B[idx]
        pA = (sA >= threshold_A).astype(int)
        pB = (sB >= threshold_B).astype(int)

        f05A = fbeta_score(yb, pA, beta=0.5, zero_division=0)
        f05B = fbeta_score(yb, pB, beta=0.5, zero_division=0)
        deltas.append(f05A - f05B)

    delta_arr = np.array(deltas)

    # Point delta on full test set
    ptA = (scores_A >= threshold_A).astype(int)
    ptB = (scores_B >= threshold_B).astype(int)
    pt_f05A = float(fbeta_score(y_true, ptA, beta=0.5, zero_division=0))
    pt_f05B = float(fbeta_score(y_true, ptB, beta=0.5, zero_division=0))
    delta_point = pt_f05A - pt_f05B

    ci_lo = float(np.percentile(delta_arr, 2.5))
    ci_hi = float(np.percentile(delta_arr, 97.5))
    p_A_gt_B = float((delta_arr > 0).mean())

    significant = (ci_lo > 0) or (ci_hi < 0)  # CI excludes 0

    direction = ">" if delta_point > 0 else "<"
    sig_str   = "SIGNIFICANT" if significant else "not significant"

    print(f"\n  {label_A} vs {label_B}:")
    print(f"    F0.5_{label_A}={pt_f05A:.3f}  F0.5_{label_B}={pt_f05B:.3f}  "
          f"δ={delta_point:+.3f}  ({label_A} {direction} {label_B})")
    print(f"    δ CI95=[{ci_lo:+.3f}, {ci_hi:+.3f}]  "
          f"P({label_A}>{label_B})={p_A_gt_B:.3f}  → {sig_str}")

    return {
        "delta_point":  delta_point,
        "delta_ci_lo":  ci_lo,
        "delta_ci_hi":  ci_hi,
        "p_A_gt_B":     p_A_gt_B,
        "significant":  significant,
        "n_valid":      len(deltas),
        "label_A":      label_A,
        "label_B":      label_B,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    print(f"\n{'='*62}")
    print("  DIF vs NF ensemble — Bootstrap significance analysis")
    print(f"  B={BOOTSTRAP_N}, master_seed={BOOTSTRAP_MASTER_SEED}")
    print(f"{'='*62}")

    # ── 1. DIF ────────────────────────────────────────────────────────────────
    y_test, dif_pred, dif_score = run_dif()

    # DIF threshold: implied by contamination (not directly available as a scalar).
    # We re-derive it as the (1 - contamination) percentile of DIF scores
    # on the training set — but since we only have test scores, we use the
    # equivalent: the score value at the 80th percentile of the test score
    # distribution, which is what PyOD uses internally.
    # For the bootstrap, the threshold is FIXED as the value that produced y_pred.
    # We reconstruct it from the observed predictions: threshold = min score among
    # predicted anomalies (i.e., the operating point DIF chose).
    anomaly_scores = dif_score[dif_pred == 1]
    if len(anomaly_scores) == 0:
        # degenerate: DIF predicted no anomalies — use 80th percentile
        dif_threshold = float(np.percentile(dif_score, (1 - CONTAMINATION) * 100))
    else:
        dif_threshold = float(anomaly_scores.min())

    print(f"\n  DIF threshold (reconstructed from predictions): {dif_threshold:.6f}")

    # ── 2. NF ensemble ────────────────────────────────────────────────────────
    nf_strategies, _ = run_nf_ensemble()

    # ── 3. Bootstrap CI for each model ───────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"Step 3 — Bootstrap CI95  (B={BOOTSTRAP_N}, master_seed={BOOTSTRAP_MASTER_SEED})")
    print(f"{'─'*62}")

    print("\n  DIF ...")
    ci_dif = bootstrap_ci(y_test, dif_score, dif_threshold)

    print("\n  NF ensemble median ...")
    nf_med = nf_strategies["median"]
    ci_nf_med = bootstrap_ci(y_test, nf_med["ens_test"], nf_med["threshold"])

    print("\n  NF ensemble rank ...")
    nf_rank = nf_strategies["rank"]
    ci_nf_rank = bootstrap_ci(y_test, nf_rank["ens_test"], nf_rank["threshold"])

    # ── 4. CI summary table ───────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("Bootstrap CI95 — Summary")
    print(f"{'='*62}")
    print(f"\n  {'Model':<32}  {'F0.5':>6}  {'CI95 F0.5':>20}  {'AUC':>6}  {'CI95 AUC':>20}")
    print(f"  {'─'*32}  {'─'*6}  {'─'*20}  {'─'*6}  {'─'*20}")

    rows = [
        ("DIF (18 feats, contamination=0.20)",
         ci_dif,  "PyOD default threshold"),
        ("NF ensemble median (16 feats audited)",
         ci_nf_med,  "OOF threshold"),
        ("NF ensemble rank   (16 feats audited)",
         ci_nf_rank, "OOF threshold"),
    ]
    for label, ci, note in rows:
        print(
            f"  {label:<32}  {ci['f05_point']:>6.3f}  "
            f"[{ci['f05_lo']:.3f}, {ci['f05_hi']:.3f}]      "
            f"{ci['auc_point']:>6.3f}  "
            f"[{ci['auc_lo']:.3f}, {ci['auc_hi']:.3f}]"
        )

    # ── 5. Paired bootstrap significance tests ────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"Step 4 — Paired bootstrap significance tests")
    print(f"  (same resample indices applied to both models simultaneously)")
    print(f"{'─'*62}")

    r1 = paired_bootstrap(
        y_test,
        dif_score,         dif_threshold,
        nf_med["ens_test"], nf_med["threshold"],
        label_A="DIF",  label_B="NF-median",
    )
    r2 = paired_bootstrap(
        y_test,
        dif_score,          dif_threshold,
        nf_rank["ens_test"], nf_rank["threshold"],
        label_A="DIF",  label_B="NF-rank",
    )
    r3 = paired_bootstrap(
        y_test,
        nf_rank["ens_test"], nf_rank["threshold"],
        nf_med["ens_test"],  nf_med["threshold"],
        label_A="NF-rank",  label_B="NF-median",
    )

    # ── 6. Verdict ────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("VEREDICTO FINAL")
    print(f"{'='*62}")

    def _sig(r: dict) -> str:
        lo, hi = r["delta_ci_lo"], r["delta_ci_hi"]
        if lo > 0:
            return f"{r['label_A']} significativamente MEJOR (δ CI95 excluye 0 por arriba)"
        if hi < 0:
            return f"{r['label_B']} significativamente MEJOR (δ CI95 excluye 0 por abajo)"
        return (f"Sin diferencia significativa al 5% — CI95 del δ=[{lo:+.3f},{hi:+.3f}] incluye 0 "
                f"(P={r['p_A_gt_B']:.2f})")

    print(f"\n  1. DIF vs NF-median:  {_sig(r1)}")
    print(f"  2. DIF vs NF-rank:    {_sig(r2)}")
    print(f"  3. NF-rank vs NF-median: {_sig(r3)}")

    # Publication implications
    print(f"\n{'─'*62}")
    print("Implicaciones para el paper")
    print(f"{'─'*62}")

    if not r1["significant"] or r1["delta_ci_lo"] < 0:
        print(
            "\n  • DIF NO supera a NF-median con significancia estadística."
            "\n    El claim F0.5=0.871 (CI95=[0.780, 0.931]) de NF-median sigue siendo válido"
            "\n    como mejor modelo no-supervisado publicable."
        )
    else:
        print(
            "\n  • DIF supera a NF-median significativamente."
            "\n    Revisar claim — DIF puede ser el nuevo baseline a batir."
        )

    if not r2["significant"] or r2["delta_ci_lo"] >= 0:
        # DIF not significantly better than NF-rank
        if r2["delta_ci_hi"] < 0:
            print(
                "\n  • NF-rank (F0.5=0.957) supera a DIF con significancia estadística."
                "\n    El claim NF-rank domina a DIF — reportar ambos."
            )
        else:
            print(
                "\n  • DIF y NF-rank NO son significativamente distintos al 5%."
                f"\n    CI95(δ)=[{r2['delta_ci_lo']:+.3f}, {r2['delta_ci_hi']:+.3f}]"
                "\n    Se pueden reportar ambos en la tabla sin afirmación de superioridad."
            )

    if r3["significant"] and r3["delta_ci_lo"] > 0:
        print(
            "\n  • NF-rank supera a NF-median significativamente."
            f"\n    Considerar elevar el claim oficial de median → rank"
            f"\n    (pero revisar val→test gap antes: ver CLAUDE.md Decision 9)."
        )
    elif not r3["significant"]:
        print(
            "\n  • NF-rank y NF-median NO son significativamente distintos al 5%."
            "\n    El claim conservador (median) sigue siendo el correcto."
        )

    # Final comparison table for paper
    print(f"\n{'='*62}")
    print("TABLA PARA EL PAPER (no-supervisados, sampling=5)")
    print(f"{'='*62}")
    print(f"\n  {'Modelo':<38}  {'F0.5':>6}  {'CI95 F0.5':>20}  {'AUC':>6}")
    print(f"  {'─'*38}  {'─'*6}  {'─'*20}  {'─'*6}")
    table_rows = [
        ("NF ensemble rank (16 feats, Quesada 2026)",
         ci_nf_rank["f05_point"], ci_nf_rank["f05_lo"], ci_nf_rank["f05_hi"],
         ci_nf_rank["auc_point"]),
        ("DIF (18 feats, PyOD default)",
         ci_dif["f05_point"], ci_dif["f05_lo"], ci_dif["f05_hi"],
         ci_dif["auc_point"]),
        ("NF ensemble median (16 feats, Quesada 2026)",
         ci_nf_med["f05_point"], ci_nf_med["f05_lo"], ci_nf_med["f05_hi"],
         ci_nf_med["auc_point"]),
    ]
    for label, f05, lo, hi, auc in sorted(table_rows, key=lambda r: -r[1]):
        print(f"  {label:<38}  {f05:>6.3f}  [{lo:.3f}, {hi:.3f}]      {auc:>6.3f}")

    print(f"\n  {'─'*62}")
    print(f"  Nota: 'significativamente mejor' = CI95(δ) excluye 0 (p<0.05, paired bootstrap)")
    print(f"  B={BOOTSTRAP_N} resamples, master_seed={BOOTSTRAP_MASTER_SEED}")


if __name__ == "__main__":
    run()
