"""
S3 — Autoencoder Temporal → Embedding → LPI pipeline  (Quesada 2026)

Two-stage anomaly detection:
  Stage 1 (AE):  Conv1D autoencoder trained on normal windows.
                 The bottleneck embedding captures the "shape" of a normal
                 sequence — anomalous windows are mapped to off-distribution
                 regions in embedding space.
  Stage 2 (LPI): LPINormalizingFlow ensemble on the embeddings.
                 Same 5-seed, 5-fold CV, bootstrap CI protocol as run_nf_ensemble_s3.py.

Motivation (see CLAUDE.md Decisión 10):
  Static features (catch22) have supervised ceiling AUC=0.60 on ESA-Mission1
  ch14 — the anomaly signal is temporal, not captured by summary statistics.
  Learned embeddings from a temporal AE provide richer representations.

──────────────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────────────
  # Step 1 — prepare raw windows:
  python experiments/s3_esa_adb/prepare_mission1_raw.py \\
      --data_dir /workspace/ESA-Mission1/ESA-Mission1 \\
      --out_dir  reference/data/esa_mission1_raw_ch14 \\
      --channels channel_14

  # Step 2 — run AE + LPI:
  python experiments/s3_esa_adb/run_ae_lpi.py \\
      --raw_dir  reference/data/esa_mission1_raw_ch14

  # Quick smoke-test (CPU-friendly, single seed, 2-fold CV):
  python experiments/s3_esa_adb/run_ae_lpi.py \\
      --raw_dir  reference/data/esa_mission1_raw_ch14 \\
      --quick-test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

# Force line-buffered output so nohup logs appear in real time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import mlflow
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, roc_auc_score

from src.evaluation.metrics import compute_metrics
from src.models.conv_autoencoder import (
    ConvAutoencoder,
    extract_embeddings,
    train_autoencoder,
)
from src.models.lpi_v2 import LPINormalizingFlow

# Reuse ensemble-building utilities from the S3 NF pipeline (no duplication)
from experiments.s3_esa_adb.run_nf_ensemble_s3 import (
    BOOTSTRAP_MASTER_SEED,
    BOOTSTRAP_N,
    CV_FOLDS,
    NF_PARAMS,
    SEEDS,
    SWEEP_PERCENTILES,
    bootstrap_ci_metrics,
    build_ensembles,
    normalize_minmax,
    select_threshold_oof,
    to_fractional_ranks,
    train_single_seed,
)

# ─── Constants ────────────────────────────────────────────────────────────────

MLFLOW_EXPERIMENT = "s3_ae_lpi_v1"

AE_DEFAULTS = dict(
    window_size   = 256,
    embedding_dim = 32,
    ae_epochs     = 100,
    ae_lr         = 1e-3,
    ae_batch_size = 512,
    ae_patience   = 15,
)

S2_BASELINES = {
    "OCSVM (S1, OPS-SAT-AD)":                     {"f05": 0.669, "auc": 0.800},
    "LPI v1 sin n_peaks (S2, OPS-SAT-AD)":         {"f05": 0.670, "auc": 0.920},
    "NF ensemble median 5 seeds (S2, OPS-SAT-AD)": {"f05": 0.871, "auc": 0.997},
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_raw(
    raw_dir:        Path,
    channel_filter: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load windows.npy + meta.csv from a prepare_mission1_raw.py output directory.

    Returns
    -------
    windows_train_normal  — (N_tn, T)  windows for AE training
    windows_all_train     — (N_tr, T)  all train windows (for embedding)
    windows_all_test      — (N_te, T)  all test windows  (for embedding)
    y_train               — (N_tr,)    labels for train windows
    y_test                — (N_te,)    labels for test windows
    meta                  — (N,)       full meta DataFrame aligned with windows.npy
    """
    windows_path = raw_dir / "windows.npy"
    meta_path    = raw_dir / "meta.csv"

    if not windows_path.exists():
        raise FileNotFoundError(
            f"{windows_path} not found. "
            "Run prepare_mission1_raw.py first."
        )

    windows = np.load(windows_path)                    # (N, T)
    meta    = pd.read_csv(meta_path, index_col="segment")

    if len(windows) != len(meta):
        raise ValueError(
            f"windows.npy has {len(windows)} rows but meta.csv has {len(meta)} — "
            "corrupted output directory."
        )

    if channel_filter:
        mask = (meta["channel"] == channel_filter).values
        if not mask.any():
            raise ValueError(
                f"--channel '{channel_filter}' not found in meta.csv. "
                f"Available: {sorted(meta['channel'].unique())}"
            )
        windows = windows[mask]
        meta    = meta[mask]
        print(f"  Channel filter: {channel_filter}  ({len(meta)} windows)")

    if meta["train"].sum() == 0:
        raise RuntimeError("No train windows — check temporal split in prepare_mission1_raw.py.")
    if (meta["train"] == 0).sum() == 0:
        raise RuntimeError("No test windows — check temporal split.")

    y    = meta["anomaly"].values.astype(int)
    tr   = meta["train"].values.astype(int)

    mask_train        = tr == 1
    mask_test         = tr == 0
    mask_train_normal = mask_train & (y == 0)

    if mask_train_normal.sum() == 0:
        raise RuntimeError("No normal train windows — cannot train AE.")
    if (y[mask_test] == 1).sum() == 0:
        raise RuntimeError(
            "Test set has 0 anomalies — evaluation is undefined. "
            "Choose a channel with anomalies in both H1 and H2."
        )

    print(f"\n{'─'*60}")
    print(f"[INFO] Raw windows loaded")
    print(f"  Total   : {len(windows):,}  shape={windows.shape}")
    print(f"  Train   : {mask_train.sum():,}  "
          f"(normal={mask_train_normal.sum():,}, "
          f"anom={(mask_train & (y==1)).sum():,})")
    print(f"  Test    : {mask_test.sum():,}  "
          f"(normal={(mask_test & (y==0)).sum():,}, "
          f"anom={(mask_test & (y==1)).sum():,})")
    print(f"  Anomaly rate — train={y[mask_train].mean():.1%}  "
          f"test={y[mask_test].mean():.1%}")

    return (
        windows[mask_train_normal],
        windows[mask_train],
        windows[mask_test],
        y[mask_train],
        y[mask_test],
        meta,
    )


# ─── RF baseline ──────────────────────────────────────────────────────────────

def rf_baseline_on_embeddings(
    emb_train: np.ndarray,
    y_train:   np.ndarray,
    emb_test:  np.ndarray,
    y_test:    np.ndarray,
) -> dict:
    """
    Train a RandomForest on embeddings — supervised ceiling for this dataset.

    The RF has access to anomaly labels during training; the LPI does not
    (it uses labels only to compute cluster enrichments, not the GMM geometry).
    RF AUC is the upper bound the unsupervised pipeline tries to approach.
    """
    print(f"\n{'─'*60}")
    print("RF baseline on embeddings (supervised ceiling)")
    print(f"{'─'*60}")

    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    t0 = time.perf_counter()
    rf.fit(emb_train, y_train)
    t_fit = time.perf_counter() - t0

    proba = rf.predict_proba(emb_test)[:, 1]
    auc   = float(roc_auc_score(y_test, proba))

    # Best F0.5 via threshold sweep on predict_proba
    best_f05, best_thr = 0.0, 0.5
    for p in SWEEP_PERCENTILES:
        thr   = float(np.percentile(proba, p))
        preds = (proba >= thr).astype(int)
        f05   = fbeta_score(y_test, preds, beta=0.5, zero_division=0)
        if f05 > best_f05:
            best_f05, best_thr = f05, thr

    print(f"  AUC={auc:.3f}  best_F0.5={best_f05:.3f}  fit={t_fit:.1f}s")
    return {"rf_auc": auc, "rf_f05": best_f05}


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run(
    raw_dir:        Path,
    channel_filter: str | None,
    quick_test:     bool = False,
    window_size:    int  = AE_DEFAULTS["window_size"],
    embedding_dim:  int  = AE_DEFAULTS["embedding_dim"],
    ae_epochs:      int  = AE_DEFAULTS["ae_epochs"],
    ae_lr:          float = AE_DEFAULTS["ae_lr"],
    ae_batch_size:  int  = AE_DEFAULTS["ae_batch_size"],
    ae_patience:    int  = AE_DEFAULTS["ae_patience"],
    bic_subsample:  int  = 10_000,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_tag = channel_filter or "all_channels"

    if quick_test:
        print("\n" + "=" * 60)
        print("  QUICK TEST MODE  (smoke-test — not for claim)")
        print("=" * 60)
        if device == "cuda":
            print(f"  Device: cuda — {torch.cuda.get_device_name(0)}")
        else:
            print("  Device: cpu")
    else:
        print(f"Device: {device}")
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Step 0: Load raw windows ──────────────────────────────────────────────
    (
        windows_train_normal,
        windows_train_all,
        windows_test_all,
        y_train,
        y_test,
        meta_full,
    ) = load_raw(raw_dir, channel_filter)

    # Window size consistency check
    T = windows_train_normal.shape[1]
    if T != window_size:
        print(
            f"  [WARN] CLI --window_size={window_size} but windows.npy has T={T}. "
            f"Using T={T}."
        )
        window_size = T

    # ── Quick-test overrides ──────────────────────────────────────────────────
    seeds_run      = SEEDS
    cv_folds_run   = CV_FOLDS
    nf_params_run  = {**NF_PARAMS, "bic_subsample_size": bic_subsample}
    bootstrap_n    = BOOTSTRAP_N
    ae_epochs_run  = ae_epochs

    if quick_test:
        seeds_run     = [0]
        cv_folds_run  = 2
        bootstrap_n   = 100
        ae_epochs_run = 10
        nf_params_run = {**NF_PARAMS, "n_epochs": 10, "n_bootstrap": 3,
                         "flow_patience": 5, "bic_subsample_size": 0}

        # Subsample train_normal for AE (keep anomaly rate roughly intact)
        QT_AE = min(2_000, len(windows_train_normal))
        rng_qt = np.random.RandomState(0)
        idx_qt = rng_qt.choice(len(windows_train_normal), size=QT_AE, replace=False)
        windows_train_normal = windows_train_normal[idx_qt]

        print(f"  AE train subsample  : {len(windows_train_normal)} windows")
        print(f"  AE epochs           : {ae_epochs_run}")
        print(f"  Seeds               : {seeds_run}")
        print(f"  CV folds            : {cv_folds_run}")
        print("=" * 60)

    # ── Step 1: Train Conv1D Autoencoder ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 1 — Conv1D Autoencoder training")
    print(f"{'='*60}")
    print(f"  Architecture  : window_size={window_size}  embedding_dim={embedding_dim}")
    print(f"  Train normal  : {len(windows_train_normal):,} windows")

    # Temporal val split: last 10% of train_normal (by current order = sorted by time)
    n_val  = max(1, len(windows_train_normal) // 10)
    n_tr_ae = len(windows_train_normal) - n_val
    ae_train_wins = windows_train_normal[:n_tr_ae]
    ae_val_wins   = windows_train_normal[n_tr_ae:]

    print(f"  AE train      : {len(ae_train_wins):,}  AE val: {len(ae_val_wins):,}")

    model = ConvAutoencoder(window_size=window_size, embedding_dim=embedding_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params  : {n_params:,}")

    model = train_autoencoder(
        model,
        train_windows = ae_train_wins,
        val_windows   = ae_val_wins,
        epochs        = ae_epochs_run,
        lr            = ae_lr,
        batch_size    = ae_batch_size,
        device        = device,
        patience      = ae_patience,
    )

    # ── Step 2: Extract embeddings ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 2 — Embedding extraction")
    print(f"{'='*60}")

    t_emb0 = time.perf_counter()
    emb_train = extract_embeddings(model, windows_train_all, ae_batch_size, device)
    emb_test  = extract_embeddings(model, windows_test_all,  ae_batch_size, device)
    t_emb     = time.perf_counter() - t_emb0

    print(f"  Train embeddings : {emb_train.shape}  "
          f"({emb_train.shape[0]:,} × {emb_train.shape[1]})")
    print(f"  Test  embeddings : {emb_test.shape}")
    print(f"  Extraction time  : {t_emb:.1f}s")

    # ── Step 3: Save dataset_embeddings.csv ───────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3 — Save dataset_embeddings.csv")
    print(f"{'='*60}")

    emb_all  = np.concatenate([emb_train, emb_test], axis=0)
    meta_train = meta_full[meta_full["train"] == 1]
    meta_test  = meta_full[meta_full["train"] == 0]

    if channel_filter:
        meta_train = meta_train[meta_train["channel"] == channel_filter]
        meta_test  = meta_test[meta_test["channel"]   == channel_filter]

    # Align: meta is indexed by segment, emb rows match meta rows
    assert len(meta_train) == len(emb_train), (
        f"meta_train len {len(meta_train)} != emb_train len {len(emb_train)}"
    )
    assert len(meta_test) == len(emb_test), (
        f"meta_test len {len(meta_test)} != emb_test len {len(emb_test)}"
    )

    emb_cols = [f"emb_{i}" for i in range(embedding_dim)]
    df_train = meta_train.copy()
    df_test  = meta_test.copy()

    for i, col in enumerate(emb_cols):
        df_train[col] = emb_train[:, i]
        df_test[col]  = emb_test[:, i]

    df_emb = pd.concat([df_train, df_test], axis=0).sort_index()
    emb_path = raw_dir / "dataset_embeddings.csv"
    df_emb.to_csv(emb_path)
    print(f"  Saved: {emb_path}  ({len(df_emb):,} rows, {embedding_dim} emb dims)")

    # ── Step 4: RF baseline (supervised ceiling) ──────────────────────────────
    rf_metrics = rf_baseline_on_embeddings(emb_train, y_train, emb_test, y_test)

    # ── Step 5: LPI NF Ensemble ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 5 — LPINormalizingFlow Seed Ensemble  ({run_tag})")
    print(f"{'='*60}")
    print(f"Seeds: {seeds_run}  |  NF config: {nf_params_run}")

    # NaN/Inf guard (embeddings should be finite but be safe)
    for arr, name in [(emb_train, "emb_train"), (emb_test, "emb_test")]:
        if not np.isfinite(arr).all():
            n_bad = (~np.isfinite(arr)).sum()
            print(f"  [WARN] {n_bad} non-finite values in {name} — replacing with 0")
            arr[:] = np.where(np.isfinite(arr), arr, 0.0)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    run_name = f"ae_lpi_quicktest_{run_tag}" if quick_test else f"ae_lpi_{run_tag}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "run_tag":         run_tag,
            "quick_test":      quick_test,
            "window_size":     window_size,
            "embedding_dim":   embedding_dim,
            "ae_epochs":       ae_epochs_run,
            "ae_lr":           ae_lr,
            "ae_batch_size":   ae_batch_size,
            "ae_patience":     ae_patience,
            "ae_n_params":     n_params,
            "ae_train_normal": len(ae_train_wins),
            "n_train":         len(emb_train),
            "n_test":          len(emb_test),
            "n_features":      embedding_dim,
            "anomaly_rate_train": f"{y_train.mean():.3f}",
            "anomaly_rate_test":  f"{y_test.mean():.3f}",
            "seeds":           str(seeds_run),
            "cv_folds":        cv_folds_run,
            "n_flow_layers":   nf_params_run["n_flow_layers"],
            "flow_hidden":     nf_params_run["flow_hidden"],
            "n_epochs_lpi":    nf_params_run["n_epochs"],
        })
        mlflow.log_metrics({
            "rf_auc": rf_metrics["rf_auc"],
            "rf_f05": rf_metrics["rf_f05"],
        })

        # ── Train seeds ────────────────────────────────────────────────────────
        t_total0 = time.perf_counter()
        print(f"\n{'='*60}")
        print("STEP 5a — Individual seed training")
        print(f"{'='*60}")

        seed_results: list[dict] = []
        for seed in seeds_run:
            res = train_single_seed(
                seed, emb_train, y_train, emb_test, y_test,
                device=device, nf_params=nf_params_run, cv_folds=cv_folds_run,
            )
            seed_results.append(res)
            mlflow.log_metrics({
                f"seed{seed}_val_auc":  res["val_auc"],
                f"seed{seed}_val_f05":  res["val_f05"],
                f"seed{seed}_test_f05": res["test_f05"],
                f"seed{seed}_test_auc": res["test_auc"],
            })

        # ── Ensembles ──────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("STEP 5b — Ensemble strategies")
        print(f"{'='*60}")

        strategies = build_ensembles(seed_results, y_train, y_test)

        for name, strat in strategies.items():
            mlflow.log_metrics({
                f"ens_{name}_val_f05":  strat["val_f05"],
                f"ens_{name}_val_auc":  strat["val_auc"],
                f"ens_{name}_test_f05": strat["test_f05"],
                f"ens_{name}_test_auc": strat["test_auc"],
            })

        # ── Bootstrap CI ───────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"STEP 5c — Bootstrap CI95  (B={bootstrap_n})")
        print(f"{'='*60}")

        ci_results: dict[str, dict] = {}
        for name, strat in strategies.items():
            print(f"\n  {name} ensemble ...")
            ci = bootstrap_ci_metrics(
                strat["ens_test"], y_test, strat["threshold"],
                n_bootstrap=bootstrap_n, seed=BOOTSTRAP_MASTER_SEED,
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
            mlflow.log_metrics({
                f"ens_{name}_f05_point":    ci["f05_point"],
                f"ens_{name}_f05_ci_lower": ci["f05_ci_lower"],
                f"ens_{name}_f05_ci_upper": ci["f05_ci_upper"],
                f"ens_{name}_auc_point":    ci["auc_point"],
                f"ens_{name}_auc_ci_lower": ci["auc_ci_lower"],
                f"ens_{name}_auc_ci_upper": ci["auc_ci_upper"],
            })

    # ── Tables ────────────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_total0

    print(f"\n{'='*60}")
    print("TABLE 1 — Individual seeds (test set)")
    print(f"{'='*60}")

    seed_rows = []
    for r in seed_results:
        seed_rows.append({
            "Seed":      r["seed"],
            "Val F0.5":  f"{r['val_f05']:.3f}",
            "Val AUC":   f"{r['val_auc']:.3f}",
            "Test F0.5": f"{r['test_f05']:.3f}",
            "Test AUC":  f"{r['test_auc']:.3f}",
            "Prec":      f"{r['test_precision']:.3f}",
            "Rec":       f"{r['test_recall']:.3f}",
            "p":         r["best_p"],
        })
    seed_f05 = [r["test_f05"] for r in seed_results]
    seed_auc = [r["test_auc"] for r in seed_results]
    seed_rows.append({
        "Seed":      "mean±std",
        "Val F0.5":  "—",
        "Val AUC":   "—",
        "Test F0.5": f"{np.mean(seed_f05):.3f}±{np.std(seed_f05):.3f}",
        "Test AUC":  f"{np.mean(seed_auc):.3f}±{np.std(seed_auc):.3f}",
        "Prec": "—", "Rec": "—", "p": "—",
    })
    print(pd.DataFrame(seed_rows).set_index("Seed").to_string())

    print(f"\n{'='*60}")
    print("TABLE 2 — Ensemble + RF + S2 cross-dataset comparison")
    print(f"{'='*60}")

    rows = []
    for bname, bmet in S2_BASELINES.items():
        rows.append({
            "Model":    bname,
            "F0.5":     f"{bmet['f05']:.3f}",
            "F0.5 CI95": "— (S2 OPS-SAT-AD)",
            "AUC":      f"{bmet['auc']:.3f}",
            "Note":     "S2 reference",
        })
    rows.append({
        "Model":     f"RF on embeddings (supervised ceiling)",
        "F0.5":      f"{rf_metrics['rf_f05']:.3f}",
        "F0.5 CI95": "—",
        "AUC":       f"{rf_metrics['rf_auc']:.3f}",
        "Note":      "supervised",
    })

    display_names = {
        "mean":   f"AE+NF ensemble (mean,   S3 {run_tag})",
        "median": f"AE+NF ensemble (median, S3 {run_tag})",
        "rank":   f"AE+NF ensemble (rank,   S3 {run_tag})",
    }
    for sname, dname in display_names.items():
        strat = strategies[sname]
        ci    = ci_results[sname]
        rows.append({
            "Model":     dname,
            "F0.5":      f"{ci['f05_point']:.3f}",
            "F0.5 CI95": f"[{ci['f05_ci_lower']:.3f}, {ci['f05_ci_upper']:.3f}]",
            "AUC":       f"{ci['auc_point']:.3f}",
            "Note":      f"p{strat['best_p']}  val_F0.5={strat['val_f05']:.3f}",
        })

    print(pd.DataFrame(rows).set_index("Model").to_string())

    # ── Winner ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("WINNER SELECTION")
    print(f"{'='*60}")

    best_strat = max(
        strategies.keys(),
        key=lambda s: (strategies[s]["val_f05"], -ci_results[s]["f05_std"]),
    )
    winner = strategies[best_strat]
    w_ci   = ci_results[best_strat]

    print(f"\n  Best strategy : {best_strat}")
    print(f"  Val  F0.5={winner['val_f05']:.3f}  AUC={winner['val_auc']:.3f}")
    print(
        f"  Test F0.5={w_ci['f05_point']:.3f}  "
        f"CI95=[{w_ci['f05_ci_lower']:.3f}, {w_ci['f05_ci_upper']:.3f}]"
    )
    print(
        f"  Test AUC={w_ci['auc_point']:.3f}  "
        f"CI95=[{w_ci['auc_ci_lower']:.3f}, {w_ci['auc_ci_upper']:.3f}]"
    )
    print(f"\n  RF ceiling (supervised) : AUC={rf_metrics['rf_auc']:.3f}  "
          f"F0.5={rf_metrics['rf_f05']:.3f}")
    gap_auc = w_ci["auc_point"] - rf_metrics["rf_auc"]
    print(f"  AE+LPI vs RF ceiling    : ΔAUC={gap_auc:+.3f}")

    if quick_test:
        print(f"\n{'='*60}")
        print("QUICK TEST SUMMARY")
        print(f"{'='*60}")
        print(f"  Total wall time : {t_total:.1f}s")
        print(f"  Device          : {device}")
        print(f"  AE params       : {n_params:,}")
        print(f"  Pipeline OK — ready for full run.")
        return

    # ── Claim ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"CLAIM S3 AE+LPI ({run_tag})")
    print(f"{'='*60}")
    print(
        f"\n  AE→embedding(dim={embedding_dim}) + "
        f"LPINormalizingFlow ensemble ({best_strat}, {len(seeds_run)} seeds),"
        f"\n  F0.5={w_ci['f05_point']:.3f} (CI95=[{w_ci['f05_ci_lower']:.3f}, {w_ci['f05_ci_upper']:.3f}]),"
        f"\n  AUC={w_ci['auc_point']:.3f} (CI95=[{w_ci['auc_ci_lower']:.3f}, {w_ci['auc_ci_upper']:.3f}]),"
        f"\n  ESA-Mission1 {run_tag}, one-shot test, GroupKFold threshold selection"
        f"\n  RF ceiling: AUC={rf_metrics['rf_auc']:.3f}  (supervised)"
        f"\n  Total time: {t_total:.0f}s"
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="S3 — Conv1D Autoencoder → Embedding → LPI pipeline",
    )
    parser.add_argument(
        "--raw_dir", required=True,
        help="Output directory from prepare_mission1_raw.py (contains windows.npy + meta.csv)",
    )
    parser.add_argument(
        "--channel", default=None, metavar="CHANNEL_NAME",
        help="Filter to a single channel (e.g. 'channel_14'). Default: all channels.",
    )
    parser.add_argument("--window_size",    type=int,   default=AE_DEFAULTS["window_size"])
    parser.add_argument("--embedding_dim",  type=int,   default=AE_DEFAULTS["embedding_dim"])
    parser.add_argument("--ae_epochs",      type=int,   default=AE_DEFAULTS["ae_epochs"])
    parser.add_argument("--ae_lr",          type=float, default=AE_DEFAULTS["ae_lr"])
    parser.add_argument("--ae_batch_size",  type=int,   default=AE_DEFAULTS["ae_batch_size"])
    parser.add_argument("--ae_patience",    type=int,   default=AE_DEFAULTS["ae_patience"])
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Smoke-test: 2k AE windows, 10 AE epochs, 1 seed, 2-fold CV. Not for claim.",
    )
    parser.add_argument(
        "--bic_subsample", type=int, default=10_000,
        help="Max samples used for BIC selection in LPINormalizingFlow (0 = all). "
             "Default 10000 gives same K as full-data but ~3× faster.",
    )
    args = parser.parse_args()

    run(
        raw_dir        = Path(args.raw_dir),
        channel_filter = args.channel,
        quick_test     = args.quick_test,
        window_size    = args.window_size,
        embedding_dim  = args.embedding_dim,
        ae_epochs      = args.ae_epochs,
        ae_lr          = args.ae_lr,
        ae_batch_size  = args.ae_batch_size,
        ae_patience    = args.ae_patience,
        bic_subsample  = args.bic_subsample,
    )


if __name__ == "__main__":
    main()
