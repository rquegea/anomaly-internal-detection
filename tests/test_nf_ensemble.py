"""
Tests for run_nf_seed_ensemble.py

Scope
─────
  T1 — Sanity: 5 different seeds produce distinct OOF score distributions.
  T2 — Ensemble on identical scores: mean == median == rank (after normalization).
  T3 — Bootstrap CI95 always brackets the point estimate.
  T4 — Ensemble AUC >= min individual seed AUC − epsilon (ensemble doesn't catastrophically degrade).

These tests use tiny synthetic data + few epochs to stay fast (<60s total).
Real-data end-to-end validation is via the experiment script.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

# Add project root to path
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

# ─── Load experiment module ────────────────────────────────────────────────────


def _load_mod():
    spec = importlib.util.spec_from_file_location(
        "run_nf_seed_ensemble",
        ROOT / "experiments" / "s2_lpi_v2" / "run_nf_seed_ensemble.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_mod()

normalize_minmax = _mod.normalize_minmax
to_fractional_ranks = _mod.to_fractional_ranks
bootstrap_ci_metrics = _mod.bootstrap_ci_metrics
select_threshold_oof = _mod.select_threshold_oof


# ─── Synthetic data ───────────────────────────────────────────────────────────


def _make_toy_data(n_normal: int = 80, n_anomaly: int = 20, n_features: int = 16, seed: int = 0):
    """Small synthetic dataset with 16 features, linearly separable."""
    rng = np.random.default_rng(seed)
    X_normal  = rng.standard_normal((n_normal, n_features))
    X_anomaly = rng.standard_normal((n_anomaly, n_features)) + 2.5
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * n_normal + [1] * n_anomaly, dtype=int)
    return X, y


# ─── T1: Different seeds produce different models ─────────────────────────────


def test_five_seeds_produce_different_oof_scores():
    """
    Training with 5 different seeds should yield non-identical OOF score vectors.
    Tests that model training is stochastic (RealNVP weight init differs per seed).
    """
    from src.models.lpi_v2 import LPINormalizingFlow

    X_train, y_train = _make_toy_data(n_normal=100, n_anomaly=30)

    fast_params = dict(
        n_components_range=(2, 5),
        n_bootstrap=5,
        scaler="robust",
        n_flow_layers=2,
        flow_hidden=16,
        n_epochs=5,   # fast for tests
        flow_lr=1e-3,
        flow_patience=10,
    )

    oof_per_seed = []
    for seed in [0, 1, 42, 123, 999]:
        det = LPINormalizingFlow(**fast_params, random_state=seed)
        oof = det.fit_predict_cv(X_train, y_train, cv=3)
        oof_per_seed.append(oof)

    # Check that not all seeds produce identical OOF scores
    all_oof = np.stack(oof_per_seed, axis=1)  # (n_train, 5)
    cross_seed_std = float(all_oof.std(axis=1).mean())
    assert cross_seed_std > 1e-6, (
        f"All 5 seeds produced identical OOF scores (std={cross_seed_std:.2e}). "
        "Model training may not be seed-dependent."
    )


# ─── T2: Ensemble on identical inputs ─────────────────────────────────────────


def test_ensemble_identical_scores_same_ordering():
    """
    When all 5 seeds return the same scores, mean/median/rank ensembles
    must agree on the full ranking (Spearman rho = 1.0).

    Min-max and rank produce different *values* (min-max preserves distances,
    ranks only preserve ordinal order), so we check ordering equivalence,
    not value equality.
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    base_scores = rng.uniform(0, 1, size=50)

    # Simulate 5 seeds with identical scores
    oof_list  = [base_scores.copy() for _ in range(5)]
    test_list = [base_scores.copy() for _ in range(5)]

    # Mean ensemble (min-max normalised)
    norm_oof  = np.stack([normalize_minmax(o, o) for o in oof_list], axis=1)
    mean_ens  = norm_oof.mean(axis=1)
    median_ens = np.median(norm_oof, axis=1)

    # Rank ensemble
    rank_oof  = np.stack([to_fractional_ranks(o) for o in oof_list], axis=1)
    rank_ens  = rank_oof.mean(axis=1)

    # Mean and median should be numerically identical (same inputs)
    np.testing.assert_allclose(mean_ens, median_ens, atol=1e-10,
                               err_msg="mean != median on identical inputs")

    # Mean and rank may differ in *values* but must agree on ordering
    rho_mean_rank, _ = spearmanr(mean_ens, rank_ens)
    assert rho_mean_rank == pytest.approx(1.0, abs=1e-9), (
        f"mean and rank ensembles disagree on ordering (Spearman rho={rho_mean_rank:.6f}) "
        "on identical inputs — aggregation is not order-consistent"
    )


# ─── T3: Bootstrap CI95 contains point estimate ───────────────────────────────


def test_bootstrap_ci_contains_point_estimate():
    """
    The bootstrap CI95 [lower, upper] should always bracket the full-test point estimate.
    This is a distributional property of the percentile bootstrap.
    """
    rng = np.random.default_rng(7)
    n = 200
    scores = rng.uniform(0, 1, size=n)
    y = (scores > 0.6).astype(int)  # artificial labels correlated with scores

    threshold = 0.6  # fixed threshold (would come from OOF in production)

    ci = bootstrap_ci_metrics(
        scores, y, threshold, n_bootstrap=500, seed=42
    )

    assert ci["f05_ci_lower"] <= ci["f05_point"] + 1e-9, (
        f"CI95 lower ({ci['f05_ci_lower']:.4f}) > point estimate ({ci['f05_point']:.4f})"
    )
    assert ci["f05_ci_upper"] >= ci["f05_point"] - 1e-9, (
        f"CI95 upper ({ci['f05_ci_upper']:.4f}) < point estimate ({ci['f05_point']:.4f})"
    )
    assert ci["auc_ci_lower"] <= ci["auc_point"] + 1e-9
    assert ci["auc_ci_upper"] >= ci["auc_point"] - 1e-9
    assert 0 < ci["n_valid_boots"] <= 500


# ─── T4: Ensemble doesn't catastrophically degrade vs individual seeds ─────────


def test_ensemble_auc_not_worse_than_worst_individual():
    """
    Ensemble of 5 seeds should achieve AUC >= (min individual AUC - 0.10).
    Guards against degenerate aggregation that inverts scores.
    Uses tiny models (5 epochs) on linearly separable synthetic data.
    """
    from src.models.lpi_v2 import LPINormalizingFlow

    X, y = _make_toy_data(n_normal=120, n_anomaly=40, seed=5)
    n = len(X)
    split = int(0.7 * n)
    idx = np.random.default_rng(0).permutation(n)
    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_test,  y_test  = X[idx[split:]], y[idx[split:]]

    fast_params = dict(
        n_components_range=(2, 5),
        n_bootstrap=5,
        scaler="robust",
        n_flow_layers=2,
        flow_hidden=16,
        n_epochs=5,
        flow_lr=1e-3,
        flow_patience=10,
    )

    oof_list, test_list, seed_aucs = [], [], []
    for seed in [0, 1, 42, 123, 999]:
        det = LPINormalizingFlow(**fast_params, random_state=seed)
        oof = det.fit_predict_cv(X_train, y_train, cv=3)
        det.fit(X_train, y_train)
        ts = det.score(X_test)
        oof_list.append(oof)
        test_list.append(ts)
        if len(np.unique(y_test)) > 1:
            seed_aucs.append(float(roc_auc_score(y_test, ts)))

    # Build mean ensemble
    norm_oof  = np.stack([normalize_minmax(o, o) for o in oof_list], axis=1)
    norm_test = np.stack([normalize_minmax(o, t) for o, t in zip(oof_list, test_list)], axis=1)
    ens_oof   = norm_oof.mean(axis=1)
    ens_test  = norm_test.mean(axis=1)

    if len(np.unique(y_test)) > 1:
        ens_auc = float(roc_auc_score(y_test, ens_test))
        min_individual_auc = min(seed_aucs)
        assert ens_auc >= min_individual_auc - 0.10, (
            f"Ensemble AUC ({ens_auc:.3f}) is more than 0.10 below "
            f"worst individual ({min_individual_auc:.3f}). "
            "Aggregation may be inverting scores."
        )
    else:
        pytest.skip("Degenerate test split — all same label")


# ─── T5: normalize_minmax properties ─────────────────────────────────────────


def test_normalize_minmax_bounds():
    """Normalised reference scores should be in [0, 1]."""
    rng = np.random.default_rng(99)
    scores = rng.uniform(-5, 5, size=100)
    normed = normalize_minmax(scores, scores)
    assert float(normed.min()) >= 0.0 - 1e-9
    assert float(normed.max()) <= 1.0 + 1e-9


def test_normalize_minmax_constant_ref():
    """Constant reference should return zeros (not NaN)."""
    ref = np.ones(10)
    target = np.array([0.5, 1.0, 1.5])
    result = normalize_minmax(ref, target)
    assert not np.any(np.isnan(result))
    np.testing.assert_array_equal(result, np.zeros_like(target))


# ─── T6: to_fractional_ranks properties ──────────────────────────────────────


def test_fractional_ranks_bounds():
    """Ranks should be in [0, 1] with min=0 and max=1 for length > 1."""
    rng = np.random.default_rng(3)
    scores = rng.uniform(0, 1, size=50)
    ranks = to_fractional_ranks(scores)
    assert float(ranks.min()) >= 0.0
    assert float(ranks.max()) <= 1.0
    # Min and max should be exactly 0 and 1 (no ties)
    assert float(ranks.min()) == pytest.approx(0.0, abs=1e-9)
    assert float(ranks.max()) == pytest.approx(1.0, abs=1e-9)


def test_fractional_ranks_preserves_order():
    """Higher score → higher rank."""
    scores = np.array([0.1, 0.5, 0.3, 0.9, 0.7])
    ranks = to_fractional_ranks(scores)
    # Relative order must be preserved
    for i in range(len(scores)):
        for j in range(len(scores)):
            if scores[i] < scores[j]:
                assert ranks[i] < ranks[j], (
                    f"Order violated: score[{i}]={scores[i]} < score[{j}]={scores[j]} "
                    f"but rank[{i}]={ranks[i]} >= rank[{j}]={ranks[j]}"
                )
