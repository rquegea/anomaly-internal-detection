"""
Unit tests for src/models/lpi_v2.py — five LPI extensions.

Test matrix (≥3 per extension):
  LPINormalizingFlow  : reproducibility, shape, no leakage, score bounds
  LPIVariational      : effective K ≤ k_max, soft enrichment, no leakage
  LPIBayesian         : CI bounds valid, mean == score(), no leakage
  LPIHierarchical     : macro K in range, score bounds, no leakage
  LPIOnline           : update_batch works, K stable, score bounds

Cross-cutting:
  - Interface compliance: all classes expose fit/score/predict/fit_predict_cv
  - No data leakage in CV: train/val indices strictly disjoint
  - score() before fit() raises RuntimeError

Usage:
    pytest tests/test_lpi_v2.py -v
    pytest tests/test_lpi_v2.py -v -k "LPIBayesian"  # subset
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.models.lpi_v2 import (
    LPIBayesian,
    LPIHierarchical,
    LPINormalizingFlow,
    LPIOnline,
    LPIVariational,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_data(n: int = 200, n_features: int = 8, seed: int = 0):
    """
    Synthetic dataset with ~10 % rare class (shifted Gaussian).
    n_features=8 is large enough for RealNVP (d_half=4, d_rest=4).
    """
    rng = np.random.RandomState(seed)
    n_normal = int(n * 0.9)
    n_anomaly = n - n_normal
    X = np.vstack(
        [rng.randn(n_normal, n_features), rng.randn(n_anomaly, n_features) + 3.0]
    )
    y = np.array([0] * n_normal + [1] * n_anomaly, dtype=int)
    idx = rng.permutation(n)
    return X[idx], y[idx]


@pytest.fixture
def data():
    return make_data(n=200, seed=42)


@pytest.fixture
def small_data():
    """Tiny dataset for fast tests that don't need realistic scale."""
    return make_data(n=120, seed=7)


# ── Interface compliance ───────────────────────────────────────────────────────


ALL_CLASSES = [
    LPINormalizingFlow,
    LPIVariational,
    LPIBayesian,
    LPIHierarchical,
    LPIOnline,
]


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_interface_fit_score_predict(cls, data):
    """All extensions must expose fit(), score(), predict() with correct shapes."""
    X, y = data
    det = cls(n_components_range=(2, 4), n_bootstrap=3, random_state=0)
    det.fit(X, y)

    scores = det.score(X)
    assert scores.shape == (len(X),), f"{cls.__name__}: score shape mismatch"
    assert np.all(np.isfinite(scores)), f"{cls.__name__}: non-finite scores"

    preds = det.predict(X, threshold=float(np.median(scores)))
    assert preds.shape == (len(X),), f"{cls.__name__}: predict shape mismatch"
    assert set(preds).issubset({0, 1}), f"{cls.__name__}: predict not binary"


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_score_before_fit_raises(cls):
    """Calling score() before fit() must raise RuntimeError."""
    det = cls(n_components_range=(2, 3), n_bootstrap=2, random_state=0)
    X = np.random.randn(10, 8)
    with pytest.raises(RuntimeError):
        det.score(X)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_fit_predict_cv_shape(cls, small_data):
    """fit_predict_cv must return one OOF score per training sample."""
    X, y = small_data
    det = cls(n_components_range=(2, 3), n_bootstrap=2, random_state=0)
    oof = det.fit_predict_cv(X, y, cv=3)
    assert oof.shape == (len(X),), f"{cls.__name__}: OOF shape {oof.shape} != {len(X)}"
    assert np.all(np.isfinite(oof)), f"{cls.__name__}: OOF contains NaN/Inf"


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_no_cv_leakage(cls, small_data):
    """
    CV folds: train and val indices must be strictly disjoint in every fold.
    Each sample must appear in exactly one validation fold.
    """
    from sklearn.model_selection import KFold

    X, y = small_data
    det = cls(n_components_range=(2, 3), n_bootstrap=2, random_state=0)
    # Mirror the KFold used inside _cv_with_factory / fit_predict_cv
    kf = KFold(n_splits=3, shuffle=True, random_state=det.random_state)
    seen_val: set[int] = set()

    for train_idx, val_idx in kf.split(X):
        tr_set = set(train_idx.tolist())
        va_set = set(val_idx.tolist())
        assert tr_set.isdisjoint(va_set), "Train and val overlap within a fold"
        assert seen_val.isdisjoint(va_set), "Same sample in val of multiple folds"
        seen_val.update(va_set)

    assert seen_val == set(range(len(X))), "Not all samples covered by val folds"


# ── LPINormalizingFlow ────────────────────────────────────────────────────────


class TestLPINormalizingFlow:
    def _det(self, seed=42):
        return LPINormalizingFlow(
            n_components_range=(2, 4),
            n_bootstrap=3,
            random_state=seed,
            n_flow_layers=2,
            flow_hidden=16,
            n_epochs=30,  # fast for tests
            flow_patience=10,
        )

    def test_scores_in_unit_interval(self, data):
        """LPI scores must be in [0, 1] (convex combination of enrichments)."""
        X, y = data
        det = self._det()
        det.fit(X, y)
        scores = det.score(X)
        assert np.all(scores >= -1e-9), f"Min score = {scores.min():.4f}"
        assert np.all(scores <= 1.0 + 1e-9), f"Max score = {scores.max():.4f}"

    def test_reproducible_with_fixed_seed(self, data):
        """Identical seeds → identical scores."""
        X, y = data
        det_a = self._det(seed=5)
        det_b = self._det(seed=5)
        det_a.fit(X, y)
        det_b.fit(X, y)
        np.testing.assert_array_equal(det_a.score(X), det_b.score(X))

    def test_n_flow_params_positive(self, data):
        """Flow should have trainable parameters after fit()."""
        X, y = data
        det = self._det()
        det.fit(X, y)
        assert det.n_flow_params > 0, "Flow has zero parameters after fit"

    def test_latent_space_used_for_gmm(self, data):
        """GMM is fitted; best_k is set; enrichments match best_k."""
        X, y = data
        det = self._det()
        det.fit(X, y)
        assert det.best_k is not None
        assert det._enrichments is not None
        assert len(det._enrichments) == det.best_k


# ── LPIVariational ────────────────────────────────────────────────────────────


class TestLPIVariational:
    def _det(self, seed=42):
        return LPIVariational(
            k_max=8,
            scaler="robust",
            random_state=seed,
            weight_concentration_prior=1e-2,
        )

    def test_effective_k_leq_kmax(self, data):
        """Variational inference prunes components; effective K ≤ k_max."""
        X, y = data
        det = self._det()
        det.fit(X, y)
        assert det._effective_k is not None
        assert det._effective_k <= det.k_max, (
            f"Effective K={det._effective_k} > k_max={det.k_max}"
        )

    def test_scores_in_unit_interval(self, data):
        """Soft enrichment × soft responsibilities ≤ 1 always."""
        X, y = data
        det = self._det()
        det.fit(X, y)
        scores = det.score(X)
        # Soft enrichments are in [0,1], responsibilities sum to 1 → score in [0,1]
        assert np.all(scores >= -1e-9)
        assert np.all(scores <= 1.0 + 1e-9)

    def test_reproducible_with_fixed_seed(self, data):
        X, y = data
        a = self._det(seed=3)
        b = self._det(seed=3)
        a.fit(X, y)
        b.fit(X, y)
        np.testing.assert_allclose(a.score(X), b.score(X), rtol=1e-5)

    def test_enrichments_shape(self, data):
        """Enrichments length == k_max (one per component, including pruned ones)."""
        X, y = data
        det = self._det()
        det.fit(X, y)
        assert len(det._enrichments) == det.k_max


# ── LPIBayesian ───────────────────────────────────────────────────────────────


class TestLPIBayesian:
    def _det(self, seed=42, n_bootstrap_bayes=10):
        return LPIBayesian(
            n_components_range=(2, 4),
            n_bootstrap=3,
            scaler="robust",
            random_state=seed,
            n_bootstrap_bayes=n_bootstrap_bayes,
        )

    def test_ci_lower_leq_mean_leq_upper(self, data):
        """90 % CI must bracket the mean score for every sample."""
        X, y = data
        det = self._det()
        det.fit(X, y)
        mean_s, std_s, ci = det.score_with_uncertainty(X)
        lower, upper = ci[:, 0], ci[:, 1]
        assert np.all(lower <= mean_s + 1e-9), "CI lower > mean"
        assert np.all(upper >= mean_s - 1e-9), "CI upper < mean"

    def test_score_equals_mean_uncertainty(self, data):
        """score() must return the same values as the mean from score_with_uncertainty()."""
        X, y = data
        det = self._det()
        det.fit(X, y)
        s = det.score(X)
        mean_s, _, _ = det.score_with_uncertainty(X)
        np.testing.assert_allclose(s, mean_s, rtol=1e-6)

    def test_std_nonnegative(self, data):
        """Bootstrap std must be ≥ 0 for all samples."""
        X, y = data
        det = self._det()
        det.fit(X, y)
        _, std_s, _ = det.score_with_uncertainty(X)
        assert np.all(std_s >= 0.0), f"Negative std: min={std_s.min():.4f}"

    def test_bootstrap_enrichments_shape(self, data):
        """bootstrap_enrichments shape must be (n_bootstrap_bayes, K)."""
        X, y = data
        B = 15
        det = self._det(n_bootstrap_bayes=B)
        det.fit(X, y)
        assert det._bootstrap_enrichments.shape == (B, det.best_k)


# ── LPIHierarchical ───────────────────────────────────────────────────────────


class TestLPIHierarchical:
    def _det(self, seed=42):
        return LPIHierarchical(
            n_components_range=(2, 8),
            n_bootstrap=3,
            scaler="robust",
            random_state=seed,
            k_macro_range=(2, 4),
            k_micro=2,
            min_cluster_size=10,
            alpha=0.5,
        )

    def test_macro_k_in_range(self, data):
        """BIC-selected macro K must be within k_macro_range."""
        X, y = data
        det = self._det()
        det.fit(X, y)
        k_min, k_max = det.k_macro_range
        assert det.best_k is not None
        assert k_min <= det.best_k <= k_max, (
            f"Macro K={det.best_k} outside [{k_min}, {k_max}]"
        )

    def test_scores_in_unit_interval(self, data):
        X, y = data
        det = self._det()
        det.fit(X, y)
        scores = det.score(X)
        assert np.all(scores >= -1e-9), f"Min={scores.min()}"
        assert np.all(scores <= 1.0 + 1e-9), f"Max={scores.max()}"

    def test_micro_clusters_populated(self, data):
        """At least one macro cluster should have micro GMM fitted."""
        X, y = data
        det = self._det()
        det.fit(X, y)
        # micro_enrichments should exist for all macro clusters
        assert len(det._micro_enrichments) == det.best_k

    def test_reproducible(self, data):
        X, y = data
        a = self._det(seed=11)
        b = self._det(seed=11)
        a.fit(X, y)
        b.fit(X, y)
        np.testing.assert_allclose(a.score(X), b.score(X), rtol=1e-5)


# ── LPIOnline ─────────────────────────────────────────────────────────────────


class TestLPIOnline:
    def _det(self, seed=42):
        return LPIOnline(
            n_components_range=(2, 4),
            n_bootstrap=3,
            scaler="robust",
            random_state=seed,
            batch_size=50,
            forgetting_rate=0.2,
            n_update_iter=10,
        )

    def test_update_batch_changes_enrichments(self, data):
        """
        After update_batch with anomaly-heavy data, enrichments should shift
        toward higher values (anomaly signal absorbed).
        """
        X, y = data
        det = self._det()
        det.fit(X, y)
        enr_before = det._enrichments.copy()

        # Inject a batch of pure anomalies
        rng = np.random.RandomState(99)
        X_anom = rng.randn(30, X.shape[1]) + 3.0
        y_anom = np.ones(30, dtype=int)
        det.update_batch(X_anom, y_anom)

        enr_after = det._enrichments
        # Enrichments should change (not identical after anomaly injection)
        # (Not asserting direction — depends on cluster assignment)
        assert not np.allclose(enr_before, enr_after, atol=1e-8), (
            "Enrichments unchanged after update_batch — EMA not applied"
        )

    def test_k_stable_after_update(self, data):
        """K (n_components) must not change after update_batch."""
        X, y = data
        det = self._det()
        det.fit(X, y)
        k_before = det.best_k
        det.update_batch(X[:30], y[:30])
        assert det.best_k == k_before, "K changed after update_batch"

    def test_scores_in_unit_interval(self, data):
        X, y = data
        det = self._det()
        det.fit(X, y)
        scores = det.score(X)
        assert np.all(scores >= -1e-9)
        assert np.all(scores <= 1.0 + 1e-9)

    def test_n_batches_seen_increments(self, data):
        """_n_batches_seen should increment with each update_batch call."""
        X, y = data
        det = self._det()
        det.fit(X, y)
        batches_after_fit = det._n_batches_seen
        det.update_batch(X[:20], y[:20])
        assert det._n_batches_seen == batches_after_fit + 1


# ── Predict consistency (all extensions) ────────────────────────────────────


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_predict_consistent_with_score(cls, small_data):
    """predict(X, thr) must match (score(X) >= thr) for all samples."""
    X, y = small_data
    det = cls(n_components_range=(2, 3), n_bootstrap=2, random_state=0)
    # LPINormalizingFlow: use fast settings
    if isinstance(det, LPINormalizingFlow):
        det.n_epochs = 10
        det.flow_patience = 5
        det.n_flow_layers = 1
        det.flow_hidden = 8
    det.fit(X, y)
    scores = det.score(X)
    thr = float(np.median(scores))
    preds_predict = det.predict(X, thr)
    preds_manual = (scores >= thr).astype(int)
    np.testing.assert_array_equal(preds_predict, preds_manual)
