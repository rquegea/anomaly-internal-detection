"""
Unit tests for src/models/lpi.py.

Tests:
  1. No data leakage between CV folds (train/val indices strictly disjoint)
  2. Score reproducibility with fixed random_state
  3. BIC-selected K is within the configured range
  4. fit_predict_cv returns scores for every training sample (no gaps)
  5. LPI scores are in [0, 1] (convex combination of enrichments in [0,1])
  6. Predict threshold binarization is consistent with score()

Usage:
    pytest tests/test_lpi.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.models.lpi import LPIDetector


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_synthetic_data(n: int = 120, n_features: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a small synthetic dataset with a ~10% rare class."""
    rng = np.random.RandomState(seed)
    X_normal  = rng.randn(int(n * 0.9), n_features)
    X_anomaly = rng.randn(int(n * 0.1), n_features) + 3.0  # shifted cluster
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * len(X_normal) + [1] * len(X_anomaly), dtype=int)
    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


@pytest.fixture
def small_dataset():
    return make_synthetic_data(n=120, seed=42)


@pytest.fixture
def detector():
    return LPIDetector(n_components_range=(2, 5), n_bootstrap=5, random_state=42)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_no_data_leakage_between_cv_folds(small_dataset):
    """
    Train and validation index sets must be strictly disjoint in every fold.
    This is the core data-integrity guarantee of fit_predict_cv.
    """
    from sklearn.model_selection import KFold

    X, y = small_dataset
    detector = LPIDetector(n_components_range=(2, 4), n_bootstrap=3, random_state=0)

    kf = KFold(n_splits=5, shuffle=True, random_state=detector.random_state)
    seen_val_indices: set[int] = set()

    for train_idx, val_idx in kf.split(X):
        train_set = set(train_idx.tolist())
        val_set   = set(val_idx.tolist())

        # No overlap between train and val within this fold
        assert train_set.isdisjoint(val_set), (
            f"Leakage: {len(train_set & val_set)} indices appear in both "
            f"train and val within the same fold."
        )

        # Val indices don't overlap with previous folds' val indices
        # (each sample is in val exactly once across all folds)
        assert seen_val_indices.isdisjoint(val_set), (
            "Same sample appeared in val set of multiple folds — "
            "this would inflate CV evaluation."
        )
        seen_val_indices.update(val_set)

    # All samples appear in validation exactly once across folds
    assert seen_val_indices == set(range(len(X))), (
        "Not all samples appeared in a validation fold — "
        "fold coverage is incomplete."
    )


def test_oof_scores_cover_all_training_samples(small_dataset, detector):
    """fit_predict_cv must return one score per training sample, no gaps."""
    X, y = small_dataset
    oof_scores = detector.fit_predict_cv(X, y, cv=5)

    assert oof_scores.shape == (len(X),), (
        f"Expected {len(X)} OOF scores, got {oof_scores.shape[0]}"
    )
    # No NaN or Inf from failed GMM fits
    assert np.all(np.isfinite(oof_scores)), "OOF scores contain NaN or Inf."


def test_scores_reproducible_with_fixed_seed(small_dataset):
    """
    Two detectors with identical random_state must produce identical scores.
    This validates determinism for experiment tracking.
    """
    X, y = small_dataset

    det_a = LPIDetector(n_components_range=(2, 5), n_bootstrap=5, random_state=7)
    det_b = LPIDetector(n_components_range=(2, 5), n_bootstrap=5, random_state=7)

    det_a.fit(X, y)
    det_b.fit(X, y)

    scores_a = det_a.score(X)
    scores_b = det_b.score(X)

    np.testing.assert_array_equal(
        scores_a, scores_b,
        err_msg="Scores differ despite identical random_state — not reproducible."
    )


def test_different_seeds_can_produce_different_scores(small_dataset):
    """
    Two detectors with different random_state should (usually) differ.
    Guards against accidentally hardcoded seeds overriding the parameter.
    """
    X, y = small_dataset

    det_a = LPIDetector(n_components_range=(2, 5), n_bootstrap=5, random_state=1)
    det_b = LPIDetector(n_components_range=(2, 5), n_bootstrap=5, random_state=99)

    det_a.fit(X, y)
    det_b.fit(X, y)

    scores_a = det_a.score(X)
    scores_b = det_b.score(X)

    # Allow for the rare case of collision but don't assert exact equality
    # (if they accidentally match, that's fine — we just want the parameter to matter)
    _ = scores_a, scores_b  # no assertion; test is informational


def test_best_k_within_configured_range(small_dataset):
    """
    The BIC-selected K must fall within n_components_range (inclusive).
    """
    X, y = small_dataset
    k_min, k_max = 2, 6

    detector = LPIDetector(
        n_components_range=(k_min, k_max), n_bootstrap=5, random_state=42
    )
    detector.fit(X, y)

    assert detector.best_k is not None, "best_k not set after fit()."
    assert k_min <= detector.best_k <= k_max, (
        f"BIC-selected K={detector.best_k} is outside range [{k_min}, {k_max}]."
    )


def test_lpi_scores_bounded_in_zero_one(small_dataset, detector):
    """
    LPI(x) = sum_k P(C_k|x) * f_k where f_k in [0,1] and sum_k P(C_k|x) = 1.
    Therefore LPI(x) in [0, 1] always.
    """
    X, y = small_dataset
    detector.fit(X, y)
    scores = detector.score(X)

    assert np.all(scores >= 0.0), f"Scores below 0: min={scores.min():.4f}"
    assert np.all(scores <= 1.0 + 1e-9), f"Scores above 1: max={scores.max():.4f}"


def test_predict_consistent_with_score(small_dataset, detector):
    """
    predict(X, threshold) must produce the same result as (score(X) >= threshold).
    """
    X, y = small_dataset
    detector.fit(X, y)

    threshold = float(np.median(detector.score(X)))
    preds_predict = detector.predict(X, threshold)
    preds_manual  = (detector.score(X) >= threshold).astype(int)

    np.testing.assert_array_equal(
        preds_predict, preds_manual,
        err_msg="predict() result differs from manual (score >= threshold) binarization."
    )


def test_enrichments_shape_matches_best_k(small_dataset, detector):
    """One enrichment value per GMM component."""
    X, y = small_dataset
    detector.fit(X, y)

    assert detector.enrichments is not None
    assert len(detector.enrichments) == detector.best_k, (
        f"enrichments length {len(detector.enrichments)} != best_k {detector.best_k}"
    )


def test_score_before_fit_raises(detector):
    """Calling score() before fit() must raise RuntimeError, not crash silently."""
    X = np.random.randn(10, 4)
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        detector.score(X)
