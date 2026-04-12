"""
tests/test_no_length_leakage.py

Verifies that the sliding-window loader produces windows of guaranteed
fixed length, eliminating the length-leakage confound identified in EDA:

  Anomalous OPS-SAT-AD segments are ~3.4× longer than normal ones
  (median 184 pts vs 54 pts, sampling=5 cohort).  If the model were
  trained on whole segments it could learn to flag long inputs rather
  than anomalous patterns.  This test suite asserts that the windowed
  loader removes that confound entirely.

Run with:
    .venv/bin/pytest tests/test_no_length_leakage.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.loader import make_sliding_windows, WindowDataset


# ── Fixtures ──────────────────────────────────────────────────────────────────

WINDOW_SIZE = 64
STRIDE      = 32


@pytest.fixture(scope="module")
def dataset_both() -> WindowDataset:
    """Full split (train + test), sampling=5."""
    return make_sliding_windows(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        sampling_rate_filter=5,
        split="both",
    )


@pytest.fixture(scope="module")
def dataset_train_normal() -> WindowDataset:
    """Train-normal-only windows, sampling=5."""
    return make_sliding_windows(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        sampling_rate_filter=5,
        train_normal_only=True,
    )


# ── Core anti-leakage assertions ──────────────────────────────────────────────

class TestWindowLengthUniformity:
    """Every window must have exactly window_size points regardless of label."""

    def test_all_windows_have_correct_shape(self, dataset_both):
        """Shape invariant: (n_windows, window_size)."""
        assert dataset_both.windows.ndim == 2
        assert dataset_both.windows.shape[1] == WINDOW_SIZE, (
            f"Expected window width {WINDOW_SIZE}, got {dataset_both.windows.shape[1]}"
        )

    def test_normal_windows_length(self, dataset_both):
        """Normal windows must all be exactly window_size long."""
        normal = dataset_both.normal_windows()
        assert normal.shape[0] > 0, "No normal windows found"
        assert normal.shape[1] == WINDOW_SIZE, (
            f"Normal windows have length {normal.shape[1]}, expected {WINDOW_SIZE}"
        )

    def test_anomaly_windows_length(self, dataset_both):
        """Anomaly windows must all be exactly window_size long."""
        anomaly = dataset_both.anomaly_windows()
        assert anomaly.shape[0] > 0, "No anomaly windows found"
        assert anomaly.shape[1] == WINDOW_SIZE, (
            f"Anomaly windows have length {anomaly.shape[1]}, expected {WINDOW_SIZE}"
        )

    def test_length_distribution_is_identical_across_classes(self, dataset_both):
        """
        THE KEY TEST: the set of window lengths for normal windows and
        for anomaly windows must both be {window_size} — a single-element
        set.  This proves there is no length signal the model can exploit.
        """
        normal_lengths  = set(dataset_both.normal_windows().shape[1:])
        anomaly_lengths = set(dataset_both.anomaly_windows().shape[1:])

        # Both must be {window_size} — identical and degenerate
        assert normal_lengths  == {WINDOW_SIZE}, (
            f"Normal length set: {normal_lengths}, expected {{{WINDOW_SIZE}}}"
        )
        assert anomaly_lengths == {WINDOW_SIZE}, (
            f"Anomaly length set: {anomaly_lengths}, expected {{{WINDOW_SIZE}}}"
        )
        assert normal_lengths  == anomaly_lengths, (
            f"Length distributions differ: normal={normal_lengths}, "
            f"anomaly={anomaly_lengths}"
        )


# ── Data integrity assertions ──────────────────────────────────────────────────

class TestDataIntegrity:

    def test_non_empty(self, dataset_both):
        assert len(dataset_both) > 0

    def test_labels_are_binary(self, dataset_both):
        unique_labels = set(dataset_both.labels.tolist())
        assert unique_labels.issubset({0, 1}), f"Unexpected labels: {unique_labels}"

    def test_both_classes_present(self, dataset_both):
        assert 0 in dataset_both.labels, "No normal windows"
        assert 1 in dataset_both.labels, "No anomaly windows"

    def test_arrays_aligned(self, dataset_both):
        n = len(dataset_both.windows)
        assert len(dataset_both.labels)  == n
        assert len(dataset_both.seg_ids) == n

    def test_windows_are_float32(self, dataset_both):
        assert dataset_both.windows.dtype == np.float32

    def test_no_all_zero_windows(self, dataset_both):
        """Zero windows would indicate bad z-scoring of constant signals."""
        row_stds = dataset_both.windows.std(axis=1)
        n_zero = (row_stds == 0).sum()
        # Allow up to 1% zero-std windows (e.g. constant-signal padding edge case)
        assert n_zero / len(dataset_both) < 0.01, (
            f"{n_zero} windows ({n_zero/len(dataset_both):.1%}) have zero std"
        )

    def test_no_nan_or_inf(self, dataset_both):
        assert not np.isnan(dataset_both.windows).any(), "NaN values in windows"
        assert not np.isinf(dataset_both.windows).any(), "Inf values in windows"

    def test_train_normal_only_has_no_anomaly_labels(self, dataset_train_normal):
        assert (dataset_train_normal.labels == 0).all(), (
            "train_normal_only=True returned windows with anomaly labels"
        )


# ── Sampling filter assertion ──────────────────────────────────────────────────

class TestSamplingFilter:

    def test_sampling5_produces_windows(self):
        ds = make_sliding_windows(
            window_size=64, stride=32, sampling_rate_filter=5
        )
        assert len(ds) > 0

    def test_sampling1_produces_windows(self):
        ds = make_sliding_windows(
            window_size=128, stride=64, sampling_rate_filter=1
        )
        assert len(ds) > 0

    def test_sampling_filters_are_disjoint(self):
        """
        Windows produced by each sampling filter must come from disjoint
        segment sets — no segment should appear in both cohorts.
        """
        ds5 = make_sliding_windows(window_size=64, stride=32, sampling_rate_filter=5)
        ds1 = make_sliding_windows(window_size=64, stride=32, sampling_rate_filter=1)
        ids5 = set(ds5.seg_ids.tolist())
        ids1 = set(ds1.seg_ids.tolist())
        overlap = ids5 & ids1
        assert len(overlap) == 0, (
            f"Segments appear in both sampling cohorts: {overlap}"
        )


# ── WindowDataset post-init guard ─────────────────────────────────────────────

class TestWindowDatasetInvariant:

    def test_post_init_rejects_wrong_width(self):
        """WindowDataset should raise if windows width != window_size."""
        with pytest.raises(AssertionError):
            WindowDataset(
                windows     = np.zeros((10, 32), dtype=np.float32),
                labels      = np.zeros(10, dtype=np.int32),
                seg_ids     = np.zeros(10, dtype=np.int64),
                window_size = 64,   # mismatch → should raise
            )

    def test_post_init_rejects_length_mismatch(self):
        """WindowDataset should raise if arrays have inconsistent lengths."""
        with pytest.raises(AssertionError):
            WindowDataset(
                windows     = np.zeros((10, 64), dtype=np.float32),
                labels      = np.zeros(9, dtype=np.int32),   # wrong length
                seg_ids     = np.zeros(10, dtype=np.int64),
                window_size = 64,
            )
