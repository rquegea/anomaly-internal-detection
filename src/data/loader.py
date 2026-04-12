"""
Data loading utilities for OPS-SAT-AD and ESA-AD datasets.

Two loading modes
-----------------
1. Feature-level (legacy, used by S1 baselines):
   load_opssat_features()  →  pre-computed statistical features from dataset.csv

2. Window-level (correct, used by S2+ Transformer):
   make_sliding_windows()  →  fixed-size windows from raw segments.csv

The window-level mode eliminates two data-integrity issues found during EDA:

  * Length leakage: anomalous segments are ~3.4× longer than normal ones.
    If the model sees whole segments, it can learn to flag long sequences
    regardless of their actual pattern. Sliding windows of fixed size
    remove this confound: every sample seen by the model has exactly
    `window_size` points, whether it comes from a normal or anomalous segment.

  * Sampling-rate mixing: segments are recorded at two rates (1 Hz and
    5-second intervals). Mixing them would present the model with sequences
    that represent very different time spans despite having the same number
    of points. The `sampling_rate_filter` parameter enforces homogeneity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_DIR           = Path(__file__).parents[2] / "data"
REFERENCE_DATA_DIR = Path(__file__).parents[2] / "reference" / "data"


# ── Feature-level loading (S1 baselines) ─────────────────────────────────────

def load_opssat_features(
    path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load the pre-computed feature dataset from OPS-SAT-AD (dataset.csv).
    Returns (X_train, y_train, X_test, y_test).
    """
    csv = path or REFERENCE_DATA_DIR / "dataset.csv"
    df  = pd.read_csv(csv, index_col="segment")

    meta_cols = {"anomaly", "train", "channel", "sampling"}
    features  = [c for c in df.columns if c not in meta_cols]

    X_train = df.loc[df.train == 1, features]
    y_train = df.loc[df.train == 1, "anomaly"]
    X_test  = df.loc[df.train == 0, features]
    y_test  = df.loc[df.train == 0, "anomaly"]

    return X_train, y_train, X_test, y_test


def load_opssat_segments(path: Path | None = None) -> pd.DataFrame:
    """Load raw time-series segments from OPS-SAT-AD (segments.csv)."""
    csv = path or REFERENCE_DATA_DIR / "segments.csv"
    return pd.read_csv(csv, parse_dates=["timestamp"])


def scale_for_unsupervised(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on nominal train samples only.
    Returns (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    X_nominal = X_train.loc[y_train == 0]
    scaler    = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_nominal)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ── Window-level loading (S2+ Transformer) ───────────────────────────────────

@dataclass
class WindowDataset:
    """
    Container for sliding-window outputs.

    Attributes
    ----------
    windows : float32 array of shape (n_windows, window_size)
        Each row is one z-scored fixed-length window.
    labels  : int array of shape (n_windows,)
        Segment-level anomaly label (0 = normal, 1 = anomalous).
        All windows from a segment inherit that segment's label.
    seg_ids : int array of shape (n_windows,)
        Segment identifier. Used at evaluation time to aggregate
        window-level scores back to segment-level predictions.
    window_size : int
        Guaranteed length of every window in `windows`.
    """
    windows     : np.ndarray
    labels      : np.ndarray
    seg_ids     : np.ndarray
    window_size : int

    def __post_init__(self):
        assert self.windows.shape[1] == self.window_size, (
            f"Window shape mismatch: expected width {self.window_size}, "
            f"got {self.windows.shape[1]}"
        )
        assert len(self.labels)  == len(self.windows)
        assert len(self.seg_ids) == len(self.windows)

    def __len__(self):
        return len(self.windows)

    def normal_windows(self) -> np.ndarray:
        return self.windows[self.labels == 0]

    def anomaly_windows(self) -> np.ndarray:
        return self.windows[self.labels == 1]


def make_sliding_windows(
    path: Path | None = None,
    window_size: int = 64,
    stride: int | None = None,
    sampling_rate_filter: int | None = 5,
    split: str = "both",           # "train" | "test" | "both"
    train_normal_only: bool = False,
    min_segment_length: int | None = None,
) -> WindowDataset:
    """
    Generate fixed-size sliding windows from raw OPS-SAT-AD segments.

    Parameters
    ----------
    path : Path, optional
        Path to segments.csv. Defaults to reference/data/segments.csv.
    window_size : int
        Number of time steps per window. Every window has exactly this length.
    stride : int, optional
        Step size between consecutive windows.
        Default: window_size // 2  (50 % overlap).
    sampling_rate_filter : int, optional
        If set, only include segments whose `sampling` column equals this value.
        Use 5 for the 5-second-interval cohort, 1 for the 1-Hz cohort.
        Set to None to include all (not recommended — see module docstring).
    split : {"train", "test", "both"}
        Which dataset split to include.
    train_normal_only : bool
        If True, only return windows from train-split normal segments.
        Useful for building the unsupervised training set.
    min_segment_length : int, optional
        Discard segments shorter than this many points before windowing.
        Default: window_size (segments shorter than one window are skipped).

    Returns
    -------
    WindowDataset
        All windows are guaranteed to have shape (window_size,).
        Labels and seg_ids are aligned with `windows` row-by-row.

    Design note — why sliding windows?
    -----------------------------------
    Using whole segments as model inputs leaks length information:
    anomalous segments are on average 3.4× longer than normal ones
    (EDA finding, notebooks/01_explore_segments.ipynb). A model that
    sees variable-length inputs can trivially learn to flag long sequences.
    Sliding windows of fixed size remove this confound entirely.
    """
    if stride is None:
        stride = window_size // 2
    if min_segment_length is None:
        min_segment_length = window_size

    seg_df = load_opssat_segments(path)

    # ── Filters ───────────────────────────────────────────────────────────────
    if sampling_rate_filter is not None:
        seg_df = seg_df[seg_df["sampling"] == sampling_rate_filter]

    if split == "train":
        seg_df = seg_df[seg_df["train"] == 1]
    elif split == "test":
        seg_df = seg_df[seg_df["train"] == 0]

    if train_normal_only:
        seg_df = seg_df[(seg_df["train"] == 1) & (seg_df["anomaly"] == 0)]

    # ── Build windows ─────────────────────────────────────────────────────────
    all_windows : list[np.ndarray] = []
    all_labels  : list[int]        = []
    all_seg_ids : list[int]        = []

    for seg_id, group in seg_df.groupby("segment", sort=False):
        values  = group.sort_values("timestamp")["value"].to_numpy(dtype=np.float32)
        anomaly = int(group["anomaly"].iloc[0])

        if len(values) < min_segment_length:
            continue

        # Z-score normalise per segment before windowing.
        # Fitting on the full segment (not per-window) preserves the
        # relative scale across the sequence.
        std = values.std()
        values = (values - values.mean()) / (std if std > 1e-8 else 1.0)

        # Slide
        n = len(values)
        start = 0
        while start + window_size <= n:
            window = values[start : start + window_size]
            all_windows.append(window)
            all_labels.append(anomaly)
            all_seg_ids.append(int(seg_id))
            start += stride

    windows = np.stack(all_windows, axis=0)   # (n_windows, window_size)
    labels  = np.array(all_labels,  dtype=np.int32)
    seg_ids = np.array(all_seg_ids, dtype=np.int64)

    return WindowDataset(
        windows     = windows,
        labels      = labels,
        seg_ids     = seg_ids,
        window_size = window_size,
    )


def segment_scores_from_windows(
    window_scores: np.ndarray,
    seg_ids: np.ndarray,
    agg: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate per-window anomaly scores to segment-level scores.

    Parameters
    ----------
    window_scores : (n_windows,) anomaly score per window
    seg_ids       : (n_windows,) segment id per window
    agg           : aggregation function — "max" or "mean"
                    "max" is recommended: one anomalous window is enough
                    to flag the segment.

    Returns
    -------
    unique_seg_ids   : (n_segments,) ordered segment ids
    segment_scores   : (n_segments,) aggregated scores
    """
    agg_fn = np.max if agg == "max" else np.mean
    unique_ids = np.unique(seg_ids)
    scores     = np.array([agg_fn(window_scores[seg_ids == sid]) for sid in unique_ids])
    return unique_ids, scores
