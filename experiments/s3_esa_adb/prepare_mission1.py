"""
S3 — ESA-AD Mission1 → dataset.csv adapter  (Quesada 2026)

Converts the ESA-Mission1 raw format (per-channel pickles + labels.csv with
anomaly intervals) into the OPS-SAT-AD-compatible feature format that the
LPINormalizingFlow ensemble pipeline consumes.

──────────────────────────────────────────────────────────────────────────────
Design decisions
──────────────────────────────────────────────────────────────────────────────

DECISION A — Per-channel univariate (not multivariate concatenation)
  Each channel is processed independently, producing one row per (channel,
  window) pair with 16–18 statistical features. This mirrors OPS-SAT-AD
  exactly, enabling direct benchmark comparison.
  Cross-channel correlations (e.g. Pearson matrix flattened) are a natural
  S4 extension but are NOT included here to keep the claim clean.

DECISION B — Fixed WINDOW_SIZE windows (no length leakage)
  Both anomalous and normal windows are exactly WINDOW_SIZE points.
  This prevents the model from learning segment length instead of anomaly
  pattern — the same issue that invalidated OPS-SAT-AD segment-level
  training (Decision 2 in CLAUDE.md).
  WINDOW_SIZE=64 matches the OPS-SAT-AD sampling=5 cohort median length.

DECISION C — Features: all 18 (exclusion deferred to training)
  We compute all 18 OPS-SAT-AD-equivalent features including n_peaks and
  gaps_squared. The training script (run_nf_seed_ensemble.py) applies
  EXCLUDE_FEATURES = {"n_peaks", "gaps_squared"}. Computing them here keeps
  the dataset format identical to OPS-SAT-AD, so the same loader works
  for both datasets without modification.

DECISION D — Temporal 80/20 train/test split within Mission1
  The first TRAIN_FRACTION of each channel's timeline → train=1.
  The remaining → train=0 (held-out test).
  Rationale: cross-mission generalization (Mission2/3 as zero-shot test)
  is a stronger claim reserved for after Mission2/3 are downloaded.
  The temporal split ensures no future data leaks into training while
  leaving a clean held-out test for local validation.

DECISION E — Anomaly segment construction
  For each interval [t_start, t_end] in labels.csv:
    • If interval contains ≥ WINDOW_SIZE points:
        → non-overlapping WINDOW_SIZE windows tiling the interval
    • If interval contains [MIN_ANOMALY_POINTS, WINDOW_SIZE) points:
        → one window centered on the interval, padded with surrounding context
    • If interval contains < MIN_ANOMALY_POINTS points:
        → skip (too short for meaningful features)
  Normal windows: sliding window (stride=STRIDE_NORMAL) over the gaps
  between labeled anomaly intervals. Gaps shorter than WINDOW_SIZE are skipped.

──────────────────────────────────────────────────────────────────────────────
Assumed ESA-Mission1 directory structure
──────────────────────────────────────────────────────────────────────────────
  ESA-Mission1/
  ├── labels.csv          # anomaly intervals
  └── *.pkl               # one pickle per channel (87 channels)

labels.csv expected columns:
  start, end              # timestamps matching pickle time axis
  channel (optional)      # if present: anomaly is channel-specific
                          # if absent: anomaly applies to ALL channels

Pickle formats handled (auto-detected):
  • pd.DataFrame with DatetimeIndex or column named 'time'/'timestamp' + 'value'/'values'
  • dict with keys 'time'/'times'/'t' and 'value'/'values'/'y'
  • np.ndarray of shape (N,) — values only, times reconstructed from sampling rate
  • np.ndarray of shape (N, 2) — columns [time, value]

──────────────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────────────
  cd /workspace/anomaly-internal-detection
  python experiments/s3_esa_adb/prepare_mission1.py \\
      --data_dir /workspace/ESA-Mission1 \\
      --out_dir  reference/data/esa_mission1

Output:
  reference/data/esa_mission1/dataset.csv   ← drop-in for reference/data/dataset.csv
  reference/data/esa_mission1/prep_summary.txt

The output CSV has the same schema as OPS-SAT-AD dataset.csv so that
run_nf_seed_ensemble.py can consume it with --data_path flag (S3 version).
"""
from __future__ import annotations

import argparse
import pickle
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import skew as sp_skew

# ─── Configuration ────────────────────────────────────────────────────────────

WINDOW_SIZE       = 64    # fixed window length (points) — no length leakage
STRIDE_NORMAL     = 32    # stride for normal windows (50% overlap → more diversity)
MIN_ANOMALY_PTS   = 8     # skip intervals shorter than this
TRAIN_FRACTION    = 0.80  # temporal split: first 80% of each channel → train

# OPS-SAT-AD feature columns (excluding segment index and meta columns).
# n_peaks and gaps_squared are computed but flagged for exclusion at train time.
FEATURE_COLS = [
    "duration", "len",
    "mean", "var", "std", "kurtosis", "skew",
    "n_peaks",          # ← excluded at training time (length leakage)
    "smooth10_n_peaks", "smooth20_n_peaks",
    "diff_peaks", "diff2_peaks",
    "diff_var", "diff2_var",
    "gaps_squared",     # ← excluded at training time (length confounding)
    "len_weighted",
    "var_div_duration", "var_div_len",
]

META_COLS = ["anomaly", "train", "channel", "sampling"]

# ─── Labels loader ────────────────────────────────────────────────────────────

def load_labels(labels_path: Path) -> pd.DataFrame:
    """
    Load anomaly intervals from labels.csv.

    Returns DataFrame with columns:
      start_s, end_s    — interval boundaries in seconds (float)
      channel           — channel name string, or None if mission-wide

    Handles multiple timestamp formats:
      • ISO datetime strings  →  converted to seconds since first event
      • Unix float seconds    →  used as-is
      • Integer row indices   →  used as-is (treated as point indices, not seconds)
    """
    df = pd.read_csv(labels_path)
    df.columns = df.columns.str.strip().str.lower()

    # Normalise column names for start/end
    rename = {}
    for c in df.columns:
        if c in {"start", "start_time", "begin", "t_start", "tstart"}:
            rename[c] = "start_raw"
        elif c in {"end", "end_time", "finish", "t_end", "tend", "stop"}:
            rename[c] = "end_raw"
        elif c in {"channel", "chan", "channel_id", "telemetry_channel"}:
            rename[c] = "channel"
    df = df.rename(columns=rename)

    if "start_raw" not in df.columns or "end_raw" not in df.columns:
        raise ValueError(
            f"labels.csv must have start/end columns. Found: {list(df.columns)}"
        )

    # Detect format and convert to float seconds
    sample = df["start_raw"].iloc[0]
    if isinstance(sample, str):
        # ISO datetime or similar
        start_dt = pd.to_datetime(df["start_raw"])
        end_dt   = pd.to_datetime(df["end_raw"])
        t0 = start_dt.min()
        df["start_s"] = (start_dt - t0).dt.total_seconds()
        df["end_s"]   = (end_dt   - t0).dt.total_seconds()
    else:
        df["start_s"] = df["start_raw"].astype(float)
        df["end_s"]   = df["end_raw"].astype(float)

    if "channel" not in df.columns:
        df["channel"] = None   # mission-wide anomaly

    result = df[["start_s", "end_s", "channel"]].copy()
    result = result.sort_values("start_s").reset_index(drop=True)

    print(f"  labels.csv: {len(result)} anomaly intervals loaded")
    if result["channel"].notna().any():
        print(f"    channel-specific: {result['channel'].nunique()} channels affected")
    else:
        print("    mission-wide (anomalies apply to all channels)")

    return result


# ─── Pickle loader ────────────────────────────────────────────────────────────

def load_channel_pickle(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one channel pickle into (times_s, values) arrays.

    times_s : float64 array, seconds since start of channel recording
    values  : float64 array, telemetry values

    Handles: pd.DataFrame, dict, np.ndarray (1D values or 2D [time, value]).
    """
    with open(path, "rb") as f:
        raw: Any = pickle.load(f)

    times_s: np.ndarray
    values:  np.ndarray

    if isinstance(raw, pd.DataFrame):
        # Find time column
        time_col = _find_col(raw, {"time", "timestamp", "t", "index", "times"})
        val_col  = _find_col(raw, {"value", "values", "y", "v", "telemetry", "data"})
        if time_col is None and isinstance(raw.index, pd.DatetimeIndex):
            times_raw = raw.index
            t0 = times_raw.min()
            times_s = (times_raw - t0).total_seconds().values.astype(float)
        elif time_col is None:
            # Assume index is time in some unit — treat as sequential
            times_s = np.arange(len(raw), dtype=float)
        else:
            t_series = raw[time_col]
            times_s = _to_float_seconds(t_series)
        if val_col is None:
            # Use first numeric column that isn't the time column
            numeric = raw.select_dtypes(include=[np.number]).columns.tolist()
            numeric = [c for c in numeric if c != time_col]
            if not numeric:
                raise ValueError(f"No numeric value column found in {path.name}")
            val_col = numeric[0]
        values = raw[val_col].values.astype(float)

    elif isinstance(raw, dict):
        tkey = _find_dict_key(raw, {"time", "times", "t", "timestamp", "timestamps"})
        vkey = _find_dict_key(raw, {"value", "values", "y", "v", "telemetry", "data"})
        if tkey is None:
            times_s = np.arange(len(raw[vkey]), dtype=float)
        else:
            times_s = _to_float_seconds(pd.Series(raw[tkey]))
        values = np.asarray(raw[vkey], dtype=float)

    elif isinstance(raw, np.ndarray):
        if raw.ndim == 2 and raw.shape[1] == 2:
            times_s = raw[:, 0].astype(float)
            # Convert to seconds since start if needed
            if times_s[0] > 1e9:  # looks like Unix timestamp
                times_s = times_s - times_s[0]
            values = raw[:, 1].astype(float)
        elif raw.ndim == 1:
            values  = raw.astype(float)
            times_s = np.arange(len(values), dtype=float)
        else:
            raise ValueError(f"Unexpected ndarray shape {raw.shape} in {path.name}")

    else:
        raise TypeError(
            f"Unsupported pickle content type {type(raw)} in {path.name}.\n"
            "Expected pd.DataFrame, dict, or np.ndarray."
        )

    # Ensure sorted by time
    order   = np.argsort(times_s)
    times_s = times_s[order]
    values  = values[order]

    # Shift to start at 0
    if times_s[0] != 0:
        times_s = times_s - times_s[0]

    # Remove NaN
    mask = np.isfinite(values) & np.isfinite(times_s)
    if mask.sum() < len(mask):
        warnings.warn(f"{path.name}: dropped {(~mask).sum()} NaN/Inf points")
    times_s = times_s[mask]
    values  = values[mask]

    return times_s, values


def _to_float_seconds(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(s) or isinstance(
        s.iloc[0] if len(s) else "", str
    ):
        try:
            dt = pd.to_datetime(s)
            t0 = dt.min()
            return (dt - t0).dt.total_seconds().values.astype(float)
        except Exception:
            pass
    arr = s.values.astype(float)
    if arr[0] > 1e9:          # Unix epoch → relative
        arr = arr - arr[0]
    return arr


def _find_col(df: pd.DataFrame, names: set) -> str | None:
    for c in df.columns:
        if c.lower().strip() in names:
            return c
    return None


def _find_dict_key(d: dict, names: set) -> str | None:
    for k in d.keys():
        if str(k).lower().strip() in names:
            return k
    return None


# ─── Sampling-rate detection ──────────────────────────────────────────────────

def detect_sampling_rate(times_s: np.ndarray) -> float:
    """Return median inter-sample interval in seconds."""
    if len(times_s) < 2:
        return 1.0
    diffs = np.diff(times_s)
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else 1.0


# ─── Segmentation ─────────────────────────────────────────────────────────────

def _get_point_indices(times_s: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
    """Indices of points with t_start <= time <= t_end."""
    return np.where((times_s >= t_start) & (times_s <= t_end))[0]


def extract_anomaly_windows(
    times_s:           np.ndarray,
    values:            np.ndarray,
    anomaly_intervals: list[tuple[float, float]],
    window_size:       int = WINDOW_SIZE,
    min_pts:           int = MIN_ANOMALY_PTS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create fixed-size windows from anomaly intervals.

    Returns list of (window_times_s, window_values) pairs, each exactly
    window_size points.  All windows carry label=1.
    """
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    n = len(times_s)

    for t_start, t_end in anomaly_intervals:
        idx = _get_point_indices(times_s, t_start, t_end)
        n_pts = len(idx)

        if n_pts < min_pts:
            continue   # too short for meaningful features

        if n_pts >= window_size:
            # Tile the interval with non-overlapping windows
            i = 0
            while i + window_size <= n_pts:
                seg_idx = idx[i: i + window_size]
                windows.append((times_s[seg_idx], values[seg_idx]))
                i += window_size
        else:
            # Center the anomaly interval in a window_size context
            center = (idx[0] + idx[-1]) // 2
            half   = window_size // 2
            i_start = max(0, center - half)
            i_end   = i_start + window_size
            if i_end > n:
                i_end   = n
                i_start = max(0, i_end - window_size)
            if i_end - i_start == window_size:
                seg_idx = np.arange(i_start, i_end)
                windows.append((times_s[seg_idx], values[seg_idx]))
            # else: not enough context around the interval — skip

    return windows


def extract_normal_windows(
    times_s:           np.ndarray,
    values:            np.ndarray,
    anomaly_intervals: list[tuple[float, float]],
    window_size:       int = WINDOW_SIZE,
    stride:            int = STRIDE_NORMAL,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Slide a window over the gaps between anomaly intervals.

    A small safety margin (window_size // 2 points) is added around each
    anomaly interval boundary to avoid windows that partially overlap
    anomaly regions at the edge.
    """
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    n = len(times_s)

    # Build boolean mask: True = definitely normal
    is_normal = np.ones(n, dtype=bool)
    dt_median = detect_sampling_rate(times_s)
    margin_s  = (window_size // 2) * dt_median

    for t_start, t_end in anomaly_intervals:
        masked = _get_point_indices(
            times_s, t_start - margin_s, t_end + margin_s
        )
        is_normal[masked] = False

    # Walk normal regions
    i = 0
    while i + window_size <= n:
        if np.all(is_normal[i: i + window_size]):
            windows.append((times_s[i: i + window_size], values[i: i + window_size]))
            i += stride
        else:
            # Skip past the contaminated region
            bad_region_end = i + window_size
            while bad_region_end < n and not is_normal[bad_region_end]:
                bad_region_end += 1
            i = bad_region_end

    return windows


# ─── Feature extraction ───────────────────────────────────────────────────────

def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Uniform moving-average with window k."""
    if len(x) < k:
        return x.copy()
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def compute_features(
    window_times_s: np.ndarray,
    window_values:  np.ndarray,
) -> dict[str, float]:
    """
    Compute the 18 OPS-SAT-AD-equivalent statistical features for one window.

    Feature definitions match the KP Labs dataset.csv schema exactly so that
    the output CSV is a drop-in replacement for reference/data/dataset.csv.
    """
    v   = window_values
    t   = window_times_s
    n   = len(v)

    # Basic statistics
    mean_v = float(np.mean(v))
    var_v  = float(np.var(v))
    std_v  = float(np.std(v))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kurt = float(sp_kurtosis(v, bias=True))
        skew = float(sp_skew(v, bias=True))

    # Duration and length
    duration = float(t[-1] - t[0]) if n > 1 else 0.0
    len_v    = n

    # Peak counts (find_peaks on value, smoothed, and derivatives)
    n_peaks, _          = find_peaks(v)
    n_peaks_count       = len(n_peaks)

    sm10                = _smooth(v, 10)
    sm20                = _smooth(v, 20)
    n_peaks_sm10, _     = find_peaks(sm10)
    n_peaks_sm20, _     = find_peaks(sm20)

    d1 = np.diff(v)
    d2 = np.diff(d1)
    diff_peaks_idx, _   = find_peaks(d1)
    diff2_peaks_idx, _  = find_peaks(d2)

    # Variance of derivatives
    diff_var  = float(np.var(d1)) if len(d1) > 0 else 0.0
    diff2_var = float(np.var(d2)) if len(d2) > 0 else 0.0

    # Gaps-based features (for irregular or regular sampling)
    dt_arr       = np.diff(t)                              # time gaps
    gaps_sq      = float(np.sum(dt_arr ** 2))              # sum of squared gaps
    dt_median    = float(np.median(dt_arr)) if len(dt_arr) else 1.0
    len_weighted = float(len_v * dt_median)                # effective duration per pt

    var_div_duration = var_v / duration      if duration > 0 else 0.0
    var_div_len      = var_v / len_v         if len_v    > 0 else 0.0

    return {
        "duration":          duration,
        "len":               float(len_v),
        "mean":              mean_v,
        "var":               var_v,
        "std":               std_v,
        "kurtosis":          kurt,
        "skew":              skew,
        "n_peaks":           float(n_peaks_count),
        "smooth10_n_peaks":  float(len(n_peaks_sm10)),
        "smooth20_n_peaks":  float(len(n_peaks_sm20)),
        "diff_peaks":        float(len(diff_peaks_idx)),
        "diff2_peaks":       float(len(diff2_peaks_idx)),
        "diff_var":          diff_var,
        "diff2_var":         diff2_var,
        "gaps_squared":      gaps_sq,
        "len_weighted":      len_weighted,
        "var_div_duration":  var_div_duration,
        "var_div_len":       var_div_len,
    }


# ─── Train / test split ───────────────────────────────────────────────────────

def assign_train_test(
    rows: list[dict],
    train_fraction: float = TRAIN_FRACTION,
) -> list[dict]:
    """
    Temporal split: for each channel, the first TRAIN_FRACTION of its
    time span → train=1, remainder → train=0.

    'First' is defined by the window's midpoint timestamp within the channel.
    """
    by_channel: dict[str, list] = {}
    for r in rows:
        by_channel.setdefault(r["channel"], []).append(r)

    result: list[dict] = []
    for ch, ch_rows in by_channel.items():
        ch_rows.sort(key=lambda r: r["_t_mid"])
        cutoff_idx = max(1, int(len(ch_rows) * train_fraction))
        for i, r in enumerate(ch_rows):
            r["train"] = 1 if i < cutoff_idx else 0
            result.append(r)

    return result


# ─── Main pipeline ────────────────────────────────────────────────────────────

def build_dataset(
    data_dir:       Path,
    out_dir:        Path,
    window_size:    int   = WINDOW_SIZE,
    stride_normal:  int   = STRIDE_NORMAL,
    min_anom_pts:   int   = MIN_ANOMALY_PTS,
    train_fraction: float = TRAIN_FRACTION,
) -> pd.DataFrame:
    """
    Full pipeline: discover pickles, load labels, segment, featurise, split.
    Returns the final DataFrame (also written to out_dir/dataset.csv).
    """
    # ── Locate files ──────────────────────────────────────────────────────────
    labels_candidates = list(data_dir.glob("labels*.csv")) + list(
        data_dir.glob("Labels*.csv")
    )
    if not labels_candidates:
        raise FileNotFoundError(
            f"No labels*.csv found in {data_dir}. Contents:\n"
            + "\n".join(str(p.name) for p in sorted(data_dir.iterdir()))
        )
    labels_path = labels_candidates[0]
    print(f"\n  Labels file : {labels_path.name}")

    pickle_paths = sorted(data_dir.glob("*.pkl"))
    if not pickle_paths:
        pickle_paths = sorted(data_dir.rglob("*.pkl"))
    if not pickle_paths:
        raise FileNotFoundError(f"No .pkl files found under {data_dir}")
    print(f"  Channels    : {len(pickle_paths)} pickle files")

    # ── Load labels ───────────────────────────────────────────────────────────
    labels_df = load_labels(labels_path)

    # ── Process each channel ──────────────────────────────────────────────────
    rows: list[dict] = []
    seg_id = 0

    for pkl_path in sorted(pickle_paths):
        chan_name = pkl_path.stem

        try:
            times_s, values = load_channel_pickle(pkl_path)
        except Exception as e:
            warnings.warn(f"Skipping {chan_name}: {e}")
            continue

        if len(times_s) < window_size:
            warnings.warn(f"Skipping {chan_name}: only {len(times_s)} points")
            continue

        dt_s = detect_sampling_rate(times_s)

        # Filter anomaly intervals for this channel
        mask_all   = labels_df["channel"].isna()
        mask_chan  = labels_df["channel"] == chan_name
        relevant   = labels_df[mask_all | mask_chan]

        # Shift labels to match this channel's time axis (times_s start=0)
        # Labels may reference absolute timestamps; they may already be relative.
        # Strategy: if labels fall outside [times_s.min(), times_s.max()], warn.
        t_min, t_max = times_s[0], times_s[-1]
        intervals: list[tuple[float, float]] = []
        for _, row in relevant.iterrows():
            ts, te = float(row.start_s), float(row.end_s)
            if te < t_min or ts > t_max:
                continue   # this interval is outside this channel's recording
            intervals.append((
                max(ts, t_min),
                min(te, t_max),
            ))

        # Extract windows
        anom_wins = extract_anomaly_windows(
            times_s, values, intervals, window_size, min_anom_pts
        )
        norm_wins = extract_normal_windows(
            times_s, values, intervals, window_size, stride_normal
        )

        # Featurise
        for label, wins in [(1, anom_wins), (0, norm_wins)]:
            for wt, wv in wins:
                feats = compute_features(wt, wv)
                seg_id += 1
                rows.append({
                    "segment":  seg_id,
                    "anomaly":  label,
                    "channel":  chan_name,
                    "sampling": round(dt_s),          # nearest integer seconds
                    "_t_mid":   float(wt[len(wt) // 2]),  # for temporal split
                    **feats,
                })

        print(
            f"  [{chan_name:20s}]  {len(times_s):7,} pts | "
            f"anom_wins={len(anom_wins):4d}  norm_wins={len(norm_wins):5d}"
        )

    if not rows:
        raise RuntimeError("No segments were created. Check data directory and labels.")

    # ── Temporal train/test split ─────────────────────────────────────────────
    rows = assign_train_test(rows, train_fraction)

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df = df.drop(columns=["_t_mid"])

    # Reorder columns to match OPS-SAT-AD schema exactly
    col_order = ["segment"] + META_COLS + FEATURE_COLS
    df = df[col_order]
    df = df.set_index("segment")

    return df


def print_summary(df: pd.DataFrame, out_path: Path) -> str:
    train = df[df["train"] == 1]
    test  = df[df["train"] == 0]

    anom_train = (train["anomaly"] == 1).sum()
    norm_train = (train["anomaly"] == 0).sum()
    anom_test  = (test["anomaly"]  == 1).sum()
    norm_test  = (test["anomaly"]  == 0).sum()

    lines = [
        "",
        "=" * 60,
        "ESA-AD Mission1 — dataset.csv summary",
        "=" * 60,
        f"  Total segments : {len(df):,}",
        f"  Channels       : {df['channel'].nunique()}",
        f"  Sampling rates : {sorted(df['sampling'].unique())} (seconds)",
        "",
        f"  TRAIN ({len(train):,} segs)",
        f"    anomalous : {anom_train:,}  ({anom_train/len(train):.1%})",
        f"    normal    : {norm_train:,}  ({norm_train/len(train):.1%})",
        "",
        f"  TEST  ({len(test):,} segs)",
        f"    anomalous : {anom_test:,}  ({anom_test/len(test):.1%})",
        f"    normal    : {norm_test:,}  ({norm_test/len(test):.1%})",
        "",
        f"  Output : {out_path}",
        "=" * 60,
        "",
        "Next step:",
        "  python experiments/s3_esa_adb/run_nf_ensemble_s3.py \\",
        "      --data_path reference/data/esa_mission1/dataset.csv",
        "=" * 60,
    ]
    summary = "\n".join(lines)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ESA-Mission1 → dataset.csv for LPI pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Example:
              python experiments/s3_esa_adb/prepare_mission1.py \\
                  --data_dir /workspace/ESA-Mission1 \\
                  --out_dir  reference/data/esa_mission1
        """),
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to ESA-Mission1 directory (with labels.csv and *.pkl files)",
    )
    parser.add_argument(
        "--out_dir",  required=True,
        help="Output directory for dataset.csv",
    )
    parser.add_argument(
        "--window_size",   type=int,   default=WINDOW_SIZE,
        help=f"Fixed window length in points (default: {WINDOW_SIZE})",
    )
    parser.add_argument(
        "--stride_normal", type=int,   default=STRIDE_NORMAL,
        help=f"Stride for normal windows (default: {STRIDE_NORMAL})",
    )
    parser.add_argument(
        "--train_fraction", type=float, default=TRAIN_FRACTION,
        help=f"Temporal train fraction (default: {TRAIN_FRACTION})",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nESA-AD Mission1 adapter")
    print(f"  data_dir     : {data_dir}")
    print(f"  out_dir      : {out_dir}")
    print(f"  window_size  : {args.window_size}")
    print(f"  stride_normal: {args.stride_normal}")
    print(f"  train_frac   : {args.train_fraction}")

    df = build_dataset(
        data_dir       = data_dir,
        out_dir        = out_dir,
        window_size    = args.window_size,
        stride_normal  = args.stride_normal,
        train_fraction = args.train_fraction,
    )

    out_csv = out_dir / "dataset.csv"
    df.to_csv(out_csv)
    print(f"\n  Wrote {len(df):,} rows → {out_csv}")

    summary = print_summary(df, out_csv)
    (out_dir / "prep_summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
