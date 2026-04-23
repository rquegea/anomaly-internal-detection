"""
S3 — ESA-AD Mission1 → dataset.csv adapter  (Quesada 2026)

Converts the ESA-Mission1 raw format (per-channel deflate64-zipped pickles +
labels.csv with anomaly intervals) into the OPS-SAT-AD-compatible feature
format that the LPINormalizingFlow ensemble pipeline consumes.

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
  At the nominal ~90 s/sample of ESA-Mission1, 64 points ≈ 96 min of
  telemetry per window.

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

DECISION F — Sequential channel processing (not parallel)
  Each channel pickle is ~10–15 M rows. Loading all channels simultaneously
  would saturate RAM. Channels are processed one at a time; memory is freed
  between channels via explicit del + gc.collect().

──────────────────────────────────────────────────────────────────────────────
ESA-Mission1 directory structure
──────────────────────────────────────────────────────────────────────────────
  ESA-Mission1/
  ├── labels.csv              # anomaly intervals (ISO timestamps, UTC)
  ├── anomaly_types.csv       # anomaly type metadata (200 types)
  └── channels/
      ├── channel_1.zip       # deflate64-compressed pickle, DatetimeIndex
      ├── channel_2.zip
      └── ...                 # 76 channels total

labels.csv columns:
  ID, Channel, StartTime, EndTime
  StartTime/EndTime are ISO-8601 with UTC suffix (e.g. 2004-12-01T20:42:15.429Z)
  Channel values are "channel_N" matching the zip file stems.

──────────────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────────────
  # Full run (all 76 channels):
  python experiments/s3_esa_adb/prepare_mission1.py \\
      --data_dir /workspace/ESA-Mission1/ESA-Mission1 \\
      --out_dir  reference/data/esa_mission1

  # Single-channel test:
  python experiments/s3_esa_adb/prepare_mission1.py \\
      --data_dir /workspace/ESA-Mission1/ESA-Mission1 \\
      --out_dir  reference/data/esa_mission1_ch1 \\
      --channel  channel_1

Output:
  <out_dir>/dataset.csv   ← drop-in for reference/data/dataset.csv
  <out_dir>/prep_summary.txt

The output CSV has the same schema as OPS-SAT-AD dataset.csv so that
run_nf_seed_ensemble.py can consume it with --data_path flag (S3 version).
"""
from __future__ import annotations

import argparse
import gc
import io
import pickle
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zipfile_deflate64 as zf
from scipy.signal import find_peaks
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import skew as sp_skew

# ─── Configuration ────────────────────────────────────────────────────────────

# WINDOW_SIZE=64 at ~90 s/sample (ESA-Mission1 nominal) ≈ 96 min of telemetry.
# Actual per-channel sampling rate is computed dynamically via detect_sampling_rate().
WINDOW_SIZE       = 64
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
      start_s, end_s    — interval boundaries in seconds relative to mission epoch
      start_dt, end_dt  — absolute UTC pd.Timestamps (NaT if source was not datetime)
      channel           — "channel_N" string, or None if mission-wide

    Handles multiple timestamp formats:
      • ISO-8601 strings with UTC (e.g. 2004-12-01T20:42:15.429Z) → absolute
      • Unix float seconds → relative
      • Integer row indices → relative (treated as point indices)

    Handles column name variants including CamelCase (StartTime/EndTime)
    and snake_case (start_time/end_time).
    """
    df = pd.read_csv(labels_path)
    df.columns = df.columns.str.strip().str.lower()

    # Normalise column names for start/end/channel
    rename = {}
    for c in df.columns:
        if c in {"start", "start_time", "starttime", "begin", "t_start", "tstart"}:
            rename[c] = "start_raw"
        elif c in {"end", "end_time", "endtime", "finish", "t_end", "tend", "stop"}:
            rename[c] = "end_raw"
        elif c in {"channel", "chan", "channel_id", "telemetry_channel"}:
            rename[c] = "channel"
    df = df.rename(columns=rename)

    if "start_raw" not in df.columns or "end_raw" not in df.columns:
        raise ValueError(
            f"labels.csv must have start/end columns. Found: {list(df.columns)}"
        )

    # Detect format and convert
    sample = df["start_raw"].iloc[0]
    if isinstance(sample, str):
        # ISO datetime strings — store absolute timestamps for DatetimeIndex alignment
        start_dt = pd.to_datetime(df["start_raw"], utc=True)
        end_dt   = pd.to_datetime(df["end_raw"],   utc=True)
        t0 = start_dt.min()
        df["start_s"]  = (start_dt - t0).dt.total_seconds()
        df["end_s"]    = (end_dt   - t0).dt.total_seconds()
        df["start_dt"] = start_dt   # absolute UTC — used to align with DatetimeIndex channels
        df["end_dt"]   = end_dt
    else:
        df["start_s"]  = df["start_raw"].astype(float)
        df["end_s"]    = df["end_raw"].astype(float)
        df["start_dt"] = pd.NaT
        df["end_dt"]   = pd.NaT

    if "channel" not in df.columns:
        df["channel"] = None   # mission-wide anomaly

    result = df[["start_s", "end_s", "start_dt", "end_dt", "channel"]].copy()
    result = result.sort_values("start_s").reset_index(drop=True)

    print(f"  labels.csv: {len(result)} anomaly intervals loaded")
    if result["channel"].notna().any():
        print(f"    channel-specific: {result['channel'].nunique()} channels affected")
    else:
        print("    mission-wide (anomalies apply to all channels)")

    return result


# ─── Channel loaders ──────────────────────────────────────────────────────────

def load_channel_zip(path: Path) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """
    Read one channel_N.zip (deflate64) → (times_s, values, t0).

    The zip contains a single pickle: a pd.DataFrame with DatetimeIndex
    and one column 'channel_N'. t0 is the channel's first timestamp and
    is used to convert absolute label timestamps to relative seconds.

    Returns:
      times_s : float64 array, seconds since t0 (channel start)
      values  : float64 array, telemetry values
      t0      : pd.Timestamp (UTC-naive, as stored in the pickle DatetimeIndex)
    """
    with zf.ZipFile(path) as z:
        data = z.read(z.namelist()[0])
    df = pd.read_pickle(io.BytesIO(data))

    t0 = df.index[0]
    times_s = (df.index - t0).total_seconds().values.astype(float)
    values  = df.iloc[:, 0].values.astype(float)

    order   = np.argsort(times_s)
    times_s = times_s[order]
    values  = values[order]

    mask = np.isfinite(values) & np.isfinite(times_s)
    if mask.sum() < len(mask):
        warnings.warn(f"{path.name}: dropped {(~mask).sum()} NaN/Inf points")
    return times_s[mask], values[mask], t0


def load_channel_pickle(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one channel pickle (.pkl, not zipped) into (times_s, values) arrays.

    times_s : float64 array, seconds since start of channel recording
    values  : float64 array, telemetry values

    Handles: pd.DataFrame, dict, np.ndarray (1D values or 2D [time, value]).
    Used as fallback for non-zip formats; ESA-Mission1 uses load_channel_zip().
    """
    with open(path, "rb") as f:
        raw: Any = pickle.load(f)

    times_s: np.ndarray
    values:  np.ndarray

    if isinstance(raw, pd.DataFrame):
        time_col = _find_col(raw, {"time", "timestamp", "t", "index", "times"})
        val_col  = _find_col(raw, {"value", "values", "y", "v", "telemetry", "data"})
        if time_col is None and isinstance(raw.index, pd.DatetimeIndex):
            times_raw = raw.index
            t0 = times_raw.min()
            times_s = (times_raw - t0).total_seconds().values.astype(float)
        elif time_col is None:
            times_s = np.arange(len(raw), dtype=float)
        else:
            t_series = raw[time_col]
            times_s = _to_float_seconds(t_series)
        if val_col is None:
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
            if times_s[0] > 1e9:
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

    order   = np.argsort(times_s)
    times_s = times_s[order]
    values  = values[order]

    if times_s[0] != 0:
        times_s = times_s - times_s[0]

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
    if arr[0] > 1e9:
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
            continue

        if n_pts >= window_size:
            i = 0
            while i + window_size <= n_pts:
                seg_idx = idx[i: i + window_size]
                windows.append((times_s[seg_idx], values[seg_idx]))
                i += window_size
        else:
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

    is_normal = np.ones(n, dtype=bool)
    dt_median = detect_sampling_rate(times_s)
    margin_s  = (window_size // 2) * dt_median

    for t_start, t_end in anomaly_intervals:
        masked = _get_point_indices(
            times_s, t_start - margin_s, t_end + margin_s
        )
        is_normal[masked] = False

    i = 0
    while i + window_size <= n:
        if np.all(is_normal[i: i + window_size]):
            windows.append((times_s[i: i + window_size], values[i: i + window_size]))
            i += stride
        else:
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

    mean_v = float(np.mean(v))
    var_v  = float(np.var(v))
    std_v  = float(np.std(v))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kurt = float(sp_kurtosis(v, bias=True))
        skew = float(sp_skew(v, bias=True))

    duration = float(t[-1] - t[0]) if n > 1 else 0.0
    len_v    = n

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

    diff_var  = float(np.var(d1)) if len(d1) > 0 else 0.0
    diff2_var = float(np.var(d2)) if len(d2) > 0 else 0.0

    dt_arr       = np.diff(t)
    gaps_sq      = float(np.sum(dt_arr ** 2))
    dt_median    = float(np.median(dt_arr)) if len(dt_arr) else 1.0
    len_weighted = float(len_v * dt_median)

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
    only_channel:   str | None = None,
) -> pd.DataFrame:
    """
    Full pipeline: discover channel zips, load labels, segment, featurise, split.

    only_channel: if given (e.g. 'channel_1'), process only that channel.
                  Useful for quick sanity checks before running all 76 channels.

    Returns the final DataFrame (also written to out_dir/dataset.csv).
    Channels are processed sequentially to avoid RAM saturation with 10M+ pt
    channels (Decision F).
    """
    # ── Locate labels ─────────────────────────────────────────────────────────
    labels_candidates = (
        list(data_dir.glob("labels*.csv")) +
        list(data_dir.glob("Labels*.csv"))
    )
    if not labels_candidates:
        raise FileNotFoundError(
            f"No labels*.csv found in {data_dir}. Contents:\n"
            + "\n".join(str(p.name) for p in sorted(data_dir.iterdir()))
        )
    labels_path = labels_candidates[0]
    print(f"\n  Labels file : {labels_path.name}")

    # ── Locate channel zips (ESA-Mission1 format: channels/channel_N.zip) ─────
    channels_dir = data_dir / "channels"
    if channels_dir.is_dir():
        zip_paths = sorted(channels_dir.glob("channel_*.zip"))
    else:
        zip_paths = sorted(data_dir.glob("channel_*.zip"))

    if zip_paths:
        print(f"  Channels    : {len(zip_paths)} channel zips (deflate64)")
        use_zip = True
    else:
        # Fallback: plain .pkl files (other datasets / older format)
        pkl_paths = sorted(data_dir.glob("*.pkl"))
        if not pkl_paths:
            pkl_paths = sorted(data_dir.rglob("*.pkl"))
        if not pkl_paths:
            raise FileNotFoundError(
                f"No channel_*.zip or .pkl files found under {data_dir}"
            )
        zip_paths = pkl_paths  # reuse variable name for unified loop below
        print(f"  Channels    : {len(zip_paths)} pickle files")
        use_zip = False

    # ── Apply single-channel filter ───────────────────────────────────────────
    if only_channel:
        zip_paths = [p for p in zip_paths if p.stem == only_channel]
        if not zip_paths:
            raise ValueError(
                f"--channel '{only_channel}' not found. "
                f"Available: {[p.stem for p in (channels_dir or data_dir).glob('channel_*.zip')]}"
            )
        print(f"  Filter      : only '{only_channel}'")

    # ── Load labels ───────────────────────────────────────────────────────────
    labels_df = load_labels(labels_path)
    has_abs_timestamps = not labels_df["start_dt"].isna().all()

    # ── Process each channel sequentially ────────────────────────────────────
    rows: list[dict] = []
    seg_id = 0

    for ch_path in zip_paths:
        chan_name = ch_path.stem

        try:
            if use_zip:
                times_s, values, t0 = load_channel_zip(ch_path)
            else:
                times_s, values = load_channel_pickle(ch_path)
                t0 = None
        except Exception as e:
            warnings.warn(f"Skipping {chan_name}: {e}")
            continue

        if len(times_s) < window_size:
            warnings.warn(f"Skipping {chan_name}: only {len(times_s)} points")
            continue

        dt_s = detect_sampling_rate(times_s)

        # Filter anomaly intervals for this channel
        mask_all  = labels_df["channel"].isna()
        mask_chan = labels_df["channel"] == chan_name
        relevant  = labels_df[mask_all | mask_chan]

        t_min, t_max = times_s[0], times_s[-1]
        intervals: list[tuple[float, float]] = []

        for _, row in relevant.iterrows():
            if has_abs_timestamps and t0 is not None and pd.notna(row["start_dt"]):
                # Convert absolute label timestamps to seconds relative to channel t0.
                # t0 from the pickle is tz-naive; labels are UTC — treat as same zone.
                t0_utc = t0.tz_localize("UTC") if t0.tzinfo is None else t0
                ts = float((row["start_dt"] - t0_utc).total_seconds())
                te = float((row["end_dt"]   - t0_utc).total_seconds())
            else:
                ts, te = float(row["start_s"]), float(row["end_s"])

            if te < t_min or ts > t_max:
                continue
            intervals.append((max(ts, t_min), min(te, t_max)))

        # Extract and featurise windows
        anom_wins = extract_anomaly_windows(
            times_s, values, intervals, window_size, min_anom_pts
        )
        norm_wins = extract_normal_windows(
            times_s, values, intervals, window_size, stride_normal
        )

        for label, wins in [(1, anom_wins), (0, norm_wins)]:
            for wt, wv in wins:
                feats = compute_features(wt, wv)
                seg_id += 1
                rows.append({
                    "segment":  seg_id,
                    "anomaly":  label,
                    "channel":  chan_name,
                    "sampling": round(dt_s),
                    "_t_mid":   float(wt[len(wt) // 2]),
                    **feats,
                })

        print(
            f"  [{chan_name:20s}]  {len(times_s):10,} pts | "
            f"dt={dt_s:.0f}s | "
            f"anom_wins={len(anom_wins):4d}  norm_wins={len(norm_wins):5d}"
        )

        # Free channel data before loading the next one (Decision F)
        del times_s, values
        gc.collect()

    if not rows:
        raise RuntimeError("No segments were created. Check data directory and labels.")

    # ── Temporal train/test split ─────────────────────────────────────────────
    rows = assign_train_test(rows, train_fraction)

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df = df.drop(columns=["_t_mid"])

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


# ─── Channel scan (--scan mode) ───────────────────────────────────────────────

def scan_channels(
    data_dir:      Path,
    only_channel:  str | None = None,
    window_size:   int = WINDOW_SIZE,
    stride_normal: int = STRIDE_NORMAL,
    min_anom_pts:  int = MIN_ANOMALY_PTS,
) -> None:
    """
    Print per-channel anomaly statistics without writing any output files.

    For each channel reports:
      anom_wins  — total anomalous windows extracted
      H1 / H2    — anom_wins in first vs second temporal half of the channel
      norm_wins  — total normal windows extracted
      ratio      — anom_wins / (anom_wins + norm_wins)
    """
    channels_dir = data_dir / "channels"
    zip_paths = sorted(
        channels_dir.glob("channel_*.zip") if channels_dir.is_dir()
        else data_dir.glob("channel_*.zip")
    )
    if not zip_paths:
        raise FileNotFoundError(f"No channel_*.zip found under {data_dir}")
    if only_channel:
        zip_paths = [p for p in zip_paths if p.stem == only_channel]

    labels_candidates = (
        list(data_dir.glob("labels*.csv")) + list(data_dir.glob("Labels*.csv"))
    )
    if not labels_candidates:
        raise FileNotFoundError(f"No labels*.csv found in {data_dir}")
    labels_df = load_labels(labels_candidates[0])
    has_abs = not labels_df["start_dt"].isna().all()

    W = 22
    print(f"\n{'Channel':<{W}} {'anom_wins':>9}  {'H1':>5}  {'H2':>5}  {'norm_wins':>9}  {'ratio':>6}")
    print("─" * 64)

    total_anom = total_norm = channels_with_anom = 0

    for ch_path in zip_paths:
        chan_name = ch_path.stem
        try:
            times_s, values, t0 = load_channel_zip(ch_path)
        except Exception as e:
            print(f"  {chan_name:<{W - 2}}  ERROR: {e}")
            continue

        if len(times_s) < window_size:
            print(f"  {chan_name:<{W - 2}}  SKIP ({len(times_s)} pts < window_size)")
            del times_s, values
            gc.collect()
            continue

        mask_all  = labels_df["channel"].isna()
        mask_chan = labels_df["channel"] == chan_name
        relevant  = labels_df[mask_all | mask_chan]

        t_min, t_max = times_s[0], times_s[-1]
        intervals: list[tuple[float, float]] = []
        for _, row in relevant.iterrows():
            if has_abs and pd.notna(row["start_dt"]):
                t0_utc = t0.tz_localize("UTC") if t0.tzinfo is None else t0
                ts = float((row["start_dt"] - t0_utc).total_seconds())
                te = float((row["end_dt"]   - t0_utc).total_seconds())
            else:
                ts, te = float(row["start_s"]), float(row["end_s"])
            if te < t_min or ts > t_max:
                continue
            intervals.append((max(ts, t_min), min(te, t_max)))

        anom_list = extract_anomaly_windows(times_s, values, intervals, window_size, min_anom_pts)
        norm_list = extract_normal_windows(times_s, values, intervals, window_size, stride_normal)

        n_anom = len(anom_list)
        n_norm = len(norm_list)

        t_mid_ch = (t_min + t_max) / 2
        h1 = sum(1 for wt, _ in anom_list if wt[len(wt) // 2] < t_mid_ch)
        h2 = n_anom - h1

        total_ch = n_anom + n_norm
        ratio = f"{n_anom / total_ch:.1%}" if total_ch else "    —"

        print(
            f"  {chan_name:<{W - 2}} {n_anom:>9}  {h1:>5}  {h2:>5}  {n_norm:>9}  {ratio:>6}"
        )

        total_anom += n_anom
        total_norm += n_norm
        if n_anom > 0:
            channels_with_anom += 1

        del times_s, values
        gc.collect()

    total_wins  = total_anom + total_norm
    global_ratio = f"{total_anom / total_wins:.1%}" if total_wins else "    —"
    print("─" * 64)
    print(
        f"  {'TOTAL':<{W - 2}} {total_anom:>9}  {'':>5}  {'':>5}  {total_norm:>9}  {global_ratio:>6}"
    )
    print(f"\n  Channels scanned        : {len(zip_paths)}")
    print(f"  Channels with anomalies : {channels_with_anom} / {len(zip_paths)}")
    print(f"  Global anomaly ratio    : {global_ratio}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ESA-Mission1 → dataset.csv for LPI pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              # Scan all channels — anomaly stats, no CSV output:
              python experiments/s3_esa_adb/prepare_mission1.py \\
                  --data_dir /workspace/ESA-Mission1/ESA-Mission1 \\
                  --scan

              # Full run (all 76 channels — ~hours):
              python experiments/s3_esa_adb/prepare_mission1.py \\
                  --data_dir /workspace/ESA-Mission1/ESA-Mission1 \\
                  --out_dir  reference/data/esa_mission1

              # Single-channel sanity check (fast):
              python experiments/s3_esa_adb/prepare_mission1.py \\
                  --data_dir /workspace/ESA-Mission1/ESA-Mission1 \\
                  --out_dir  reference/data/esa_mission1_ch1 \\
                  --channel  channel_1
        """),
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to ESA-Mission1 directory (with labels.csv and channels/ subdir)",
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Output directory for dataset.csv (required unless --scan is used)",
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="Print per-channel anomaly stats without writing any CSV. Ignores --out_dir.",
    )
    parser.add_argument(
        "--channel", default=None, metavar="CHANNEL_NAME",
        help="Process only this channel (e.g. 'channel_1'). Default: all channels.",
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

    if args.scan:
        scan_channels(
            data_dir      = data_dir,
            only_channel  = args.channel,
            window_size   = args.window_size,
            stride_normal = args.stride_normal,
            min_anom_pts  = MIN_ANOMALY_PTS,
        )
        return

    if args.out_dir is None:
        parser.error("--out_dir is required unless --scan is used")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nESA-AD Mission1 adapter")
    print(f"  data_dir     : {data_dir}")
    print(f"  out_dir      : {out_dir}")
    print(f"  channel      : {args.channel or 'all'}")
    print(f"  window_size  : {args.window_size}  (~{args.window_size * 90 // 60} min at 90s/sample)")
    print(f"  stride_normal: {args.stride_normal}")
    print(f"  train_frac   : {args.train_fraction}")

    df = build_dataset(
        data_dir       = data_dir,
        out_dir        = out_dir,
        window_size    = args.window_size,
        stride_normal  = args.stride_normal,
        train_fraction = args.train_fraction,
        only_channel   = args.channel,
    )

    out_csv = out_dir / "dataset.csv"
    df.to_csv(out_csv)
    print(f"\n  Wrote {len(df):,} rows → {out_csv}")

    summary = print_summary(df, out_csv)
    (out_dir / "prep_summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
