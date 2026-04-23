"""
S3 — ESA-AD Mission1 raw-window adapter  (Quesada 2026)

Extracts fixed-size raw windows (z-score normalised, no features) from the
ESA-Mission1 dataset and saves them in a format ready for Conv1D autoencoder
training (run_ae_lpi.py).

Output (in --out_dir):
  windows.npy   — float32 array of shape (N, window_size), z-score per channel
  meta.csv      — columns: segment, anomaly, train, channel, sampling
  config.json   — window_size, stride_normal, channels, etc.

Design differences from prepare_mission1.py:
  • Per-channel z-score: μ/σ computed on the full channel signal before
    windowing (1-pass). This preserves absolute-level information — a
    potential anomaly signal — while bringing all channels to the same scale.
    A constant channel (σ < 1e-8) is left as-is (zeros after centring).
  • Output is raw windows, not feature vectors. window_size=256 default
    (vs 64 in prepare_mission1.py) to capture longer temporal structures.
  • Multiprocessing: same mp.Pool pattern as prepare_mission1.py.

All segmentation logic (extract_anomaly_windows, extract_normal_windows,
assign_train_test, …) is imported directly from prepare_mission1.py — no
code duplication.

──────────────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────────────
  # Single channel (fast test):
  python experiments/s3_esa_adb/prepare_mission1_raw.py \\
      --data_dir /workspace/ESA-Mission1/ESA-Mission1 \\
      --out_dir  reference/data/esa_mission1_raw_ch14 \\
      --channels channel_14

  # All channels, 16 workers:
  python experiments/s3_esa_adb/prepare_mission1_raw.py \\
      --data_dir /workspace/ESA-Mission1/ESA-Mission1 \\
      --out_dir  reference/data/esa_mission1_raw \\
      --n_workers 16
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import sys
import textwrap
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parents[2]))

from experiments.s3_esa_adb.prepare_mission1 import (
    MIN_ANOMALY_PTS,
    MIN_OVERLAP_FRAC,
    STRIDE_NORMAL,
    TRAIN_FRACTION,
    assign_train_test,
    detect_sampling_rate,
    extract_anomaly_windows,
    extract_normal_windows,
    load_channel_pickle,
    load_channel_zip,
    load_labels,
)

# ─── Defaults ─────────────────────────────────────────────────────────────────

WINDOW_SIZE    = 256   # ~6.4 h at 90 s/sample (ESA-Mission1 nominal)
STRIDE_RAW     = 128   # 50 % overlap for normal windows


# ─── Per-channel worker (module-level for mp.Pool pickling) ───────────────────

def _process_channel_raw(
    args: tuple,
) -> tuple[str, list[tuple[dict, np.ndarray]], dict]:
    """
    Worker: load one channel, z-score normalise, extract raw windows.

    Returns
    -------
    chan_name : str
    entries   : list of (meta_dict, window_array)
        meta_dict  — {anomaly, channel, sampling, _t_mid}
        window_array — float32 (window_size,) normalised signal
    stats     : dict with counts / timings (or 'error' key on failure)
    """
    ch_path, labels_df, has_abs_timestamps, use_zip, config = args
    chan_name   = ch_path.stem
    window_size = config["window_size"]
    stride      = config["stride_normal"]
    t_start     = time.perf_counter()

    # ── Load raw signal ───────────────────────────────────────────────────────
    try:
        if use_zip:
            times_s, values, t0 = load_channel_zip(ch_path)
        else:
            times_s, values = load_channel_pickle(ch_path)
            t0 = None
    except Exception as e:
        return chan_name, [], {"error": f"load failed: {e}",
                               "runtime": time.perf_counter() - t_start}

    if len(times_s) < window_size:
        return chan_name, [], {"error": f"only {len(times_s)} points",
                               "runtime": time.perf_counter() - t_start}

    # ── Per-channel z-score (full signal, 1-pass) ─────────────────────────────
    mu    = float(np.nanmean(values))
    sigma = float(np.nanstd(values))
    if sigma < 1e-8:
        sigma = 1.0   # constant channel: centre only, no scaling
    values_norm = ((values - mu) / sigma).astype(np.float32)

    dt_s = detect_sampling_rate(times_s)

    # ── Anomaly intervals for this channel ────────────────────────────────────
    mask_all  = labels_df["channel"].isna()
    mask_chan = labels_df["channel"] == chan_name
    relevant  = labels_df[mask_all | mask_chan]

    t_min, t_max = times_s[0], times_s[-1]
    intervals: list[tuple[float, float]] = []
    for _, row in relevant.iterrows():
        if has_abs_timestamps and t0 is not None and pd.notna(row["start_dt"]):
            t0_utc = t0.tz_localize("UTC") if t0.tzinfo is None else t0
            ts = float((row["start_dt"] - t0_utc).total_seconds())
            te = float((row["end_dt"]   - t0_utc).total_seconds())
        else:
            ts, te = float(row["start_s"]), float(row["end_s"])
        if te < t_min or ts > t_max:
            continue
        intervals.append((max(ts, t_min), min(te, t_max)))

    # ── Extract windows (on normalised signal) ────────────────────────────────
    anom_wins, n_disc = extract_anomaly_windows(
        times_s, values_norm, intervals,
        window_size, MIN_ANOMALY_PTS, config["min_overlap_frac"],
    )
    norm_wins = extract_normal_windows(
        times_s, values_norm, intervals, window_size, stride,
    )

    # ── Build (meta, window) pairs ────────────────────────────────────────────
    entries: list[tuple[dict, np.ndarray]] = []
    for label, wins in [(1, anom_wins), (0, norm_wins)]:
        for wt, wv in wins:
            meta = {
                "anomaly":  label,
                "channel":  chan_name,
                "sampling": round(dt_s),
                "_t_mid":   float(wt[len(wt) // 2]),
            }
            entries.append((meta, wv.astype(np.float32)))

    runtime  = time.perf_counter() - t_start
    n_anom   = len(anom_wins)
    n_norm   = len(norm_wins)
    total    = n_anom + n_norm
    ratio    = (n_anom / total) if total else 0.0

    print(
        f"[CHANNEL_DONE] {chan_name} | n_windows={total} | "
        f"n_anomaly={n_anom} | ratio={ratio:.3f} | "
        f"n_disc={n_disc} | runtime={runtime:.1f}s",
        flush=True,
    )

    del times_s, values, values_norm
    gc.collect()

    return chan_name, entries, {
        "n_anom_wins": n_anom,
        "n_norm_wins": n_norm,
        "n_disc":      n_disc,
        "dt_s":        dt_s,
        "runtime":     runtime,
    }


# ─── Main pipeline ────────────────────────────────────────────────────────────

def build_raw_dataset(
    data_dir:         Path,
    out_dir:          Path,
    window_size:      int   = WINDOW_SIZE,
    stride_normal:    int   = STRIDE_RAW,
    train_fraction:   float = TRAIN_FRACTION,
    only_channel:     str | None = None,
    only_channels:    list[str] | None = None,
    min_overlap_frac: float = MIN_OVERLAP_FRAC,
    n_workers:        int   = 1,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Full pipeline: discover channels → load → z-score → window → split.

    Returns
    -------
    windows : np.ndarray, shape (N, window_size), float32
    meta    : pd.DataFrame, columns [segment, anomaly, train, channel, sampling]
    """
    # ── Locate labels ─────────────────────────────────────────────────────────
    labels_candidates = (
        list(data_dir.glob("labels*.csv")) +
        list(data_dir.glob("Labels*.csv"))
    )
    if not labels_candidates:
        raise FileNotFoundError(
            f"No labels*.csv found in {data_dir}. Contents:\n"
            + "\n".join(p.name for p in sorted(data_dir.iterdir()))
        )
    labels_path = labels_candidates[0]
    print(f"\n  Labels file : {labels_path.name}")

    # ── Locate channel zips ───────────────────────────────────────────────────
    channels_dir = data_dir / "channels"
    if channels_dir.is_dir():
        zip_paths = sorted(channels_dir.glob("channel_*.zip"))
    else:
        zip_paths = sorted(data_dir.glob("channel_*.zip"))

    if zip_paths:
        use_zip = True
        print(f"  Channels    : {len(zip_paths)} channel zips (deflate64)")
    else:
        pkl_paths = sorted(data_dir.glob("*.pkl")) or sorted(data_dir.rglob("*.pkl"))
        if not pkl_paths:
            raise FileNotFoundError(f"No channel_*.zip or .pkl files under {data_dir}")
        zip_paths = pkl_paths
        use_zip   = False
        print(f"  Channels    : {len(zip_paths)} pickle files")

    # ── Channel filter ────────────────────────────────────────────────────────
    if only_channels:
        wanted    = set(only_channels)
        zip_paths = [p for p in zip_paths if p.stem in wanted]
        missing   = wanted - {p.stem for p in zip_paths}
        if missing:
            raise ValueError(f"--channels not found: {sorted(missing)}")
        print(f"  Filter      : {len(zip_paths)} channels from --channels list")
    elif only_channel:
        zip_paths = [p for p in zip_paths if p.stem == only_channel]
        if not zip_paths:
            raise ValueError(f"--channel '{only_channel}' not found")
        print(f"  Filter      : only '{only_channel}'")

    # ── Labels ────────────────────────────────────────────────────────────────
    labels_df         = load_labels(labels_path)
    has_abs_timestamps = not labels_df["start_dt"].isna().all()

    # ── Dispatch workers ─────────────────────────────────────────────────────
    config = {
        "window_size":      window_size,
        "stride_normal":    stride_normal,
        "min_overlap_frac": min_overlap_frac,
    }
    work = [
        (p, labels_df, has_abs_timestamps, use_zip, config)
        for p in zip_paths
    ]

    effective_workers = max(1, min(n_workers, len(work)))
    print(f"  Workers     : {effective_workers} "
          f"(requested {n_workers}, {len(work)} channels)")
    print(f"  Window size : {window_size}  "
          f"(~{window_size * 90 // 60} min at 90s/sample)")
    print(f"  Stride norm : {stride_normal}")

    results: list[tuple[str, list, dict]] = []
    if effective_workers == 1:
        for w in work:
            results.append(_process_channel_raw(w))
    else:
        with mp.Pool(effective_workers) as pool:
            for res in pool.imap_unordered(_process_channel_raw, work):
                results.append(res)

    # ── Collect ───────────────────────────────────────────────────────────────
    all_entries: list[tuple[dict, np.ndarray]] = []
    ok_channels:    list[str] = []
    error_channels: list[tuple[str, str]] = []

    for chan_name, entries, stats in results:
        if "error" in stats:
            warnings.warn(f"{chan_name}: {stats['error']}")
            error_channels.append((chan_name, stats["error"]))
            continue
        all_entries.extend(entries)
        ok_channels.append(chan_name)

    print(
        f"\n  Channels OK : {len(ok_channels)} / {len(zip_paths)}"
        + (f"  | failed: {[c for c, _ in error_channels]}" if error_channels else "")
    )

    if not all_entries:
        raise RuntimeError("No windows created. Check data and labels.")

    # ── Sort by (channel, _t_mid) for deterministic order ────────────────────
    # Add a stable unique ID before sorting so we can re-align after assign_train_test
    for idx, (meta, _) in enumerate(all_entries):
        meta["_id"] = idx

    all_entries.sort(key=lambda e: (e[0]["channel"], e[0]["_t_mid"]))

    all_metas   = [e[0] for e in all_entries]
    all_windows = [e[1] for e in all_entries]

    # ── Temporal train/test split (per channel, by _t_mid) ────────────────────
    # assign_train_test mutates dicts in-place and may re-order by channel
    all_metas = assign_train_test(all_metas, train_fraction)

    # Re-align windows with (potentially reordered) metas via _id
    id_to_window: dict[int, np.ndarray] = {
        e[0]["_id"]: e[1] for e in all_entries
    }
    all_windows = [id_to_window[m["_id"]] for m in all_metas]

    # Final deterministic sort
    paired = sorted(zip(all_metas, all_windows), key=lambda x: (x[0]["channel"], x[0]["_t_mid"]))
    all_metas, all_windows = zip(*paired) if paired else ([], [])
    all_metas   = list(all_metas)
    all_windows = list(all_windows)

    # ── Assign monotonic segment IDs ──────────────────────────────────────────
    for seg_id, meta in enumerate(all_metas, start=1):
        meta["segment"] = seg_id

    # ── Build outputs ─────────────────────────────────────────────────────────
    windows_arr = np.stack(all_windows, axis=0)   # (N, window_size)

    meta_df = pd.DataFrame([
        {
            "segment": m["segment"],
            "anomaly": m["anomaly"],
            "train":   m["train"],
            "channel": m["channel"],
            "sampling": m["sampling"],
        }
        for m in all_metas
    ]).set_index("segment")

    return windows_arr, meta_df


# ─── Summary ──────────────────────────────────────────────────────────────────

def _print_summary(meta_df: pd.DataFrame, out_dir: Path, window_size: int) -> None:
    train = meta_df[meta_df["train"] == 1]
    test  = meta_df[meta_df["train"] == 0]

    print("\n" + "=" * 60)
    print("ESA-AD Mission1 — raw windows summary")
    print("=" * 60)
    print(f"  Total windows  : {len(meta_df):,}")
    print(f"  Channels       : {meta_df['channel'].nunique()}")
    print(f"  Window size    : {window_size}")
    print(f"  Sampling rates : {sorted(meta_df['sampling'].unique())} (seconds)")
    print()
    print(f"  TRAIN ({len(train):,} windows)")
    print(f"    anomalous : {(train['anomaly']==1).sum():,}  ({(train['anomaly']==1).mean():.1%})")
    print(f"    normal    : {(train['anomaly']==0).sum():,}  ({(train['anomaly']==0).mean():.1%})")
    print()
    print(f"  TEST  ({len(test):,} windows)")
    print(f"    anomalous : {(test['anomaly']==1).sum():,}  ({(test['anomaly']==1).mean():.1%})")
    print(f"    normal    : {(test['anomaly']==0).sum():,}  ({(test['anomaly']==0).mean():.1%})")
    print()
    print(f"  Output dir : {out_dir}")
    print("=" * 60)
    print()
    print("Next step:")
    print("  python experiments/s3_esa_adb/run_ae_lpi.py \\")
    print(f"      --raw_dir {out_dir}")
    print("=" * 60)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ESA-Mission1 raw windows for AE training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              # Single-channel test (fast):
              python experiments/s3_esa_adb/prepare_mission1_raw.py \\
                  --data_dir /workspace/ESA-Mission1/ESA-Mission1 \\
                  --out_dir  reference/data/esa_mission1_raw_ch14 \\
                  --channels channel_14

              # All 76 channels, 16 workers:
              python experiments/s3_esa_adb/prepare_mission1_raw.py \\
                  --data_dir /workspace/ESA-Mission1/ESA-Mission1 \\
                  --out_dir  reference/data/esa_mission1_raw \\
                  --n_workers 16
        """),
    )
    parser.add_argument("--data_dir", required=True,
                        help="ESA-Mission1 directory (with labels.csv and channels/ subdir)")
    parser.add_argument("--out_dir",  required=True,
                        help="Output directory for windows.npy / meta.csv / config.json")
    parser.add_argument("--channel",  default=None, metavar="CHANNEL_NAME",
                        help="Single channel to process (e.g. 'channel_14')")
    parser.add_argument("--channels", nargs="+", default=None, metavar="CHANNEL_NAME",
                        help="Explicit list of channels (takes priority over --channel)")
    parser.add_argument("--window_size",    type=int,   default=WINDOW_SIZE,
                        help=f"Window length in points (default: {WINDOW_SIZE})")
    parser.add_argument("--stride_normal",  type=int,   default=STRIDE_RAW,
                        help=f"Stride for normal windows (default: {STRIDE_RAW})")
    parser.add_argument("--train_fraction", type=float, default=TRAIN_FRACTION,
                        help=f"Temporal train fraction (default: {TRAIN_FRACTION})")
    parser.add_argument("--min_overlap_frac", type=float, default=MIN_OVERLAP_FRAC,
                        help=f"Min anomaly overlap fraction (default: {MIN_OVERLAP_FRAC})")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Parallel channel workers (default: 1 = serial)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_display = (
        " ".join(args.channels) if args.channels
        else (args.channel or "all")
    )
    print(f"\nESA-AD Mission1 raw-window adapter")
    print(f"  data_dir        : {data_dir}")
    print(f"  out_dir         : {out_dir}")
    print(f"  channel(s)      : {channel_display}")
    print(f"  n_workers       : {args.n_workers}")
    print(f"  window_size     : {args.window_size}")
    print(f"  stride_normal   : {args.stride_normal}")
    print(f"  train_fraction  : {args.train_fraction}")
    print(f"  min_overlap_frac: {args.min_overlap_frac:.0%}")

    windows, meta_df = build_raw_dataset(
        data_dir         = data_dir,
        out_dir          = out_dir,
        window_size      = args.window_size,
        stride_normal    = args.stride_normal,
        train_fraction   = args.train_fraction,
        only_channel     = args.channel,
        only_channels    = args.channels,
        min_overlap_frac = args.min_overlap_frac,
        n_workers        = args.n_workers,
    )

    # ── Save windows.npy ──────────────────────────────────────────────────────
    windows_path = out_dir / "windows.npy"
    np.save(windows_path, windows)
    print(f"\n  Saved windows : {windows.shape}  →  {windows_path}")

    # ── Save meta.csv ─────────────────────────────────────────────────────────
    meta_path = out_dir / "meta.csv"
    meta_df.to_csv(meta_path)
    print(f"  Saved meta    : {len(meta_df)} rows  →  {meta_path}")

    # ── Save config.json ──────────────────────────────────────────────────────
    channels_processed = sorted(meta_df["channel"].unique().tolist())
    config = {
        "window_size":       args.window_size,
        "stride_normal":     args.stride_normal,
        "train_fraction":    args.train_fraction,
        "min_overlap_frac":  args.min_overlap_frac,
        "n_windows":         int(len(meta_df)),
        "n_channels":        int(meta_df["channel"].nunique()),
        "channels":          channels_processed,
        "normalization":     "z-score per channel (full signal mu/sigma)",
    }
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config  : {config_path}")

    _print_summary(meta_df, out_dir, args.window_size)


if __name__ == "__main__":
    main()
