"""
S3 — ESA-AD Dataset Explorer

Discovers the structure of ESA-Mission1 (and any other missions present)
before committing to a loader design.

Usage:
    python experiments/s3_transformer/explore_esa_ad.py \
        --data_dir /workspace/ESA-Mission1
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def explore_directory(root: Path, depth: int = 0, max_depth: int = 4) -> None:
    if depth > max_depth:
        return
    indent = "  " * depth
    try:
        entries = sorted(root.iterdir())
    except PermissionError:
        return
    for entry in entries:
        size_str = ""
        if entry.is_file():
            size = entry.stat().st_size
            if size > 1_000_000_000:
                size_str = f"  [{size / 1e9:.1f} GB]"
            elif size > 1_000_000:
                size_str = f"  [{size / 1e6:.1f} MB]"
            elif size > 1_000:
                size_str = f"  [{size / 1e3:.0f} KB]"
            print(f"{indent}{entry.name}{size_str}")
            if entry.suffix in {".csv", ".parquet", ".h5", ".hdf5", ".npz", ".npy"}:
                _inspect_file(entry, depth + 1)
        else:
            print(f"{indent}{entry.name}/")
            explore_directory(entry, depth + 1, max_depth)


def _inspect_file(path: Path, depth: int) -> None:
    indent = "  " * depth
    try:
        if path.suffix == ".csv":
            df = pd.read_csv(path, nrows=5)
            print(f"{indent}  columns ({len(df.columns)}): {list(df.columns)}")
            print(f"{indent}  dtypes:  {dict(df.dtypes)}")
            # Full row count
            n = sum(1 for _ in open(path)) - 1  # subtract header
            print(f"{indent}  rows: {n:,}")
            if "anomaly" in df.columns:
                full = pd.read_csv(path, usecols=["anomaly"])
                vc = full["anomaly"].value_counts()
                print(f"{indent}  anomaly dist: {dict(vc)}")
            if "train" in df.columns:
                full = pd.read_csv(path, usecols=["train"])
                vc = full["train"].value_counts()
                print(f"{indent}  train dist: {dict(vc)}")
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
            print(f"{indent}  columns ({len(df.columns)}): {list(df.columns)}")
            print(f"{indent}  rows: {len(df):,}")
        elif path.suffix in {".h5", ".hdf5"}:
            import h5py
            with h5py.File(path, "r") as f:
                print(f"{indent}  keys: {list(f.keys())}")
        elif path.suffix == ".npz":
            d = np.load(path, allow_pickle=True)
            print(f"{indent}  keys: {list(d.keys())}")
    except Exception as e:
        print(f"{indent}  [could not inspect: {e}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to ESA-AD directory")
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    print(f"\n{'='*60}")
    print(f"ESA-AD Explorer — {root.resolve()}")
    print(f"{'='*60}\n")

    explore_directory(root)

    print(f"\n{'='*60}")
    print("Done. Review the structure above before proceeding to S3 loader.")
    print("Key questions to answer:")
    print("  1. Is there a dataset.csv (pre-computed features)?")
    print("  2. Is there a segments.csv (raw time series)?")
    print("  3. What columns are present?")
    print("  4. Is there a 'train' split column or a fixed split?")
    print("  5. Anomaly ratio?")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
