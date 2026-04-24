"""Memory + time monitor for long GPU runs."""
import os
import time

import psutil
import torch


def log_mem(tag: str) -> None:
    rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
    gpu_alloc = 0.0
    gpu_reserved = 0.0
    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
    print(
        f"[MEM {tag}] RSS={rss_gb:.2f} GB  "
        f"GPU_alloc={gpu_alloc:.2f} GB  GPU_reserved={gpu_reserved:.2f} GB",
        flush=True,
    )


class Timer:
    """Context manager para medir tiempos por seed."""

    def __init__(self, tag: str):
        self.tag = tag

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.t0
        print(f"[TIME {self.tag}] {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
