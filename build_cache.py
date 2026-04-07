"""
One-time cache builder
======================
Merges individual per-sample .npy spectrograms into two contiguous files:
    specs.npy   — float32, shape (N, n_mels, n_frames)
    labels.npy  — int64,   shape (N,)

This eliminates the per-file I/O overhead during training.
Run once per split (train / valid / test) before training.

Usage:
    python build_cache.py
"""

import os
import json
import numpy as np
from pathlib import Path


def build_merged_cache(nsynth_root: str,
                       npy_cache_dir: str,
                       output_dir: str,
                       n_mels: int = 64,
                       n_frames: int = 345,
                       max_samples: int | None = None):
    """
    Reads examples.json from *nsynth_root*, matches each entry to a
    cached .npy in *npy_cache_dir*, and writes two merged files into
    *output_dir*:
        specs.npy   (N, n_mels, n_frames) float32
        labels.npy  (N,)                  int64
    """
    json_path = os.path.join(nsynth_root, "examples.json")
    with open(json_path, "r") as f:
        metadata = json.load(f)

    # Collect valid (npy_path, label) pairs
    pairs: list[tuple[str, int]] = []
    for name, info in metadata.items():
        npy_path = os.path.join(npy_cache_dir, name + ".npy")
        if os.path.exists(npy_path):
            pairs.append((npy_path, info["pitch"] % 12))

    if max_samples is not None:
        pairs = pairs[:max_samples]

    N = len(pairs)
    if N == 0:
        raise RuntimeError(
            f"No cached .npy files found in {npy_cache_dir}.\n"
            "Run your spectrogram precompute step first."
        )

    print(f"Found {N:,} cached spectrograms")

    # Allocate output arrays
    specs  = np.empty((N, n_mels, n_frames), dtype=np.float32)
    labels = np.empty(N, dtype=np.int64)

    for i, (npy_path, label) in enumerate(pairs):
        specs[i]  = np.load(npy_path)
        labels[i] = label
        if (i + 1) % 10_000 == 0 or (i + 1) == N:
            print(f"  loaded {i + 1:>7,} / {N:,}")

    # Write merged files
    os.makedirs(output_dir, exist_ok=True)
    specs_path  = os.path.join(output_dir, "specs.npy")
    labels_path = os.path.join(output_dir, "labels.npy")

    np.save(specs_path, specs)
    np.save(labels_path, labels)

    specs_mb  = os.path.getsize(specs_path)  / 1e6
    labels_mb = os.path.getsize(labels_path) / 1e6
    print(f"\nSaved:")
    print(f"  {specs_path}  — {specs_mb:.1f} MB  shape {specs.shape}")
    print(f"  {labels_path} — {labels_mb:.1f} MB  shape {labels.shape}")
    print("Done.")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Processes TRAIN and VALID splits in one run.
    # Each entry: (nsynth_root, npy_cache_dir, output_dir)
    #   nsynth_root   — folder containing examples.json
    #   npy_cache_dir — folder with individual .npy files (from precompute_cache.py)
    #   output_dir    — where to write the merged specs.npy + labels.npy
    # ------------------------------------------------------------------
    BASE = r"E:\programming\machine learning\projects\DL_project"

    SPLITS = [
        {
            "label":         "train",
            "nsynth_root":   rf"{BASE}\nsynth-train.jsonwav\nsynth-train",
            "npy_cache_dir": rf"{BASE}\nsynth-train.jsonwav\cache",
            "output_dir":    rf"{BASE}\nsynth-train.jsonwav\outputs",
        },
        {
            "label":         "valid",
            "nsynth_root":   rf"{BASE}\nsynth-valid.jsonwav\nsynth-valid",
            "npy_cache_dir": rf"{BASE}\nsynth-valid.jsonwav\cache",
            "output_dir":    rf"{BASE}\nsynth-valid.jsonwav\outputs",
        },
    ]

    for split in SPLITS:
        print(f"\n{'='*60}")
        print(f"  Split: {split['label']}")
        print(f"{'='*60}")
        build_merged_cache(
            nsynth_root   = split["nsynth_root"],
            npy_cache_dir = split["npy_cache_dir"],
            output_dir    = split["output_dir"],
            n_mels=64,
            n_frames=345,
            max_samples=None,
        )
