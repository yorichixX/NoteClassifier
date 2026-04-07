"""
Spectrogram Precompute Script
==============================
Reads every .wav in an NSynth audio folder, computes a mel spectrogram,
and saves it as an individual .npy file in the cache directory.

This is the STEP BEFORE build_cache.py:
    precompute_cache.py  →  one .npy per sample in cache/
    build_cache.py       →  merges all .npy → specs.npy + labels.npy

Run once per split (train / valid / test) before build_cache.py.

Usage:
    python precompute_cache.py
"""

import os
import numpy as np
import librosa
from pathlib import Path


# -----------------------------------------------------------------------
# Core function
# -----------------------------------------------------------------------

def precompute_spectrograms(
    audio_dir: str,
    cache_dir: str,
    sr: int = 22050,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_frames: int = 345,
    fmin: int = 80,
    fmax: int = 8000,
    skip_existing: bool = True,
):
    """
    For each .wav in *audio_dir*, compute a mel spectrogram and save as
    <stem>.npy in *cache_dir*.

    Args:
        audio_dir    : folder containing .wav files
        cache_dir    : destination for .npy files (created if absent)
        sr           : sample rate to resample to (NSynth is 16 kHz → 22050)
        n_mels       : number of mel bins          (must match training_loop.py)
        n_fft        : FFT window size             (must match training_loop.py)
        hop_length   : hop between frames          (must match training_loop.py)
        n_frames     : target time-frame count — spectrogram is cropped/padded
        fmin, fmax   : mel filterbank frequency range
        skip_existing: if True, skip files whose .npy already exists
    """
    os.makedirs(cache_dir, exist_ok=True)

    wav_files = sorted(Path(audio_dir).glob("*.wav"))
    total     = len(wav_files)

    if total == 0:
        raise RuntimeError(f"No .wav files found in: {audio_dir}")

    print(f"Found {total:,} .wav files in {audio_dir}")
    print(f"Saving .npy files to:          {cache_dir}")
    print(f"Mel params: sr={sr}, n_mels={n_mels}, n_fft={n_fft}, "
          f"hop={hop_length}, n_frames={n_frames}")
    print("-" * 60)

    errors   = []
    skipped  = 0
    computed = 0

    for i, wav_path in enumerate(wav_files):
        npy_path = os.path.join(cache_dir, wav_path.stem + ".npy")

        if skip_existing and os.path.exists(npy_path):
            skipped += 1
            continue

        try:
            y, _ = librosa.load(str(wav_path), sr=sr, mono=True)

            S = librosa.feature.melspectrogram(
                y=y, sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
            )
            S_db = librosa.power_to_db(S, ref=np.max)  # shape: (n_mels, T)

            # --- Crop or pad time axis to exactly n_frames ---
            if S_db.shape[1] >= n_frames:
                S_db = S_db[:, :n_frames]
            else:
                pad  = n_frames - S_db.shape[1]
                S_db = np.pad(S_db, ((0, 0), (0, pad)),
                              mode="constant", constant_values=S_db.min())

            # --- Normalise to [0, 1] per sample ---
            s_min, s_max = S_db.min(), S_db.max()
            if s_max > s_min:
                S_db = (S_db - s_min) / (s_max - s_min)

            np.save(npy_path, S_db.astype(np.float32))
            computed += 1

        except Exception as exc:
            errors.append((wav_path.name, str(exc)))

        # Progress every 5 000 files
        done = computed + skipped
        if (done % 5_000 == 0 and done > 0) or (i + 1) == total:
            print(f"  {done:>7,} / {total:,}  "
                  f"(computed {computed:,}, skipped {skipped:,}, "
                  f"errors {len(errors):,})")

    print("\nDone.")
    print(f"  Computed : {computed:,}")
    print(f"  Skipped  : {skipped:,}  (already existed)")
    print(f"  Errors   : {len(errors):,}")

    if errors:
        print("\nFailed files:")
        for name, msg in errors[:20]:
            print(f"  {name}: {msg}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


# -----------------------------------------------------------------------
# Entry point — edit the paths below, then run
# -----------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Edit these paths to match your layout, then run once per split.
    # ------------------------------------------------------------------

    AUDIO_DIR = r"E:\programming\machine learning\projects\DL_project\nsynth-train.jsonwav\nsynth-train\audio"
    CACHE_DIR = r"E:\programming\machine learning\projects\DL_project\nsynth-train.jsonwav\cache"

    # These MUST match the values used in build_cache.py and training_loop.py
    N_MELS     = 64
    N_FRAMES   = 345
    N_FFT      = 1024
    HOP_LENGTH = 256

    precompute_spectrograms(
        audio_dir  = AUDIO_DIR,
        cache_dir  = CACHE_DIR,
        sr         = 22050,
        n_mels     = N_MELS,
        n_fft      = N_FFT,
        hop_length = HOP_LENGTH,
        n_frames   = N_FRAMES,
        skip_existing = True,   # set False to recompute everything from scratch
    )
