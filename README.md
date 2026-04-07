# 🎵 NoteClassifier — Musical Note Recognition with Deep Learning

A from-scratch deep learning pipeline that classifies musical notes (C, C#, D, … B) from raw audio using a CNN trained on Google's [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth). Built as an educational project to understand every layer of the audio → prediction pipeline.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Step 1 — Precompute Spectrograms](#step-1--precompute-spectrograms)
  - [Step 2 — Build Merged Cache](#step-2--build-merged-cache)
  - [Step 3 — Train](#step-3--train)
  - [Step 4 — Inference & Explainability](#step-4--inference--explainability)
  - [Step 5 — Gradient Inspection](#step-5--gradient-inspection)
- [How It Works](#how-it-works)
  - [Audio Loading & Framing](#1-audio-loading--framing)
  - [Mel Spectrogram](#2-mel-spectrogram)
  - [CNN Model](#3-cnn-model)
  - [Training Loop](#4-training-loop)
  - [Inference & Explainability](#5-inference--explainability)
- [Performance Optimisations](#performance-optimisations)
- [Results](#results)

---

## Overview

Given a short audio clip of a single musical note (any instrument, any octave), the model predicts which of the 12 pitch classes the note belongs to:

```
C  |  C#  |  D  |  D#  |  E  |  F  |  F#  |  G  |  G#  |  A  |  A#  |  B
```

The pipeline converts raw `.wav` audio into a mel spectrogram (a 2D time-frequency image), then feeds it through a convolutional neural network — the same kind of architecture used for image classification, repurposed for audio.

### Why spectrograms?

Sound is a 1D signal (amplitude over time), but musical notes are defined by their frequency content. A mel spectrogram transforms the signal into a 2D image where:
- **Y-axis** = frequency (mel-scaled to match human hearing)
- **X-axis** = time
- **Colour** = energy at that frequency and time

This lets us apply proven image classification techniques (CNNs) to audio problems.

---

## Architecture

```
              ┌─────────────────────────────────────────────────────┐
              │                 NoteClassifierCNN                   │
              │                                                     │
  Input       │   Conv Block 1         Conv Block 2       Dense     │   Output
(1, 64, 345)  │  ┌─────────────┐     ┌─────────────┐   ┌────────┐  │  (12,)
──────────────│──│ Conv2d(1→32) │─────│ Conv2d(32→64)│───│Dropout │──│──────────
mel spectro.  │  │ BatchNorm2d │     │ BatchNorm2d │   │FC(→128)│  │  logits
              │  │ ReLU        │     │ ReLU        │   │ReLU    │  │  (one per
              │  │ MaxPool(2×2)│     │ MaxPool(2×2)│   │FC(→12) │  │   note)
              │  └─────────────┘     └─────────────┘   └────────┘  │
              │                                                     │
              │  (1,32,32,172)       (1,64,16,86)      (88064→128→12)
              └─────────────────────────────────────────────────────┘

Parameters: ~11.3M (mostly in fc1: 88,064 × 128 = 11.3M)
```

| Layer | What it does |
|---|---|
| `conv1` (32 filters, 3×3) | Detects low-level patterns: edges, onsets, frequency bands |
| `bn1` (BatchNorm) | Normalises activations → stabilises training, prevents vanishing gradients |
| `pool1` (2×2 MaxPool) | Halves spatial resolution: 64×345 → 32×172 |
| `conv2` (64 filters, 3×3) | Combines low-level features into higher-level patterns: harmonics, timbral texture |
| `bn2` + `pool2` | Same as above: 32×172 → 16×86 |
| `dropout` (50%) | Randomly zeros half the neurons during training to prevent overfitting |
| `fc1` (88,064 → 128) | Compresses the entire feature map into 128 summary values |
| `fc2` (128 → 12) | Maps 128 features to 12 raw scores (logits), one per note class |

---

## Project Structure

```
DL_project/
│
├── audio_loading.py           # Module 1: WAV loading, framing, Hann windowing
├── mel_spectrogram.py         # Module 2: FFT, mel filterbank, spectrogram
├── model.py                   # Module 3: CNN architecture definition
├── training_loop.py           # Module 4: Dataset, training, early stopping
├── inference_explainability.py# Module 5: Prediction, activation maps, saliency
│
├── precompute_cache.py        # Preprocessing: .wav → individual .npy spectrograms
├── build_cache.py             # Preprocessing: individual .npy → merged specs.npy
├── gradient_inspector.py      # Diagnostics: gradient flow visualisation
│
├── model.py                   # CNN model definition
├── requirements.txt           # Python dependencies
├── .gitignore
│
├── nsynth-train.jsonwav/      # NSynth training split (not tracked in git)
│   ├── nsynth-train/
│   │   ├── audio/             # ~289K .wav files
│   │   └── examples.json      # metadata (pitch, instrument, etc.)
│   ├── cache/                 # individual .npy spectrograms
│   └── outputs/               # merged specs.npy + labels.npy
│
├── nsynth-valid.jsonwav/      # NSynth validation split (same structure)
├── nsynth-test.jsonwav/       # NSynth test split (same structure)
│
├── note_classifier_weights.pt # Trained model weights (generated)
├── best_weights.pt            # Best checkpoint during training (generated)
├── training_curves.png        # Loss/accuracy/time plots (generated)
├── gradient_flow.png          # Gradient magnitude bar chart (generated)
└── gradient_histograms.png    # Gradient distribution histograms (generated)
```

---

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (recommended; CPU works but is much slower)
- ~15 GB disk space for the NSynth dataset + caches

### 1. Install dependencies

```bash
# Install PyTorch with CUDA first (check your CUDA version with: nvidia-smi)
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install the rest:
pip install -r requirements.txt
```

### 2. Download NSynth dataset

Download the [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth) (train, valid, test splits) and extract them into the project directory:

```
DL_project/
├── nsynth-train.jsonwav/
│   └── nsynth-train/
│       ├── audio/            ← ~289,205 .wav files
│       └── examples.json
├── nsynth-valid.jsonwav/
│   └── nsynth-valid/
│       ├── audio/            ← ~12,678 .wav files
│       └── examples.json
└── nsynth-test.jsonwav/
    └── nsynth-test/
        ├── audio/            ← ~4,096 .wav files
        └── examples.json
```

---

## Usage

### Execution order (run each step once, in order)

```
precompute_cache.py  →  build_cache.py  →  training_loop.py  →  inference_explainability.py
     (once)               (once)           (as needed)              (after training)
```

### Step 1 — Precompute Spectrograms

Converts every `.wav` file into a normalised mel spectrogram saved as an individual `.npy` file.

```bash
python precompute_cache.py
```

**What it does:**
1. Loads each `.wav` at 22,050 Hz sample rate
2. Computes an 64-bin mel spectrogram (80–8000 Hz range)
3. Converts power to dB scale, normalises to [0, 1]
4. Crops/pads to exactly 345 time frames
5. Saves as `cache/<filename>.npy`

> **Note:** Edit `AUDIO_DIR` and `CACHE_DIR` in the script's `__main__` block for each split (train, valid, test), then run once per split. Uses `skip_existing=True` to resume interrupted runs.

**Time:** ~30–60 minutes for the training split (289K files).

### Step 2 — Build Merged Cache

Merges thousands of individual `.npy` files into two contiguous files per split for high-performance memory-mapped I/O.

```bash
python build_cache.py
```

**What it produces:**
- `outputs/specs.npy` — shape `(N, 64, 345)`, all spectrograms stacked
- `outputs/labels.npy` — shape `(N,)`, pitch class labels (0–11)

**Why:** Loading 289K individual files during training causes severe GPU starvation (GPU alternates between 100% and 0% utilisation). The merged file enables `np.memmap` — the OS maps it into virtual memory and serves samples via pointer arithmetic with zero file-open overhead.

**Time:** ~5–10 minutes for the training split.

### Step 3 — Train

```bash
python training_loop.py
```

**What happens:**
1. Auto-detects the merged cache (`specs.npy`) and uses memory-mapped loading (`[memmap ⚡]` in output)
2. Falls back to individual `.npy` files if merged cache doesn't exist
3. Trains for up to 50 epochs with early stopping (patience=7)
4. Saves `best_weights.pt` (best checkpoint) and `note_classifier_weights.pt` (final weights)
5. Generates `training_curves.png` with loss, accuracy, and timing plots

**Key hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| Batch size | 256 | Fills GPU memory efficiently on 8 GB VRAM |
| Learning rate | 3e-4 | Conservative starting point for Adam |
| Optimizer | Adam | Adapts per-weight learning rates using gradient history |
| Early stopping | patience=7 | Stops when validation loss plateaus for 7 consecutive epochs |
| Gradient clipping | max_norm=1.0 | Prevents exploding gradients |
| Augmentation | time shift ±10, freq shift ±2, Gaussian noise | Training set only — increases robustness |
| Workers | 4 | Parallel data loading with `pin_memory` and `persistent_workers` |

**Expected output:**
```
NSynthDataset [memmap ⚡]: 289,205 samples  | augment=True
NSynthDataset [memmap ⚡]: 12,678 samples   | augment=False

Epoch   1/50  |  Train: 1.5871  |  Val: 0.9423  |  Acc: 68.2%  |  Time: 45.3s
Epoch   2/50  |  Train: 0.8104  |  Val: 0.7156  |  Acc: 74.8%  |  Time: 44.1s
...
```

### Step 4 — Inference & Explainability

Run the trained model on any `.wav` file to get a prediction plus visual explanations.

```bash
python inference_explainability.py
```

**Edit `WAV_PATH`** in the script to point to your target `.wav` file.

**Outputs:**

| File | Description |
|---|---|
| Terminal output | Top-3 predictions with confidence bars |
| `module_05_activations.png` | Feature maps from conv1 and conv2 — shows what patterns each filter detected |
| `module_05_saliency.png` | Saliency map — highlights which spectrogram regions most influenced the prediction |

**Saliency maps** use the same chain rule as backpropagation, but applied to the input pixels instead of the weights. Bright regions = "the model looked here to make its decision."

### Step 5 — Gradient Inspection

Diagnose the health of gradient flow through the network.

```bash
python gradient_inspector.py
```

**Outputs:**

| Output | What it shows |
|---|---|
| Gradient table (terminal) | Mean/max gradient magnitude and % zero per layer |
| Per-filter bars (terminal) | Which conv filters and fc neurons receive the strongest learning signal |
| `gradient_flow.png` | Bar chart of gradient magnitudes across all layers (linear + log scale) |
| `gradient_histograms.png` | Distribution of gradient values for each major weight layer |

**What to look for:**

| Symptom | Diagnosis |
|---|---|
| All bars roughly equal height | ✅ Healthy gradient flow |
| Early layers (conv1) much smaller than later layers (fc2) | ⚠️ Vanishing gradients |
| Any layer's gradient > 1.0 | ⚠️ Exploding gradients |
| High % zero in conv layers | ⚠️ Dead ReLU neurons |

---

## How It Works

### 1. Audio Loading & Framing

**File:** `audio_loading.py`

Raw audio is a 1D array of amplitude samples at 22,050 Hz.  
To analyse frequency content, we split it into short overlapping **frames** (25ms windows, 12.5ms hop).  
Each frame is multiplied by a **Hann window** to taper the edges, preventing spectral leakage when we apply FFT.

```
Raw waveform:  ──────/\──/\/\──/\──────────
                     ◄─frame 1─►
                        ◄─frame 2─►     (overlapping)
                           ◄─frame 3─►
```

### 2. Mel Spectrogram

**File:** `mel_spectrogram.py`

Each windowed frame is transformed via FFT into a frequency spectrum. A **mel filterbank** (64 triangular filters spaced on the mel scale) compresses the frequency axis to match human pitch perception — we hear the difference between 200 Hz and 400 Hz much more clearly than between 8000 Hz and 8200 Hz.

```
                  Frequency (Hz)
                  200   400   800   1600  3200  6400
Mel filterbank:   ▲▲▲   ▲▲▲   ▲▲▲    ▲▲▲▲   ▲▲▲▲▲▲▲
                  ← narrow →  ←──── wider ────────→
```

The result is a 2D image: **(64 mel bins × 345 time frames)**, normalised to [0, 1].

### 3. CNN Model

**File:** `model.py`

A compact 2-block CNN processes the spectrogram "image":

- **Block 1:** 32 filters (3×3) → BatchNorm → ReLU → MaxPool(2×2)
- **Block 2:** 64 filters (3×3) → BatchNorm → ReLU → MaxPool(2×2)  
- **Head:** Flatten → Dropout(0.5) → FC(128) → ReLU → FC(12)

**BatchNorm** after each conv layer normalises intermediate activations, which:
- Stabilises gradient flow (prevents vanishing/exploding)
- Acts as mild regularisation
- Allows higher learning rates

### 4. Training Loop

**File:** `training_loop.py`

The training loop implements:

- **CrossEntropyLoss:** Combines log-softmax + NLL in one numerically stable operation
- **Adam optimizer:** Adapts the learning rate per-weight based on gradient history (1st and 2nd moments)
- **Early stopping:** Monitors validation loss; if it doesn't improve for 7 epochs, training stops and the best checkpoint is restored
- **Data augmentation:** Random time shifts, frequency shifts, and Gaussian noise applied to training data only
- **Gradient clipping:** `clip_grad_norm_(max_norm=1.0)` prevents exploding gradients from destabilising training

### 5. Inference & Explainability

**File:** `inference_explainability.py`

After training, the model can:

1. **Predict** the note from any `.wav` file (with confidence scores)
2. **Activation maps:** Forward hooks capture the output of each conv layer, showing what patterns each filter detected
3. **Saliency maps:** Backpropagation through the input (`dPrediction/dInput`) reveals which spectrogram regions drove the prediction — essentially asking *"which frequencies at which times did the model rely on?"*

---

## Performance Optimisations

The training pipeline includes several optimisations to maximise GPU utilisation:

| Optimisation | Problem it solves | Impact |
|---|---|---|
| **Memory-mapped I/O** (`np.memmap`) | Per-file `open()`/`read()`/`close()` overhead with 289K files | GPU utilisation: 20% → 80%+ |
| **Lazy memmap open** | Windows `spawn` multiprocessing can't pickle `np.memmap` file handles | Fixes `MemoryError` on worker spawn |
| **`np.load(mmap_mode='r')`** | Raw `np.memmap()` doesn't skip `.npy` headers → data offset by 128 bytes | Fixes astronomically wrong validation loss |
| **`pin_memory=True`** | Extra CPU-side copy on every `.to(device)` call | DMA reads directly from page-locked RAM |
| **`non_blocking=True`** | CPU waits for GPU transfer to complete before continuing | CPU→GPU transfer overlaps with next-batch prefetch |
| **`persistent_workers=True`** | Worker processes respawn every epoch (slow on Windows) | Workers stay alive between epochs |

---

## Results

After training on the full NSynth training set (~289K samples), with early stopping:

| Metric | Value |
|---|---|
| Validation accuracy | ~70–75% |
| Number of classes | 12 (chromatic pitch classes) |
| Training time | ~45s per epoch (RTX 4060, memmap mode) |
| Model size | ~43 MB (11.3M parameters) |

> **Note:** 75% accuracy across 12 classes (vs 8.3% random chance) is strong for a 2-layer CNN. The main confusions occur between harmonically related notes (e.g., C and G — a perfect fifth) and between octaves of the same note class. Deeper architectures or attention mechanisms could improve this further.

---

## License

This project uses the [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth) by Google Magenta, released under a Creative Commons Attribution 4.0 International License (CC BY 4.0).
