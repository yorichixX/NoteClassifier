"""
MODULE 4 — Training Loop
==========================
Stripped to just: cached dataset -> train -> early stopping -> save weights.
Experiment mode removed for now (add back later).

What this does:
  1. Loads pre-computed .npy spectrograms from cache (fast, GPU-friendly)
  2. Applies augmentation on training set each epoch
  3. Trains with early stopping — stops when val loss plateaus,
     restores the best weights seen during training
  4. Saves final weights to note_classifier_weights.pt
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(__file__))
from model import NoteClassifierCNN


# ================================================================
# STEP 1 — Dataset (cached mode only)
# ================================================================
class NSynthDataset(Dataset):
    """
    Two modes, chosen automatically:

    FAST  (memmap) — if *cache_dir* contains specs.npy + labels.npy
        np.memmap lets the OS map the file into virtual memory and serve
        slices with zero per-file open() overhead.  Eliminates the
        100↔0→100 GPU starvation pattern caused by individual file I/O.
        Build the merged files once with build_cache.py.

    SLOW  (individual .npy) — fallback when merged files don't exist yet
        Falls back to loading one .npy per sample (requires examples.json
        via nsynth_root).  Still faster than raw WAV loading.
    """

    def __init__(self, nsynth_root, cache_dir,
                 n_mels=64, n_frames=345,
                 max_samples=None, augment=False):

        self.n_mels   = n_mels
        self.n_frames = n_frames
        self.augment  = augment

        specs_path  = os.path.join(cache_dir, 'specs.npy')
        labels_path = os.path.join(cache_dir, 'labels.npy')

        if os.path.exists(specs_path) and os.path.exists(labels_path):
            labels_arr = np.load(labels_path)   # (N,) int64 — small, safe to pickle
            N          = len(labels_arr)

            if max_samples is not None:
                labels_arr = labels_arr[:max_samples]
                N          = len(labels_arr)

            self.labels       = labels_arr
            self._specs_path  = specs_path          # stored for lazy open
            self._specs       = None                # opened lazily per worker
            self.mode         = 'memmap'
            self.samples      = None
            print(f"NSynthDataset [memmap ⚡]: {N:,} samples  "
                  f"| augment={augment}  | {cache_dir}")

        else:
            # ── SLOW PATH: individual .npy per sample ───────────────────
            json_path = os.path.join(nsynth_root, 'examples.json')
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            audio_dir = os.path.join(nsynth_root, 'audio')
            pairs: list[tuple[str, int]] = []
            for name, info in metadata.items():
                npy_path = os.path.join(cache_dir, name + '.npy')
                if os.path.exists(npy_path):
                    pairs.append((npy_path, info['pitch'] % 12))

            if max_samples is not None:
                pairs = pairs[:max_samples]

            self.samples = pairs
            self.mode    = 'files'
            self.labels  = None
            self.specs   = None
            print(f"NSynthDataset [files]: {len(pairs):,} samples  "
                  f"| augment={augment}  | {cache_dir}")

            if len(pairs) == 0:
                raise RuntimeError(
                    f"No cached .npy files found in {cache_dir}.\n"
                    f"Run precompute_cache.py first, or run build_cache.py "
                    f"to generate the fast merged cache.")

    def __len__(self):
        if self.mode == 'memmap':
            return len(self.labels)
        return len(self.samples)

    def _augment(self, S):
        """Random time shift, freq shift, and Gaussian noise — all in-place on a copy."""
        S = np.roll(S, np.random.randint(-10, 10), axis=1)   # time shift
        S = np.roll(S, np.random.randint(-2,  3),  axis=0)   # freq shift
        S = S + np.random.randn(*S.shape).astype(np.float32) * 0.01
        return np.clip(S, 0.0, 1.0)

    def __getitem__(self, idx):
        if self.mode == 'memmap':
            # Lazy open: the memmap is created once per worker process on the
            # first __getitem__ call, then reused for all subsequent calls in
            # that worker.  This avoids pickling the file handle across spawn.
            if self._specs is None:
                # np.load with mmap_mode='r' returns a memmap that correctly
                # skips the .npy header.  Raw np.memmap() starts at byte 0
                # and reads header bytes as data → garbage values → 1e26 loss.
                self._specs = np.load(self._specs_path, mmap_mode='r')
            # np.array() copies the slice into a plain contiguous array so
            # each sample returned to the DataLoader is independent.
            S     = np.array(self._specs[idx], dtype=np.float32)
            label = int(self.labels[idx])
        else:
            npy_path, label = self.samples[idx]
            S = np.load(npy_path)

        if self.augment:
            S = self._augment(S)

        # (n_mels, n_frames) → (1, n_mels, n_frames)  — channel dim for Conv2d
        return torch.FloatTensor(S).unsqueeze(0), torch.tensor(label, dtype=torch.long)


# ================================================================
# STEP 2 — Loss and optimizer
# ================================================================
def setup_loss_and_optimizer(model, learning_rate=3e-4):
    """
    CrossEntropyLoss = log-softmax + negative log-likelihood in one step.
    Numerically more stable than softmax -> log -> NLL separately.
    Expects raw logits (not softmaxed) as input.

    Adam adapts the learning rate per weight using gradient history.
    lr=3e-4 is a common safe starting point — lower than default 1e-3
    to reduce risk of overshooting with larger datasets.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer


# ================================================================
# STEP 3 — Single training step
# ================================================================
def train_one_batch(model, batch_x, batch_y, criterion, optimizer, device):
    model.train()

    batch_x = batch_x.to(device, non_blocking=True)   # async CPU→GPU (works with pin_memory)
    batch_y = batch_y.to(device, non_blocking=True)
    # non_blocking=True lets the CPU return immediately and overlap
    # the DMA transfer with other CPU work (next batch prefetch).

    optimizer.zero_grad()           # clear gradients from previous batch
    logits = model(batch_x)         # forward pass — builds computation graph
    loss   = criterion(logits, batch_y)
    loss.backward()                 # backprop — fills p.grad for every weight
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()                # W <- W - lr * Adam(grad)

    return loss.item()


# ================================================================
# STEP 4 — Evaluation
# ================================================================
def evaluate(model, dataloader, criterion, device):
    """
    Runs the model on val data with no gradient tracking.
    torch.no_grad() skips building the computation graph — saves memory
    since we won't call backward() here.
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x   = batch_x.to(device, non_blocking=True)
            batch_y   = batch_y.to(device, non_blocking=True)
            logits    = model(batch_x)
            loss      = criterion(logits, batch_y)
            total_loss += loss.item()
            predicted  = torch.argmax(logits, dim=1)
            correct   += (predicted == batch_y).sum().item()
            total     += batch_y.size(0)

    return total_loss / len(dataloader), correct / total


# ================================================================
# STEP 5 — Training loop with early stopping
# ================================================================
def train(model, train_loader, val_loader, criterion, optimizer,
          n_epochs=50, device=torch.device("cuda"), patience=7):
    """
    Trains for up to n_epochs. Stops early if val loss does not
    improve for `patience` consecutive epochs, then restores the
    best weights seen during the entire run.

    Why patience=7?
      Too low (e.g. 2) and you stop before the model has a chance
      to climb out of a temporary plateau.
      Too high (e.g. 20) and you waste time after the model has
      clearly started overfitting.
      7 is a reasonable middle ground for a dataset this size.

    Timing:
      We track wall-clock seconds per epoch so you can see the
      concrete difference between fast and slow data loading.
    """
    history = {
        'train_loss': [], 'val_loss': [],
        'val_acc':    [], 'epoch_time': []
    }

    best_val_loss     = float('inf')
    epochs_no_improve = 0
    best_weights_path = 'best_weights.pt'

    for epoch in range(n_epochs):
        t0 = time.time()

        # Training pass
        model.train()
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            loss = train_one_batch(
                model, batch_x, batch_y, criterion, optimizer, device)
            epoch_losses.append(loss)

        avg_train_loss = float(np.mean(epoch_losses))

        # Validation pass
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - t0

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)

        print(f"Epoch {epoch+1:3d}/{n_epochs}  |  "
              f"Train: {avg_train_loss:.4f}  |  "
              f"Val: {val_loss:.4f}  |  "
              f"Acc: {val_acc*100:.1f}%  |  "
              f"Time: {epoch_time:.1f}s")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_weights_path)
            # ^ snapshot the best weights so far
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered (patience={patience}).")
                print(f"Best val loss: {best_val_loss:.4f}  — restoring those weights.")
                model.load_state_dict(
                    torch.load(best_weights_path, map_location=device))
                break

    return history


# ================================================================
# STEP 6 — Plot
# ================================================================
def plot_training(history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history['train_loss'], label='Train', color='steelblue')
    axes[0].plot(history['val_loss'],   label='Val',   color='coral')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss'); axes[0].legend()

    axes[1].plot([a * 100 for a in history['val_acc']], color='teal')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Val accuracy')

    axes[2].plot(history['epoch_time'], color='purple')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Seconds')
    axes[2].set_title('Time per epoch')

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=120)
    plt.close()
    print("Saved: training_curves.png")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    BASE = r"E:\programming\machine learning\projects\DL_project"

    # NSynthDataset auto-detects which mode to use:
    #   ⚡ memmap (fast)  — if specs.npy + labels.npy exist in the cache dir
    #                        run build_cache.py once to generate these
    #   📄 files (slower) — fallback to individual .npy files
    TRAIN_ROOT  = rf"{BASE}\nsynth-train.jsonwav\nsynth-train"   # for examples.json (files mode only)
    TRAIN_CACHE = rf"{BASE}\nsynth-train.jsonwav\outputs"        # merged cache (preferred)

    VALID_ROOT  = rf"{BASE}\nsynth-valid.jsonwav\nsynth-valid"
    VALID_CACHE = rf"{BASE}\nsynth-valid.jsonwav\outputs"        # merged cache (preferred)

    # Fallback: if outputs/ doesn't have specs.npy yet, point to individual .npy files:
    import os as _os
    if not _os.path.exists(rf"{TRAIN_CACHE}\specs.npy"):
        TRAIN_CACHE = rf"{BASE}\nsynth-train.jsonwav\cache"
        print("[WARN] Merged cache not found — using individual .npy files (slower).")
        print("       Run build_cache.py to generate the fast merged cache.")
    if not _os.path.exists(rf"{VALID_CACHE}\specs.npy"):
        VALID_CACHE = rf"{BASE}\nsynth-valid.jsonwav\cache"

    N_MELS    = 64
    N_FRAMES  = 345
    N_CLASSES = 12

    # Datasets
    # max_samples=None uses the full split.
    # Set max_samples=5000 / 1000 for a quick sanity-check before the full run.
    train_dataset = NSynthDataset(
        TRAIN_ROOT, cache_dir=TRAIN_CACHE,
        n_mels=N_MELS, n_frames=N_FRAMES,
        max_samples=None, augment=True)

    val_dataset = NSynthDataset(
        VALID_ROOT, cache_dir=VALID_CACHE,
        n_mels=N_MELS, n_frames=N_FRAMES,
        max_samples=None, augment=False)

    # DataLoaders — GPU-optimised settings
    pin = (device.type == "cuda")   # pin_memory only useful on GPU
    train_loader = DataLoader(
        train_dataset, batch_size=256,
        shuffle=True,  num_workers=4,
        pin_memory=pin, persistent_workers=True)
    val_loader = DataLoader(
        val_dataset, batch_size=256,
        shuffle=False, num_workers=4,
        pin_memory=pin, persistent_workers=True)
    # pin_memory=True keeps tensors in page-locked RAM so the GPU DMA can
    # read them directly — avoids an extra CPU copy on every .to(device).
    # persistent_workers=True keeps worker processes alive between epochs.

    # Model
    model = NoteClassifierCNN(
        n_mels=N_MELS, n_frames=N_FRAMES, n_classes=N_CLASSES)
    model = model.to(device)

    criterion, optimizer = setup_loss_and_optimizer(model, learning_rate=3e-4)

    # Train
    print("\nStarting training...\n")
    print("Testing data loader...")
    batch_x, batch_y = next(iter(train_loader))
    print(f"First batch loaded OK — shape: {batch_x.shape}, labels: {batch_y[:8]}")
    history = train(
        model, train_loader, val_loader,
        criterion, optimizer,
        n_epochs=50, device=device, patience=7)

    plot_training(history)

    torch.save(model.state_dict(), 'note_classifier_weights.pt')
    print("\nWeights saved: note_classifier_weights.pt")

