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
    Loads pre-computed .npy mel spectrograms from cache_dir.
    Falls back to on-the-fly librosa if cache_dir is None,
    but cached mode is strongly preferred for GPU utilisation.

    Label: pitch % 12  →  0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F,
                           6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B

    augment=True: apply random time shift, freq shift, and noise.
    Use augment=True for training only, never for validation.
    """

    def __init__(self, nsynth_root, cache_dir,
                 n_mels=64, n_frames=345,
                 max_samples=None, augment=False):

        self.cache_dir  = cache_dir
        self.n_mels     = n_mels
        self.n_frames   = n_frames
        self.augment    = augment

        # Read examples.json for labels
        json_path = os.path.join(nsynth_root, 'examples.json')
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        audio_dir    = os.path.join(nsynth_root, 'audio')
        self.samples = []
        for name, info in metadata.items():
            pitch_class = info['pitch'] % 12
            wav_path    = os.path.join(audio_dir, name + '.wav')
            npy_path    = os.path.join(cache_dir, name + '.npy')

            # Only include samples that are both in the JSON
            # AND have a cached .npy file on disk
            if os.path.exists(npy_path):
                self.samples.append((npy_path, pitch_class))
            elif os.path.exists(wav_path):
                # npy not cached yet — skip and warn once
                pass

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"NSynthDataset: {len(self.samples)} samples  "
              f"| augment={augment}  | cache={cache_dir}")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No cached .npy files found in {cache_dir}.\n"
                "Run precompute_spectrograms() first.")

    def __len__(self):
        return len(self.samples)

    def _augment(self, S):
        """
        Three cheap augmentations, each applied randomly every access.

        Time shift  — np.roll along axis=1 (time axis)
          Shifts the spectrogram left or right by up to 10 frames.
          Simulates the note starting slightly earlier or later in the clip.
          np.roll wraps content: stuff shifted off the right appears on
          the left. Fine here because NSynth notes are sustained for 4s —
          the wrap-around region is silence or tail, not a different note.

        Frequency shift — np.roll along axis=0 (mel bin axis)
          Shifts up or down by up to 2 mel bins (~3% of the 64-bin range).
          Simulates the same note played very slightly sharp or flat.
          2 bins is small enough that the pitch class is unambiguous.

        Gaussian noise — std = 0.01 (1% of the [0,1] normalised range)
          Prevents the model from memorising exact spectrogram pixel values.
          Forces it to rely on overall spectral shape instead.
        """
        S = np.roll(S, np.random.randint(-10, 10),  axis=1)  # time shift
        S = np.roll(S, np.random.randint(-2,  3),   axis=0)  # freq shift
        S = S + np.random.randn(*S.shape).astype(np.float32) * 0.01
        return np.clip(S, 0.0, 1.0)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        S = np.load(npy_path)           # float32, shape (n_mels, n_frames)

        if self.augment:
            S = self._augment(S)

        # unsqueeze(0): (n_mels, n_frames) -> (1, n_mels, n_frames)
        # The 1 is the channel dimension Conv2d expects (like grayscale).
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

    batch_x = batch_x.to(device)   # move input to GPU
    batch_y = batch_y.to(device)   # move labels to GPU
    # Model weights are already on GPU (model.to(device) in main).
    # Both operands of every matrix multiply must be on the same device.

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
            batch_x   = batch_x.to(device)
            batch_y   = batch_y.to(device)
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

    # Paths — update if your folder layout differs
    TRAIN_ROOT  = r"" #give a valid path
    TRAIN_CACHE = r"" #give a valid path
    VALID_CACHE = r"" #give a valid path

    N_MELS    = 64
    N_FRAMES  = 345
    N_CLASSES = 12

    # Datasets
    # max_samples=None uses the full split.
    # Set max_samples=5000 / 1000 first to do a quick sanity-check run
    # before committing to the full dataset.
    train_dataset = NSynthDataset(
        TRAIN_ROOT, cache_dir=TRAIN_CACHE,
        n_mels=N_MELS, n_frames=N_FRAMES,
        max_samples=None, augment=True)

    val_dataset = NSynthDataset(
        VALID_ROOT, cache_dir=VALID_CACHE,
        n_mels=N_MELS, n_frames=N_FRAMES,
        max_samples=None, augment=False)

    # DataLoaders
    # num_workers=0 required on Windows (no fork-based multiprocessing).
    # Increase batch_size if VRAM allows — larger batches mean fewer
    # steps per epoch and more stable gradient estimates.
    train_loader = DataLoader(train_dataset, batch_size=64,
                            shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=64,
                            shuffle=False, num_workers=0)
    # pin_memory=True speeds up CPU->GPU transfers by keeping batch tensors
    # in page-locked (pinned) memory, which the GPU DMA engine can read
    # directly without an extra copy. Only worth enabling when using GPU.

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