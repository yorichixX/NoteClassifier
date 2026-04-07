"""
Gradient Flow Inspector
========================
Loads the trained model, runs a forward + backward pass on a small batch,
then prints and visualises the gradients at every layer.

This helps you understand:
  1. Are gradients flowing through all layers?  (vanishing gradient problem)
  2. Are any gradients exploding?                (exploding gradient problem)
  3. Which neurons fire the strongest gradients? (what the model cares about)

Usage:
    python gradient_inspector.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from model import NoteClassifierCNN


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# ================================================================
# STEP 1 — Load model + a small batch of real data
# ================================================================

def load_model_and_batch(weights_path, cache_dir,
                         n_mels=64, n_frames=345, n_classes=12,
                         batch_size=16):
    """
    Loads the trained model and grabs a small batch from the merged cache.
    Returns (model, batch_x, batch_y) — all on CPU for inspection.
    """
    model = NoteClassifierCNN(n_mels=n_mels, n_frames=n_frames, n_classes=n_classes)

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print(f"Loaded weights: {weights_path}")
    else:
        print(f"WARNING: {weights_path} not found — using random weights.")

    # Load a small batch from the merged cache
    specs  = np.load(os.path.join(cache_dir, 'specs.npy'), mmap_mode='r')
    labels = np.load(os.path.join(cache_dir, 'labels.npy'))

    # Grab the first batch_size samples
    batch_specs  = np.array(specs[:batch_size], dtype=np.float32)
    batch_labels = labels[:batch_size]

    # (B, n_mels, n_frames) → (B, 1, n_mels, n_frames)
    batch_x = torch.FloatTensor(batch_specs).unsqueeze(1)
    batch_y = torch.LongTensor(batch_labels)

    print(f"Batch: {batch_x.shape}, Labels: {batch_y.tolist()}")
    print(f"       → Notes: {[NOTE_NAMES[l] for l in batch_y.tolist()]}")
    return model, batch_x, batch_y


# ================================================================
# STEP 2 — Forward + backward pass, capture all gradients
# ================================================================

def compute_gradients(model, batch_x, batch_y):
    """
    Runs one forward pass and one backward pass.
    After this, every model.parameter().grad is populated.

    Returns the loss value.
    """
    model.train()                            # enable dropout + BN training mode
    model.zero_grad()                        # clear any old gradients

    logits = model(batch_x)                  # forward: input → logits
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, batch_y)        # compute loss

    loss.backward()                          # backward: dLoss/dW for every weight

    print(f"\nLoss: {loss.item():.4f}")
    return loss.item()


# ================================================================
# STEP 3 — Print gradient statistics per layer
# ================================================================

def print_gradient_table(model):
    """
    For each named parameter, prints:
      - shape (what the tensor looks like)
      - mean |grad|  (average gradient magnitude)
      - max  |grad|  (largest gradient)
      - % zero       (fraction of exactly-zero gradients)

    What to look for:
      - mean |grad| decreasing dramatically from fc2 → conv1 = vanishing gradients
      - mean |grad| > 1.0 anywhere = potentially exploding
      - % zero > 50% in conv layers = many dead filters (ReLU dying)
    """
    print("\n" + "=" * 80)
    print(f"{'Layer':<25s} {'Shape':<22s} {'Mean |grad|':>12s} "
          f"{'Max |grad|':>12s} {'% Zero':>8s}")
    print("=" * 80)

    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad.data
            abs_g = g.abs()
            mean_g = abs_g.mean().item()
            max_g  = abs_g.max().item()
            pct_zero = (g == 0).float().mean().item() * 100

            # Colour hint: flag potential issues
            flag = ""
            if mean_g < 1e-7:
                flag = "  ⚠️  VERY SMALL (vanishing?)"
            elif mean_g > 1.0:
                flag = "  ⚠️  LARGE (exploding?)"
            elif pct_zero > 50:
                flag = "  ⚠️  MANY ZEROS (dead filters?)"

            print(f"{name:<25s} {str(list(param.shape)):<22s} "
                  f"{mean_g:>12.6e} {max_g:>12.6e} {pct_zero:>7.1f}%{flag}")
        else:
            print(f"{name:<25s} {str(list(param.shape)):<22s} {'NO GRAD':>12s}")

    print("=" * 80)


# ================================================================
# STEP 4 — Visualise gradient flow (bar chart)
# ================================================================

def plot_gradient_flow(model, save_path='gradient_flow.png'):
    """
    Classic gradient flow plot: one bar per layer showing the average
    absolute gradient magnitude.

    Healthy model:  bars are roughly similar height across all layers.
    Vanishing:      bars shrink dramatically from right (output) to left (input).
    Exploding:      one or more bars are orders of magnitude taller than others.
    """
    layers = []
    avg_grads = []
    max_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            layers.append(name)
            avg_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: linear scale
    x = range(len(layers))
    axes[0].bar(x, max_grads, alpha=0.3, lw=1, color='royalblue', label='Max |grad|')
    axes[0].bar(x, avg_grads, alpha=0.8, lw=1, color='coral', label='Mean |grad|')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Gradient magnitude')
    axes[0].set_title('Gradient flow (linear scale)')
    axes[0].legend()
    axes[0].axhline(y=0, color='k', linewidth=0.5)

    # Right: log scale (reveals vanishing gradients)
    axes[1].bar(x, max_grads, alpha=0.3, lw=1, color='royalblue', label='Max |grad|')
    axes[1].bar(x, avg_grads, alpha=0.8, lw=1, color='coral', label='Mean |grad|')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('Gradient magnitude (log)')
    axes[1].set_title('Gradient flow (log scale — reveals vanishing)')
    axes[1].set_yscale('log')
    axes[1].legend()

    plt.suptitle('Gradient Flow Across Layers', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


# ================================================================
# STEP 5 — Inspect individual neuron gradients
# ================================================================

def inspect_neurons(model):
    """
    Drills into specific layers to show per-filter and per-neuron gradients.
    Bars are scaled relative to the max value in each group (max = 40 chars).
    """
    MAX_BAR = 40   # max bar width in characters

    def bar(value, max_value):
        """Return a bar string scaled so max_value maps to MAX_BAR chars."""
        if max_value == 0:
            return ""
        width = int(value / max_value * MAX_BAR)
        return "█" * width

    print("\n" + "=" * 70)
    print("Per-filter / per-neuron gradient magnitudes  (bar = relative scale)")
    print("=" * 70)

    # --- Conv1: per-filter (32 filters) ---
    conv1_grad = model.conv1.weight.grad   # (32, 1, 3, 3)
    if conv1_grad is not None:
        per_filter = conv1_grad.abs().mean(dim=[1, 2, 3])   # (32,)
        max_g = per_filter.max().item()
        print(f"\nconv1 — 32 filters  [max = {max_g:.3e}]")
        for i, g in enumerate(per_filter):
            print(f"  Filter {i:2d}: {g.item():.3e}  {bar(g.item(), max_g)}")

    # --- Conv2: per-filter, top/bottom 10 ---
    conv2_grad = model.conv2.weight.grad   # (64, 32, 3, 3)
    if conv2_grad is not None:
        per_filter = conv2_grad.abs().mean(dim=[1, 2, 3])   # (64,)
        max_g = per_filter.max().item()
        sorted_idx = per_filter.argsort(descending=True)
        print(f"\nconv2 — 64 filters, top 10 & bottom 10  [max = {max_g:.3e}]")
        for label, indices in [("STRONGEST", sorted_idx[:10]),
                               ("WEAKEST  ", sorted_idx[-10:])]:
            print(f"  {label}:")
            for i in indices:
                g = per_filter[i].item()
                print(f"    Filter {i:2d}: {g:.3e}  {bar(g, max_g)}")

    # --- FC1: per-neuron, top/bottom 5 ---
    fc1_grad = model.fc1.weight.grad   # (128, 88064)
    if fc1_grad is not None:
        per_neuron = fc1_grad.abs().mean(dim=1)   # (128,)
        max_g = per_neuron.max().item()
        sorted_idx = per_neuron.argsort(descending=True)
        print(f"\nfc1 — 128 neurons, top 5 & bottom 5  [max = {max_g:.3e}]")
        for label, indices in [("STRONGEST", sorted_idx[:5]),
                               ("WEAKEST  ", sorted_idx[-5:])]:
            print(f"  {label}:")
            for i in indices:
                g = per_neuron[i].item()
                print(f"    Neuron {i:3d}: {g:.3e}  {bar(g, max_g)}")

    # --- FC2: per output class ---
    fc2_grad = model.fc2.weight.grad   # (12, 128)
    if fc2_grad is not None:
        per_class = fc2_grad.abs().mean(dim=1)   # (12,)
        max_g = per_class.max().item()
        print(f"\nfc2 — 12 output neurons (one per note)  [max = {max_g:.3e}]")
        for i, g in enumerate(per_class):
            print(f"  {NOTE_NAMES[i]:<3s} (class {i:2d}): {g.item():.3e}  "
                  f"{bar(g.item(), max_g)}")

    print("=" * 70)


# ================================================================
# STEP 6 — Gradient distribution histogram
# ================================================================

def plot_gradient_histograms(model, save_path='gradient_histograms.png'):
    """
    Histogram of gradient values for each major layer.
    Healthy: roughly bell-shaped, centered near 0, tails don't extend too far.
    Vanishing: extremely narrow spike at 0.
    Exploding: long tails reaching large values.
    """
    layers_to_plot = [
        ('conv1.weight', model.conv1.weight),
        ('conv2.weight', model.conv2.weight),
        ('fc1.weight',   model.fc1.weight),
        ('fc2.weight',   model.fc2.weight),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (name, param) in zip(axes, layers_to_plot):
        if param.grad is not None:
            grads = param.grad.data.cpu().numpy().flatten()
            ax.hist(grads, bins=100, alpha=0.8, color='steelblue', edgecolor='none')
            ax.set_title(f'{name}\nmean={grads.mean():.2e}, std={grads.std():.2e}',
                         fontsize=10)
            ax.set_xlabel('Gradient value')
            ax.set_ylabel('Count')
            ax.axvline(x=0, color='red', linewidth=0.8, linestyle='--')
        else:
            ax.set_title(f'{name}\nNO GRADIENT')

    plt.suptitle('Gradient Value Distributions per Layer', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    WEIGHTS_PATH = "note_classifier_weights.pt"
    CACHE_DIR    = r"E:\programming\machine learning\projects\DL_project\nsynth-test.jsonwav\outputs"
    # Fallback if test outputs don't exist
    if not os.path.exists(os.path.join(CACHE_DIR, 'specs.npy')):
        CACHE_DIR = r"E:\programming\machine learning\projects\DL_project\nsynth-valid.jsonwav\outputs"

    N_MELS    = 64
    N_FRAMES  = 345
    N_CLASSES = 12

    # ------ Load ------
    model, batch_x, batch_y = load_model_and_batch(
        WEIGHTS_PATH, CACHE_DIR,
        n_mels=N_MELS, n_frames=N_FRAMES, n_classes=N_CLASSES,
        batch_size=16)

    # ------ Forward + backward ------
    compute_gradients(model, batch_x, batch_y)

    # ------ Inspect ------
    print_gradient_table(model)
    inspect_neurons(model)

    # ------ Visualise ------
    plot_gradient_flow(model)
    plot_gradient_histograms(model)

    print("\n-- Output files --")
    print("  gradient_flow.png        — bar chart: gradient magnitude per layer")
    print("  gradient_histograms.png  — distribution of gradient values per layer")
    print("\nWhat to look for:")
    print("  • Healthy: bars similar height across layers, bell-shaped histograms")
    print("  • Vanishing: early layers (conv1) have drastically smaller gradients")
    print("  • Exploding: any layer has gradients >>1")
    print("  • Dead neurons: many zero gradients in a layer (ReLU dying)")
