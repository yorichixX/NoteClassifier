import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from model import NoteClassifierCNN

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# STEP 1 -- Load trained model weights

def load_model(weights_path, n_mels=64, n_frames=345, n_classes=12):

    model = NoteClassifierCNN(n_mels=n_mels, n_frames=n_frames, n_classes=n_classes)

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print(f"Loaded weights: {weights_path}")
    else:
        print(f"WARNING: weights file not found at '{weights_path}'.")
        print("Using random weights -- train the model in Module 4 first.")

    model.eval()

    return model

# STEP 2 -- WAV file -> mel spectrogram tensor

def audio_to_tensor(wav_path, sr=22050, n_mels=64, n_fft=1024,
                    hop_length=256, n_frames=345):

    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    # librosa resamples NSynth's native 16 kHz -> 22050 Hz automatically.

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=80, fmax=8000
    )
    S_db = librosa.power_to_db(S, ref=np.max) 

    # Crop or pad to exactly n_frames columns
    if S_db.shape[1] >= n_frames:
        S_db = S_db[:, :n_frames]
    else:
        pad = n_frames - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad)),
                      mode='constant', constant_values=S_db.min())

    # Normalise to [0, 1]
    s_min, s_max = S_db.min(), S_db.max()
    if s_max > s_min:
        S_db = (S_db - s_min) / (s_max - s_min)

    tensor = torch.FloatTensor(S_db).unsqueeze(0).unsqueeze(0)
    # (n_mels, n_frames) -> (1, n_mels, n_frames) -> (1, 1, n_mels, n_frames)

    return tensor, S_db

# STEP 3 -- Predict and show confidence scores

def predict_note(model, input_tensor):
    """
    Forward pass -> softmax -> top-3 predictions with bar chart in terminal.
    """
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)              # (1, n_classes)
        probs  = F.softmax(logits, dim=1).squeeze()  # (n_classes,)

    top3_probs, top3_idx = torch.topk(probs, k=3)

    print("\n-- Prediction " + "-" * 30)
    print(f"  Note: {NOTE_NAMES[top3_idx[0]]}   ({top3_probs[0]*100:.1f}% confidence)")
    print("\n  Top 3:")
    for i in range(3):
        bar = chr(9608) * int(top3_probs[i].item() * 30)
        print(f"    {NOTE_NAMES[top3_idx[i]]:<4s}: {bar} {top3_probs[i]*100:.1f}%")
    print("-" * 44)

    return probs.numpy(), top3_idx[0].item()

# STEP 4 -- Layer activation maps

def visualize_activations(model, input_tensor, S_db, save_path='module_05_activations.png'):

    captured = {}

    def hook_fn(name):
        def fn(module, inp, out):
            captured[name] = out.detach()
        return fn

    h1 = model.conv1.register_forward_hook(hook_fn('conv1'))
    h2 = model.conv2.register_forward_hook(hook_fn('conv2'))

    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    h1.remove()
    h2.remove()

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(3, 8, hspace=0.45, wspace=0.3)

    # Row 0: input spectrogram (spans first 4 columns)
    ax0 = fig.add_subplot(gs[0, :4])
    ax0.imshow(S_db, origin='lower', aspect='auto', cmap='magma')
    ax0.set_title('Input mel spectrogram', fontsize=11)
    ax0.set_xlabel('Time frame')
    ax0.set_ylabel('Mel bin')

    # Row 1: first 8 feature maps from conv1  (shape: 32 x 32 x 172)
    act1 = captured['conv1'].squeeze(0)    # (32, H, W)
    for i in range(8):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(act1[i].numpy(), origin='lower', aspect='auto', cmap='viridis')
        ax.set_title(f'L1-f{i}', fontsize=8)
        ax.axis('off')

    # Row 2: first 8 feature maps from conv2  (shape: 64 x 16 x 86)
    act2 = captured['conv2'].squeeze(0)    # (64, H, W)
    for i in range(8):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(act2[i].numpy(), origin='lower', aspect='auto', cmap='viridis')
        ax.set_title(f'L2-f{i}', fontsize=8)
        ax.axis('off')

    fig.suptitle('What each conv layer detects (feature maps)', fontsize=12)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")



# STEP 5 -- Saliency map

def compute_saliency(model, input_tensor, target_class):

    model.eval()
    inp = input_tensor.clone().requires_grad_(True)
    # requires_grad=True tells autograd to track operations on this tensor

    logits = model(inp)                  # forward pass, graph is built
    score  = logits[0, target_class]     # scalar output for target class

    model.zero_grad()
    score.backward()                     # dScore/dInp computed via chain rule

    saliency = inp.grad.data.abs().squeeze().numpy()
    # .abs() because both positive and negative gradients matter
    # .squeeze() removes the batch and channel dims: (1,1,H,W) -> (H,W)
    return saliency


def visualize_saliency(S_db, saliency, predicted_note,
                       save_path='module_05_saliency.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(S_db, origin='lower', aspect='auto', cmap='magma')
    axes[0].set_title('Input mel spectrogram')
    axes[0].set_xlabel('Time frame')
    axes[0].set_ylabel('Mel bin')

    im = axes[1].imshow(saliency, origin='lower', aspect='auto', cmap='hot')
    axes[1].set_title(
        f'Saliency map -- predicted: {predicted_note}\n'
        f'(bright = model relies on this region)'
    )
    axes[1].set_xlabel('Time frame')
    axes[1].set_ylabel('Mel bin')
    plt.colorbar(im, ax=axes[1])

    plt.suptitle('dPrediction/dInput  (same chain rule as backprop, applied to input)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":

    # Point this at any NSynth .wav file you want to demo
    WAV_PATH = r"E:\programming\machine learning\projects\DL_project\nsynth-test.jsonwav\nsynth-test\audio\vocal_synthetic_003-096-100.wav"
    # These must match exactly what was used in Module 4 training
    N_MELS    = 64
    N_FRAMES  = 345
    N_CLASSES = 12

    # Load model (train it in Module 4 first to get meaningful results)
    model = load_model('note_classifier_weights.pt',
                       n_mels=N_MELS, n_frames=N_FRAMES, n_classes=N_CLASSES)

    # Parse ground-truth note from filename so we can compare
    import os
    basename = os.path.basename(WAV_PATH)
    try:
        midi_pitch   = int(basename.split('-')[-2])
        true_note    = NOTE_NAMES[midi_pitch % 12]
        print(f"Ground truth from filename: {true_note}  (MIDI {midi_pitch})")
    except (IndexError, ValueError):
        true_note = "unknown"

    # WAV -> tensor
    input_tensor, S_db = audio_to_tensor(
        WAV_PATH, sr=22050, n_mels=N_MELS, n_frames=N_FRAMES
    )

    # Predict
    probs, pred_idx = predict_note(model, input_tensor)
    predicted_note  = NOTE_NAMES[pred_idx]

    correct = "(CORRECT)" if predicted_note == true_note else f"(true: {true_note})"
    print(f"  Prediction: {predicted_note}  {correct}")

    # Activation maps
    print("\nGenerating layer activation maps...")
    visualize_activations(model, input_tensor, S_db)

    # Saliency map
    print("Computing saliency map...")
    saliency = compute_saliency(model, input_tensor, pred_idx)
    visualize_saliency(S_db, saliency, predicted_note)

    print("\n-- Output files --")
    print("  module_05_activations.png  -- what each filter detected")
    print("  module_05_saliency.png     -- which spectrogram regions drove prediction")