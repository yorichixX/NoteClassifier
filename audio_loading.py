import numpy as np
import librosa          # pip install librosa
import matplotlib.pyplot as plt
 
# STEP 1 — Load audio
def load_audio(filepath, target_sr=22050):

    y, sr = librosa.load(filepath, sr=target_sr, mono=True)

    print(f"Loaded: {len(y)} samples at {sr} Hz = {len(y)/sr:.2f} seconds")
    return y, sr
 
def frame_signal(y, sr, frame_duration_ms=25, hop_duration_ms=12.5):

    frame_length = int(sr * frame_duration_ms / 1000)   # 551 samples
    hop_length   = int(sr * hop_duration_ms  / 1000)    # 275 samples
 
    num_frames = 1 + (len(y) - frame_length) // hop_length
    frames = np.array([
        y[i * hop_length : i * hop_length + frame_length]
        for i in range(num_frames)
    ])
 
    print(f"Frame length: {frame_length} samples ({frame_duration_ms}ms)")
    print(f"Hop length:   {hop_length} samples ({hop_duration_ms}ms)")
    print(f"Number of frames: {num_frames}")
    print(f"Frames shape: {frames.shape}  (num_frames × frame_length)")
    return frames, frame_length, hop_length
 
 
#applying Hann window
def apply_window(frames, frame_length):

    window = np.hanning(frame_length) 
 
    windowed_frames = frames * window  
 
    print(f"Windowed frames shape: {windowed_frames.shape}")
    return windowed_frames
 
 
#visualization
def visualize_framing(y, sr, windowed_frames, frame_length, hop_length):
    """Plot the raw waveform and the first few frames to see what framing does."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
 
    # Top: full waveform with frame boundaries
    time = np.arange(len(y)) / sr
    axes[0].plot(time, y, color='steelblue', linewidth=0.5, alpha=0.8)
    # Draw first 5 frame boundaries
    for i in range(5):
        start = i * hop_length / sr
        end   = (i * hop_length + frame_length) / sr
        axes[0].axvspan(start, end, alpha=0.15, color='orange' if i%2==0 else 'red')
    axes[0].set_title('Waveform with first 5 overlapping frames highlighted')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
 
    # Bottom: first 3 individual windowed frames
    for i in range(min(3, len(windowed_frames))):
        axes[1].plot(windowed_frames[i], label=f'Frame {i}', alpha=0.8)
    axes[1].set_title('First 3 windowed frames (note tapered edges from Hann window)')
    axes[1].set_xlabel('Sample index within frame')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
 
    plt.tight_layout()
    plt.savefig('module_01_framing.png', dpi=120)
    plt.close()
    print("Saved: module_01_framing.png")
 
 
if __name__ == "__main__":
 
    WAV_PATH = r"Give_a_valid_path"

    y, sr = load_audio(WAV_PATH, target_sr=22050)


    NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    import os
    basename = os.path.basename(WAV_PATH)
    try:
        midi_pitch = int(basename.split('-')[-2])
        note_name  = NOTE_NAMES[midi_pitch % 12]
        octave     = midi_pitch // 12 - 1
        print(f"Note from filename: {note_name}{octave}  (MIDI pitch {midi_pitch})")
    except (IndexError, ValueError):
        print(f"File: {basename}")
 
    frames, frame_length, hop_length = frame_signal(y, sr)
    windowed_frames = apply_window(frames, frame_length)
    visualize_framing(y, sr, windowed_frames, frame_length, hop_length)
 
    print("\nOutput of this module:")
    print(f"  y shape:{y.shape}(raw waveform samples)")
    print(f"  windowed_frames shape: {windowed_frames.shape}  (n_frames × frame_length)")