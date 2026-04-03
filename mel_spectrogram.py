import numpy as np 
import librosa 
import librosa.display
import matplotlib.pyplot as plt

#doing fft on each frame

def compute_fft(windowed_frames):
    fft_frames= np.fft.rfft(windowed_frames, axis=1)
    power_spectrum= np.abs(fft_frames)**2

    print(f"FFT frames shape: {fft_frames.shape}  (complex)")
    print(f"Power spectrum shape: {power_spectrum.shape}  (real)")
    return power_spectrum

# Mel filterbank

def hz_to_mel(hz):

    return 2585* np.log10(1+ hz/700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1) 

def build_mel_filterbank(n_fft_bins, sr, n_mels=64, fmin=80, fmax=8000):
    mel_min=hz_to_mel(fmin)
    mel_max=hz_to_mel(fmax)
    mel_points= np.linspace(mel_min,mel_max,n_mels+2)

    hz_points= mel_to_hz(mel_points)
    bin_points= np.floor(hz_points*n_fft_bins*2/sr).astype(int)
    bin_points= np.clip(bin_points,0,n_fft_bins-1)

    filterbank= np.zeros((n_mels,n_fft_bins))
    for m in range(1,n_mels+1):
        left= bin_points[m-1]
        center= bin_points[m]
        right= bin_points[m+1]

        for k in range(left,center):
            if center != left:
                filterbank[m-1,k]=(k-left)/(center-left)

        for k in range(center,right):
            if right!=center:
                filterbank[m-1,k]=(right-k)/(right-center)

    return filterbank

#apply filterbank+ log compression

def compute_mel_spectrogram_librosa(y, sr, n_mels=64,n_fft=1024,hop_length=256):
    S= librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=80,
        fmax=8000       
    )

    S_db= librosa.power_to_db(S, ref=np.max)
    print(f"librosa mel spectrogram shape: {S_db.shape}")
    return S_db

#visualization

def visualize_spectrogram(mel_spec, sr, hop_length, title="Mel spectrogram"):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spec, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel',
        fmin=80, fmax=8000,
        cmap='magma'   # warm = high energy, cool = low energy
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('module_02_spectrogram.png', dpi=120)
    plt.close()
    print("Saved: module_02_spectrogram.png")

if __name__ == "__main__":
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
 
    # A4 = 440 Hz fundamental + harmonics (more realistic than pure sine)
    y_test = "E:\\programming\\machine learning\\projects\\DL_project\\nsynth-test.jsonwav\\nsynth-test\\audio\\bass_electronic_018-022-100.wav"
    y_test = y_test / np.max(np.abs(y_test))
 
    n_fft = 1024
    hop_length = 256
 
    mel_spec = compute_mel_spectrogram_librosa(y_test, sr, n_fft=n_fft, hop_length=hop_length)
    visualize_spectrogram(mel_spec, sr, hop_length, title="A4 (440 Hz) — mel spectrogram")
 
    print("\nOutput of this module:")
    print(f"  mel_spec: shape {mel_spec.shape}  (n_mels × n_frames)")