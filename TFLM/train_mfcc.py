# train_esp32_mfcc.py
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf

# ---------- CONFIG (match your ESP32) ----------
SAMPLE_RATE = 16000
FRAME_LENGTH = 512             # AUDIO_BUFFER_SIZE on ESP32
HOP_LENGTH = 512               # non-overlapping frames to match I2S.read per call
NUM_MEL_FILTERS = 26           # NUM_MEL_FILTERS on ESP32
NUM_MFCC = 13                  # NUM_MFCC on ESP32
N_FFT = FRAME_LENGTH
EPS = 1e-6
DATA_PATH = "TFLM/data"        # change if needed
LABELS = ['shalom', 'unknown', 'noise', 'lehit']  # match your labels
# ------------------------------------------------

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)

def create_mel_filterbank(sample_rate, n_fft, n_mels):
    # This mirrors createMelFilterbank() in your ESP32 code
    low_mel = hz_to_mel(0.0)
    high_mel = hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)  # NUM_MEL_FILTERS + 2
    freq_points = mel_to_hz(mel_points)

    fft_size = n_fft
    # We'll compute triangle weights for bins k = 0..fft_size/2
    n_bins = fft_size // 2 + 1
    mel_fb = np.zeros((n_mels, n_bins), dtype=np.float32)

    for m in range(1, n_mels + 1):
        f_m_minus = freq_points[m - 1]
        f_m = freq_points[m]
        f_m_plus = freq_points[m + 1]

        for k in range(n_bins):
            # freq computed the same way your ESP32 code does:
            freq = (sample_rate / 2.0) * k / (fft_size / 2.0)
            weight = 0.0
            if freq >= f_m_minus and freq <= f_m:
                denom = (f_m - f_m_minus)
                if denom != 0:
                    weight = (freq - f_m_minus) / denom
            elif freq > f_m and freq <= f_m_plus:
                denom = (f_m_plus - f_m)
                if denom != 0:
                    weight = (f_m_plus - freq) / denom
            mel_fb[m - 1, k] = weight
    return mel_fb

def compute_dct_type2(input_vec, output_size):
    # This reproduces computeDCT() from ESP32: DCT-II WITHOUT normalization
    input_size = len(input_vec)
    out = np.zeros(output_size, dtype=np.float32)
    for n in range(output_size):
        # sum_{m=0..M-1} input[m] * cos(pi * n * (m + 0.5) / M)
        cos_terms = np.cos(np.pi * n * (np.arange(input_size) + 0.5) / (input_size))
        out[n] = np.sum(input_vec * cos_terms)
    return out

# Precompute Hamming window and mel-filterbank to match ESP32
hamming_window = np.hamming(FRAME_LENGTH).astype(np.float32)
mel_filterbank = create_mel_filterbank(SAMPLE_RATE, N_FFT, NUM_MEL_FILTERS)

def compute_mfcc_esp32_style(frame):
    """
    frame: 1-D array of length FRAME_LENGTH, float in range [-1,1] or PCM floats
    returns: 1-D array length NUM_MFCC
    """
    # Apply Hamming window
    windowed = frame * hamming_window

    # Compute FFT magnitude like in ArduinoFFT -> they use real FFT magnitude of windowed samples
    # Use numpy rfft which returns N/2+1 bins. Use absolute (magnitude).
    spectrum = np.abs(np.fft.rfft(windowed, n=N_FFT)).astype(np.float32)

    # Apply mel filterbank (dot product)
    mel_energies = np.dot(mel_filterbank, spectrum)  # shape (NUM_MEL_FILTERS,)
    # log energy as done on ESP32
    mel_energies = np.log(mel_energies + EPS)

    # DCT (no normalization)
    mfcc = compute_dct_type2(mel_energies, NUM_MFCC)
    return mfcc

def frames_from_wav(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    """Yield non-overlapping frames (matching I2S.read behavior)."""
    total = len(audio)
    if total < frame_length:
        # pad to at least one frame
        padded = np.pad(audio, (0, frame_length - total), mode='constant')
        yield padded[:frame_length]
        return
    for start in range(0, total - frame_length + 1, hop_length):
        yield audio[start:start + frame_length]
    # If last tail exists and not enough for full frame, you can pad (optional):
    tail = total - ( (total - frame_length) // hop_length ) * hop_length
    # the above preserves only full frames; optional: pad last partial frame if needed
    # (we will not include partial tail to match device's continuous reads)

# --------- Load dataset and compute MFCCs ----------
X = []
y = []

print("Scanning dataset folders:", DATA_PATH)
for lbl in LABELS:
    folder = os.path.join(DATA_PATH, lbl)
    if not os.path.isdir(folder):
        print("Warning: folder missing:", folder)
        continue
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith('.wav'):
            continue
        path = os.path.join(folder, fname)
        audio, sr = librosa.load(path, sr=SAMPLE_RATE)
        # ensure float32 in [-1, 1]
        audio = audio.astype(np.float32)
        # break file into non-overlapping 512-sample frames
        for frame in frames_from_wav(audio, FRAME_LENGTH, HOP_LENGTH):
            # scale: ESP32 reads int16 and divides by 32768.0 -> audio range ~ [-1,1]
            # librosa already provides floats in [-1,1], so frame is consistent.
            mfcc = compute_mfcc_esp32_style(frame)
            X.append(mfcc)
            y.append(LABELS.index(lbl))

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
print("Extracted frames:", X.shape, "labels:", np.unique(y, return_counts=True))

# Optional: balance classes or reduce if dataset huge.
# For small datasets, you might want to keep all frames.

# --------- Train / Test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# --------- Build and train a small model ----------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(NUM_MFCC,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(LABELS), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Simple callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,
    batch_size=32,
    callbacks=callbacks
)

loss, acc = model.evaluate(X_test, y_test)
print("Final test accuracy:", acc)

# --------- Save float TFLite model (no quantization) ----------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
out_path = "TFLM/wakeword_model_esp32mfcc.tflite"
os.makedirs("TFLM", exist_ok=True)
with open(out_path, "wb") as f:
    f.write(tflite_model)
print("Saved TFLite float model to:", out_path)

print("\nNEXT: run:\n  d -i TFLM/wakeword_model_esp32mfccxx.tflite > ESP32/tflm/model_data.h\nthen rebuild & flash your ESP32 sketch.")
