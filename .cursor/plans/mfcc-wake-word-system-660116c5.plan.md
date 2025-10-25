<!-- 660116c5-494a-45a0-a850-9c6f98f831cb a7330768-6367-4e72-967b-c29879fe4580 -->
# MFCC-Based Wake Word Detection System

## Overview

Create a parallel MFCC-based deployment system in `xiao-esp32s3-wake-word/mfcc/` with complete training pipeline and ESP32 deployment, using **custom MFCC implementation from scratch** matching Perplexity research parameters exactly.

## Structure

```
xiao-esp32s3-wake-word/
└── mfcc/
    ├── model_training/
    │   ├── config.json (MFCC parameters)
    │   ├── custom_mfcc.py (custom MFCC from scratch)
    │   ├── prepare_dataset.py
    │   ├── train_model.py
    │   ├── convert_to_tflite.py
    │   ├── generate_header.py
    │   ├── deploy_esp32.py
    │   ├── run_pipeline.py
    │   ├── requirements.txt
    │   └── models/
    ├── esp32s3_mfcc_standalone/
    │   ├── esp32s3_mfcc_standalone.ino
    │   ├── config.h
    │   ├── model_config.h
    │   ├── mic_config.h
    │   ├── audio_provider.h (copied from STFT version)
    │   ├── mfcc_extractor.h (NEW - custom C++ MFCC from scratch)
    │   ├── inference_engine.h (copied from STFT version)
    │   └── wake_word_model.h
    └── README.md
```

## Key Implementation Details

### 1. MFCC Parameters (from Perplexity research)

- **Sample Rate**: 16000 Hz
- **Pre-emphasis**: α = 0.97
- **Frame Length**: 25 ms (400 samples at 16 kHz)
- **Frame Step**: 10 ms (160 samples at 16 kHz) - 50% overlap
- **Window**: Hamming window
- **FFT Size**: 512 points
- **Mel Filters**: 26 filters (0-8000 Hz)
- **Frequency Range**: 0-8000 Hz
- **DCT Coefficients**: 26 (configurable, using 20-26 range)
- **Normalization**: CMVN (Cepstral Mean and Variance Normalization)

### 2. Model Input Shape

- Time frames: ~99 frames for 1-second audio (16000 samples, 160 step)
- MFCC coefficients: 26 (configurable 20-26)
- Resize to: **32×26×1** (height=32 time frames, width=26 MFCC coefficients)
- Model architecture adjusted for 32×26 input instead of 32×32

### 3. Python Training Pipeline

**custom_mfcc.py** - Custom MFCC implementation from scratch:

- `apply_preemphasis()` - Pre-emphasis filter (α=0.97): `y[n] = x[n] - 0.97 * x[n-1]`
- `apply_hamming_window()` - Hamming window: `w[n] = 0.54 - 0.46 * cos(2π * n / (N-1))`
- `compute_fft_magnitude()` - FFT with 512 points, compute magnitude spectrum
- `create_mel_filterbank()` - 26 Mel filters (0-8000 Hz) using `mel(f) = 2595 * log10(1 + f/700)`
- `apply_mel_filterbank()` - Apply filters to spectrum, compute Mel energies
- `compute_dct()` - DCT Type-II for 26 coefficients (orthonormal normalization)
- `compute_mfcc()` - Main MFCC computation pipeline
- `normalize_mfcc()` - CMVN normalization: `(mfcc - mean) / std`

**config.json** - MFCC configuration:

```json
{
  "audio": {
    "sample_rate": 16000,
    "duration_seconds": 1.0,
    "num_samples": 16000,
    "channels": 1
  },
  "mfcc": {
    "pre_emphasis": 0.97,
    "frame_length_ms": 25,
    "frame_step_ms": 10,
    "frame_length": 400,
    "frame_step": 160,
    "fft_length": 512,
    "num_mel_filters": 26,
    "num_mfcc_coefficients": 26,
    "mel_low_freq": 0,
    "mel_high_freq": 8000,
    "target_height": 32,
    "target_width": 26
  },
  "model": {
    "input_shape": [32, 26, 1],
    "conv1_filters": 16,
    "conv1_kernel": 3,
    "conv2_filters": 32,
    "conv2_kernel": 3,
    "pool_size": 2,
    "dropout1_rate": 0.25,
    "dense_units": 64,
    "dropout2_rate": 0.5,
    "num_classes": 4
  },
  "training": {
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.0005,
    "early_stopping_patience": 10,
    "validation_split_noise_unknown": 0.2
  },
  "classes": {
    "labels": ["lehitraoot", "shalom", "background", "unknown"],
    "label_map": {
      "lehitraoot": 0,
      "shalom": 1,
      "background": 2,
      "unknown": 3
    }
  },
  "data_paths": {
    "lehitraoot": "../augmented_dataset/lehitraoot",
    "shalom": "../augmented_dataset/shalom",
    "background": "../augmented_dataset/background",
    "unknown": "../augmented_dataset/unknown"
  },
  "output": {
    "model_dir": "models",
    "keras_model": "models/wake_word_model.h5",
    "tflite_model": "models/wake_word_model.tflite",
    "model_header": "models/wake_word_model.h",
    "config_header": "models/model_config.h",
    "training_history": "models/training_history.png",
    "confusion_matrix": "models/confusion_matrix.png"
  },
  "esp32": {
    "tensor_arena_size": 8000,
    "confidence_threshold": 0.7,
    "deployment_dir": "../esp32s3_mfcc_standalone"
  }
}
```

**prepare_dataset.py** modifications:

- Replace STFT with MFCC computation
- Use `custom_mfcc.compute_mfcc()` for feature extraction
- Output shape: (32, 26, 1) instead of (32, 32, 1)
- Compute mean/std for CMVN normalization during training

**train_model.py** modifications:

- Model input shape: [32, 26, 1]
- Same CNN architecture, adjusted for new input dimensions
- Store MFCC normalization statistics (mean/std) for ESP32 deployment

### 4. ESP32 C++ Implementation

**mfcc_extractor.h** - Custom C++ MFCC implementation from scratch:

```cpp
class MFCCExtractor {
private:
    float* fftBuffer;          // FFT workspace (512 points * 2)
    float* hammingWindow;      // Pre-computed Hamming window (400 samples)
    float* melFilterbank;      // Pre-computed 26 filters (26 x 257 bins)
    float* melEnergies;        // Mel energies buffer (26 values)
    float* mfccCoefficients;   // MFCC coefficients (26 values)
    float* dctMatrix;          // Pre-computed DCT matrix (26x26)
    
    // CMVN statistics (from training, will be computed during training)
    float mfcc_mean[26];
    float mfcc_std[26];
    
public:
    MFCCExtractor();
    ~MFCCExtractor();
    bool init();
    bool computeMFCC(const int16_t* audio, float* mfcc_output);
    
private:
    void applyPreemphasis(const int16_t* input, float* output, int length);
    void applyHammingWindow(float* frame, int frameLength);
    void computeFFTMagnitude(const float* frame, float* magnitude, int fftLength);
    void applyMelFilterbank(const float* magnitude, float* mel_energies);
    void applyDCT(const float* mel_energies, float* mfcc);
    void normalizeMFCC(float* mfcc, int num_frames);
    void generateHammingWindow();
    void generateMelFilterbank();
    void generateDCTMatrix();
    void computeFFT(float* real, float* imag, int n);
    void bitReversePermutation(float* real, float* imag, int n);
};
```

**Key C++ functions matching Python exactly**:

- Pre-emphasis: `y[n] = x[n] - 0.97f * x[n-1]`
- Hamming window: `w[n] = 0.54f - 0.46f * cosf(2.0f * PI * n / (N-1))`
- Mel scale: `mel = 2595.0f * log10f(1.0f + f/700.0f)`
- Mel filterbank: 26 triangular filters from 0-8000 Hz
- DCT Type-II: Compute 26 coefficients from 26 Mel energies (orthonormal)
- CMVN: `normalized = (mfcc - mean) / std`

**model_config.h** - Generated from config.json:

- All MFCC parameters as #defines
- MFCC mean/std arrays for normalization (computed during training)
- Input shape: 32×26×1

**esp32s3_mfcc_standalone.ino** modifications:

- Replace `SpectrogramExtractor` with `MFCCExtractor`
- Change buffer size: `MFCC_BUFFER_SIZE = 32 * 26 * 1 = 832`
- Same inference flow, different feature extraction
- Include MFCC normalization in inference

### 5. Files to Copy (unchanged)

- `audio_provider.h` - Same PDM microphone handling
- `inference_engine.h` - Same TFLite inference
- `mic_config.h` - Same microphone configuration
- Python utilities: `convert_to_tflite.py`, `deploy_esp32.py`, `generate_header.py`

### 6. Files to Modify

- `generate_header.py` - Add MFCC mean/std arrays to model_config.h, compute during training
- `run_pipeline.py` - Update for MFCC pipeline, save normalization stats

### 7. Data Paths

- Use existing augmented dataset: `../augmented_dataset/`
- Same 4 classes: lehitraoot, shalom, background, unknown

## Implementation Steps

1. **Create folder structure** - Set up `mfcc/` directory with `model_training/` and `esp32s3_mfcc_standalone/` subdirectories
2. **Implement custom_mfcc.py** - Python MFCC from scratch with all Perplexity parameters
3. **Create config.json** - MFCC configuration matching Perplexity research (26 filters, 26 coefficients, 32×26 output)
4. **Implement mfcc_extractor.h** - C++ MFCC from scratch matching Python exactly
5. **Modify prepare_dataset.py** - Use MFCC instead of STFT, compute normalization stats
6. **Modify train_model.py** - Adjust for 32×26 input shape, save MFCC mean/std
7. **Copy and adapt utilities** - convert_to_tflite, generate_header (add MFCC stats), deploy_esp32
8. **Create ESP32 standalone sketch** - Replace STFT with MFCC extraction
9. **Copy supporting files** - audio_provider, inference_engine, mic_config from STFT version
10. **Create run_pipeline.py** - Complete training pipeline with MFCC
11. **Create README.md** - Documentation for MFCC system, parameters, usage

## Expected Results

- Model input: 32×26×1 (32 time frames × 26 MFCC coefficients)
- Model size: Similar to STFT version (~50-100 KB)
- Inference time: Comparable or slightly faster (fewer features than STFT)
- Accuracy: Potentially better for wake word detection (MFCC designed for speech)

## Key Technical Details

### MFCC Computation Steps

1. **Pre-emphasis**: `y[n] = x[n] - 0.97 * x[n-1]`
2. **Framing**: 400 samples (25 ms) with 160 sample stride (10 ms) - 50% overlap
3. **Windowing**: Apply Hamming window to each frame
4. **FFT**: 512-point FFT, compute magnitude spectrum (257 bins)
5. **Mel Filterbank**: Apply 26 triangular Mel filters (0-8000 Hz)
6. **Log**: Take log of Mel energies
7. **DCT**: Compute 26 DCT coefficients (Type-II, orthonormal)
8. **Normalization**: CMVN using training set mean/std

### ESP32 Memory Requirements

- MFCC buffer: 832 floats = 3,328 bytes (32×26×1)
- FFT workspace: 1,024 floats = 4,096 bytes (512×2)
- Hamming window: 400 floats = 1,600 bytes
- Mel filterbank: 6,682 floats = 26,728 bytes (26×257)
- Mel energies: 26 floats = 104 bytes
- MFCC coefficients: 832 floats = 3,328 bytes (26 × 32 frames)
- DCT matrix: 676 floats = 2,704 bytes (26×26)
- **Total workspace**: ~42 KB (reasonable for ESP32-S3 with 512KB RAM)