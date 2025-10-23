# ESP32-S3 Wake Word Deployment

Real-time Hebrew wake word detection system for ESP32-S3 with continuous PDM microphone listening and TFLite inference.

## Hardware Requirements

- **Seeed Studio XIAO ESP32-S3** (with built-in PDM microphone)
- **USB cable** for programming and power
- **Computer** with Arduino IDE and Python 3

## Features

- **5-class wake word detection**: lehitraoot, shalom, bait, background, unknown
- **Continuous listening**: Sliding window detection every 500ms
- **Real-time inference**: STFT spectrogram processing with TFLite Micro
- **Audio transmission**: WAV files sent via serial with confidence scores in filename
- **Memory optimized**: Uses PSRAM for large buffers, 200KB tensor arena for 2MB model

## Installation

### 1. Arduino IDE Setup

1. Install **Arduino IDE** (latest version)
2. Add ESP32 board support:
   - File → Preferences → Additional Board Manager URLs:
   - Add: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Tools → Board → Boards Manager → Search "ESP32" → Install

3. Install required libraries:
   - **ESP_I2S** library (for PDM microphone)
   - **TensorFlowLite_ESP32** library (for inference)

### 2. Board Configuration

1. Select board: **XIAO_ESP32S3**
2. Set PSRAM: **QSPI PSRAM**
3. Set Flash Mode: **QIO 80MHz**
4. Set Partition Scheme: **Huge APP (3MB No OTA/1MB SPIFFS)**

### 3. Upload Code

1. Open `esp32s3_deployment.ino` in Arduino IDE
2. Connect XIAO ESP32-S3 via USB
3. Select correct COM port
4. Click Upload

## Usage

### 1. Start ESP32-S3

After uploading, open Serial Monitor (921600 baud). You should see:

```
========================================
ESP32-S3 Wake Word Detection System
========================================
Allocating workspace buffers...
Audio window: 32000 bytes
Spectrogram: 4096 bytes
Probabilities: 20 bytes
WAV buffer: 32044 bytes
Initializing audio provider...
AudioProvider initialized: 32000 samples buffer
Sample rate: 16000 Hz, Channels: 1
PDM microphone ready
...
System initialized successfully!
Model size: 2121588 bytes (2071.86 KB)
Tensor arena: 200000 bytes (195.31 KB)
Confidence threshold: 0.70
Window stride: 500 ms

Starting continuous wake word detection...
Listening for Hebrew wake words:
  0: lehitraoot
  1: shalom
  2: bait
  3: background
  4: unknown
```

### 2. Receive WAV Files on PC

#### Option A: Python Script (Recommended)

1. Install Python 3 and required packages:
   ```bash
   pip install pyserial
   ```

2. Run the receiver script:
   ```bash
   # List available ports
   python receive_wav_files.py --list-ports
   
   # Start receiving (Windows)
   python receive_wav_files.py --port COM3
   
   # Start receiving (Linux/Mac)
   python receive_wav_files.py --port /dev/ttyUSB0
   ```

3. The script will:
   - Create a `received_wavs/` directory
   - Receive WAV files with detailed filenames
   - Show progress and save files automatically

#### Option B: Manual Serial Capture

1. Use a serial terminal program (PuTTY, Tera Term, etc.)
2. Set baud rate to 921600
3. Capture output to file
4. Parse the binary WAV data manually

### 3. Understanding the Output

#### Serial Monitor Output

```
Inference #1: background (0.234)
Confidences: [0.123, 0.045, 0.067, 0.234, 0.531]
Audio level: 0.045 (max: 0.123)

*** WAKE WORD DETECTED #1 ***
Class: shalom, Confidence: 0.856
Sending: shalom_20241019_143052_conf[0.045, 0.856, 0.034, 0.023, 0.042].wav
```

#### WAV File Names

Files are named with format: `{class}_{timestamp}_conf[{all_scores}].wav`

Example: `shalom_20241019_143052_conf[0.045,0.856,0.034,0.023,0.042].wav`

- **shalom**: Predicted class
- **20241019_143052**: Date and time
- **[0.045,0.856,0.034,0.023,0.042]**: Confidence scores for all 5 classes

## Configuration

All settings are in `config.h` and reference `model_config.h`:

- **Window stride**: 500ms (adjustable)
- **Confidence threshold**: 0.7 (adjustable)
- **Cooldown period**: 2000ms (prevents duplicate detections)
- **Memory allocation**: Optimized for ESP32-S3 with PSRAM

## Troubleshooting

### Common Issues

1. **"Failed to allocate workspace buffers"**
   - Check PSRAM is enabled in board settings
   - Reduce buffer sizes in `config.h`

2. **"Failed to initialize audio provider"**
   - Ensure XIAO_ESP32S3 board is selected
   - Check ESP_I2S library is installed

3. **"Model invoke failed"**
   - Increase tensor arena size in `config.h`
   - Check model file is valid

4. **No audio received**
   - Verify PDM microphone pins (42, 41)
   - Check audio levels in serial monitor
   - Ensure microphone is not obstructed

### Performance Optimization

- **Reduce inference frequency**: Increase `WINDOW_STRIDE_MS`
- **Lower confidence threshold**: Decrease `CONFIDENCE_THRESHOLD`
- **Optimize memory**: Adjust buffer sizes based on available PSRAM

## File Structure

```
esp32s3_deployment/
├── esp32s3_deployment.ino     # Main Arduino sketch
├── audio_provider.h            # PDM microphone management
├── spectrogram_extractor.h     # STFT processing
├── inference_engine.h          # TFLite inference
├── config.h                    # Deployment configuration
├── mic_config.h                # Microphone settings
├── model_config.h              # Model configuration
├── wake_word_model.h           # TFLite model data
├── receive_wav_files.py        # PC receiver script
└── README.md                   # This file
```

## Model Information

- **Architecture**: CNN with 2 conv layers, max pooling, dropout, dense layer
- **Input**: 32×32×1 spectrogram (STFT of 1-second 16kHz audio)
- **Output**: 5 class probabilities
- **Size**: ~2MB (compiled for ESP32-S3)
- **Training**: Hebrew wake words with data augmentation

## License

This project is part of the Hebrew wake word detection system for ESP32-S3.
