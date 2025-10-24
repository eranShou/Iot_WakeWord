# ESP32-S3 Standalone Wake Word Detection

Real-time Hebrew wake word detection system for ESP32-S3 with continuous PDM microphone listening and local TFLite inference. **No PC required** - runs completely standalone with Serial logging and LED visual feedback.

## Hardware Requirements

- **Seeed Studio XIAO ESP32-S3** (with built-in PDM microphone)
- **USB cable** for programming and power
- **Computer** with Arduino IDE (for programming only)

## Features

- **4-class wake word detection**: lehitraoot, shalom, background, unknown
- **Continuous listening**: Sliding window detection every 500ms
- **Real-time inference**: STFT spectrogram processing with TFLite Micro
- **Standalone operation**: No PC required, local inference only
- **Visual feedback**: LED blink on detection (pin 21)
- **Serial logging**: Detailed detection logs with confidence scores
- **Memory optimized**: Uses PSRAM for large buffers, 200KB tensor arena for 2MB model

## Installation

### 1. Arduino IDE Setup

1. Install **Arduino IDE** (latest version)
2. Add ESP32 board support:
   - File → Preferences → Additional Board Manager URLs:
   - Add: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Tools → Board → Boards Manager → Search "ESP32" → Install

3. Install required libraries:
   - **TensorFlowLite_ESP32** library (for inference)

### 2. Board Configuration

1. Select board: **XIAO_ESP32S3**
2. Set PSRAM: **QSPI PSRAM**
3. Set Flash Mode: **QIO 80MHz**
4. Set Partition Scheme: **Huge APP (3MB No OTA/1MB SPIFFS)**

### 3. Upload Code

1. Open `esp32s3_standalone.ino` in Arduino IDE
2. Connect XIAO ESP32-S3 via USB
3. Select correct COM port
4. Click Upload

## Usage

### 1. Start ESP32-S3

After uploading, open Serial Monitor (921600 baud). You should see:

```
========================================
ESP32-S3 Standalone Wake Word Detection System
========================================
Allocating workspace buffers...
Audio window: 32000 bytes
Spectrogram: 4096 bytes
Probabilities: 16 bytes
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
  2: background
  3: unknown
Detection threshold: 0.70
Standalone mode: Local inference only, no PC transmission
```

### 2. Monitor Detections

The system runs completely standalone. Watch the Serial Monitor for:

**Normal Operation (every 10 inferences):**
```
Inference #10: background (0.234) - below threshold
Audio level: 0.045 (max: 0.123)
Audio processing: Gain + Compression applied to window
```

**Wake Word Detection:**
```
*** WAKE WORD DETECTED #1 ***
Class: shalom, Confidence: 0.856 (threshold: 0.700)
All confidences: [0.045, 0.856, 0.023, 0.042]
```

**LED Feedback:**
- LED on pin 21 blinks for 200ms when wake word detected
- Visual confirmation of detection without needing Serial Monitor

### 3. Understanding the Output

#### Detection Behavior

- **Continuous monitoring**: System processes audio every 500ms
- **Threshold filtering**: Only detections above 0.7 confidence are reported
- **LED feedback**: Pin 21 LED blinks for 200ms on detection
- **Serial logging**: All detection details logged to Serial Monitor

#### Detection Format

```
*** WAKE WORD DETECTED #1 ***
Class: shalom, Confidence: 0.856 (threshold: 0.700)
All confidences: [0.045, 0.856, 0.023, 0.042]
```

- **shalom**: Detected wake word class
- **0.856**: Confidence score (above 0.700 threshold)
- **[0.045,0.856,0.023,0.042]**: Confidence scores for all 4 classes

## Configuration

All settings are in `config.h` and reference `model_config.h`:

- **Window stride**: 500ms (adjustable)
- **Confidence threshold**: 0.7 (adjustable)
- **Detection cooldown**: 300ms (prevents duplicate detections)
- **LED pin**: 21 (visual feedback)
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

4. **No detections despite speaking wake words**
   - Check confidence threshold (default 0.7) - try lowering it
   - Verify audio levels in serial monitor
   - Ensure microphone is not obstructed
   - Check if LED is blinking (indicates detection)

5. **"Serial port not found"**
   - Install proper USB drivers for XIAO ESP32-S3
   - Check Device Manager (Windows) or `ls /dev/tty*` (Linux/Mac)
   - Try different USB cable or port

6. **LED not blinking on detection**
   - Check LED is connected to pin 21
   - Verify detection threshold is being met
   - Check Serial Monitor for detection messages

### Performance Optimization

- **Reduce inference frequency**: Increase `WINDOW_STRIDE_MS`
- **Lower confidence threshold**: Decrease `CONFIDENCE_THRESHOLD`
- **Optimize memory**: Adjust buffer sizes based on available PSRAM

## File Structure

```
esp32s3_standalone/
├── esp32s3_standalone.ino     # Main Arduino sketch
├── audio_provider.h            # PDM microphone management
├── spectrogram_extractor.h     # STFT processing
├── inference_engine.h          # TFLite inference
├── config.h                    # Standalone configuration
├── mic_config.h                # Microphone settings
├── model_config.h              # Model configuration
├── wake_word_model.h           # TFLite model data
└── README.md                   # This file
```

## System Architecture

### Audio Processing Pipeline
1. **Audio Capture**: PDM microphone → 16kHz mono audio
2. **Windowing**: 1-second sliding windows with 500ms stride
3. **STFT Processing**: Convert audio to 32×32 spectrogram
4. **ML Inference**: TFLite CNN model → 4-class probabilities
5. **Detection**: Confidence threshold filtering + cooldown period
6. **Output**: Serial logging + LED visual feedback

### Memory Layout
- **PSRAM**: Audio buffers (32KB), spectrogram (4KB)
- **Internal RAM**: Tensor arena (200KB), inference buffers
- **Flash**: Model data (2MB), program code

## Model Information

- **Architecture**: CNN with 2 conv layers, max pooling, dropout, dense layer
- **Input**: 32×32×1 spectrogram (STFT of 1-second 16kHz audio)
- **Output**: 4 class probabilities
- **Size**: ~2MB (compiled for ESP32-S3)
- **Training**: Hebrew wake words with data augmentation

## Quick Start

1. **Hardware Setup**: Connect XIAO ESP32-S3 via USB
2. **Software Setup**: Install Arduino IDE, ESP32 board support, and required libraries
3. **Configuration**: Set board to XIAO_ESP32S3 with PSRAM enabled
4. **Upload**: Flash the `esp32s3_standalone.ino` sketch
5. **Run**: Open Serial Monitor (921600 baud) to see detection logs
6. **Test**: Speak Hebrew wake words ("lehitraoot", "shalom") and watch for LED blinks + Serial output

## Development

### Building from Source
- All source files are in the `esp32s3_standalone/` directory
- Model files are generated from the `model_training/` pipeline
- Configuration is managed through `config.h` and `model_config.h`

### Customization
- **Wake words**: Modify class labels in `model_config.h`
- **Sensitivity**: Adjust `CONFIDENCE_THRESHOLD` in `config.h`
- **Performance**: Tune `WINDOW_STRIDE_MS` for inference frequency
- **Memory**: Optimize buffer sizes based on available PSRAM

## License

This project is part of the Hebrew wake word detection system for ESP32-S3.
