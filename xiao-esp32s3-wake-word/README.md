# Hebrew Wake Word Detection Pipeline

A complete pipeline for training and deploying Hebrew wake word detection models on ESP32-S3 microcontrollers. This project enables real-time detection of Hebrew wake words like "lehitraoot" and "shalom" using the XIAO ESP32-S3's built-in microphone.

## 🎯 Project Overview

This pipeline processes audio recordings through multiple stages:
1. **Audio Recording** - Record wake word samples using ESP32
2. **Audio Processing** - Split and organize audio files
3. **Data Transfer** - Move processed data to database
4. **Data Augmentation** - Enhance dataset with synthetic variations
5. **Model Training** - Train CNN model for wake word detection
6. **ESP32 Deployment** - Deploy trained model to microcontroller

## 📋 Prerequisites

- **Hardware**: Seeed Studio XIAO ESP32-S3 (with built-in PDM microphone)
- **Software**: Arduino IDE, Python 3.7+
- **USB Cable**: For programming and data transfer

## 🚀 Complete Pipeline Workflow

### Step 1: Audio Recording (`audio_recorder_esp32_pc/`)

Record 50 samples of each wake word and background audio.

#### Setup
```bash
cd audio_recorder_esp32_pc/
pip install pyserial
```

#### Upload Arduino Code
1. Open `audio_recorder_esp32_pc.ino` in Arduino IDE
2. Select board: **XIAO_ESP32S3**
3. Select correct COM port
4. Upload the sketch

#### Record Wake Words
```bash
python record_audio.py
```

**Recording Guidelines:**
- Record **50 samples** of each wake word ("lehitraoot", "shalom")
- Record **3 minute** of background noise (silence or ambient sound)
- Organize recordings in separate folders:
  ```
  recordings/
  ├── lehitraoot/
  ├── shaloom/
  └── background/
  ```

---

### Step 2: Audio Processing (`audio_prosser/`)

Split recordings into individual word segments using silence detection.

#### Setup
```bash
cd audio_prosser/
pip install -r requirements.txt
```

#### Copy Recordings
```bash
python copy_recordings.py
```
*Copies WAV files from `../audio_recorder_esp32_pc/recordings/` to `data/`*

#### Split Audio Files
```bash
python audio_split.py
```
*Splits recordings using silence detection algorithm and saves as `{word_name}_001.wav`, `{word_name}_002.wav`, etc.*

**Output Structure:**
```
audio_prosser/data/
├── lehitraoot/
│   ├── lehitraoot_001.wav
│   ├── lehitraoot_002.wav
│   └── ...
├── shaloom/
│   ├── shaloom_001.wav
│   ├── shaloom_002.wav
│   └── ...
└── background/
    ├── background_001.wav
    └── ...
```

---

### Step 3: Data Transfer to Database

Transfer processed audio files to the main database directory.

**Final Database Structure:**
```
database/
├── lehitraoot/
│   ├── lehitraoot_001.wav
│   └── ... 
├── shalom/
│   ├── shalom_001.wav
│   └── ... 
├── background/
│   ├── background_001.wav
│   └── ... 
└── unknown/
    └── ... 
```

---

### Step 4: Data Augmentation (`wake_word_augmentation/`)

Enhance the dataset with synthetic variations for better model training.

#### Setup
```bash
cd wake_word_augmentation/
pip install -r requirements.txt
```

#### Run Augmentation
```bash
python process_complete_dataset.py
```

**Augmentation Features:**
- **Time stretching**: ±10% speed variations (0.9x-1.1x)
- **Background noise mixing**: Subtle noise at 25-35 dB SNR
- **Target samples**: ~2250 samples per wake word class
- **Conservative approach**: ESP32-focused augmentation

**Output:**
```
augmented_dataset/
├── lehitraoot/ (~2450 files)
├── shalom/ (~2420 files)
├── background/ (copied as-is)
└── unknown/ (copied as-is)
```

---

### Step 5: Model Training (`model_training/`)

Train a CNN model for Hebrew wake word detection.

#### Setup
```bash
cd model_training/
pip install -r requirements.txt
```

#### Run Complete Training Pipeline
```bash
python run_pipeline.py
```

**Training Pipeline Includes:**
1. **Dataset Preparation** - Load and preprocess audio data
2. **Model Training** - Train CNN with early stopping
3. **TFLite Conversion** - Convert to ESP32-compatible format
4. **Header Generation** - Create C headers for deployment
5. **ESP32 Deployment** - Copy files to deployment directory

**Configuration:**
All settings are in `config.json`

**Output Files:**
```
models/
├── wake_word_model.h5          # Trained Keras model
├── wake_word_model.tflite      # ESP32-compatible model
├── wake_word_model.h           # C header with model data
├── model_config.h              # Configuration header
├── training_history.png        # Training curves
└── confusion_matrix.png        # Model performance
```

---

### Step 6: ESP32-S3 Deployment (`esp32s3_deployment/`)

Deploy the trained model to ESP32-S3 for real-time inference.

#### Arduino IDE Setup
1. Install **ESP32 board support** (Board Manager version 2.0.16)
2. Install libraries:
   - **I2S** (for PDM microphone)
   - **TensorFlowLite_ESP32** (for inference)

#### Board Configuration
- Board: **XIAO_ESP32S3**
- PSRAM: **QSPI PSRAM**
- Flash Mode: **QIO 80MHz**
- Partition: **Huge APP (3MB No OTA/1MB SPIFFS)**

#### Upload and Test
1. Open `esp32s3_deployment.ino` in Arduino IDE
2. Connect XIAO ESP32-S3 via USB
3. Select correct COM port
4. Upload the sketch

#### Receive Audio Files
```bash
cd esp32s3_deployment/
pip install pyserial

# Start receiving WAV files
python receiver.py
```

**Real-time Features:**
- **Continuous listening**: Sliding window detection every 500ms
- **4-class detection**: lehitraoot, shalom, background, unknown
- **Confidence scoring**: Detailed confidence levels for all classes
- **Audio transmission**: WAV files sent via serial

## 📁 Project Structure

```
xiao-esp32s3-wake-word/
├── audio_recorder_esp32_pc/     # Step 1: Record audio samples
│   ├── audio_recorder_esp32_pc.ino
│   ├── record_audio.py
│   └── recordings/
├── audio_prosser/              # Step 2: Process and split audio
│   ├── audio_split.py
│   ├── copy_recordings.py
│   └── data/
├── database/                   # Step 3: Organized audio database
│   ├── copy_data.py
│   ├── lehitraoot/
│   ├── shalom/
│   ├── background/
│   └── unknown/
├── wake_word_augmentation/     # Step 4: Data augmentation
│   ├── process_complete_dataset.py
│   ├── audiolib.py
│   └── noise_train/
├── model_training/            # Step 5: Train ML model
│   ├── run_pipeline.py
│   ├── train_model.py
│   ├── config.json
│   └── models/
├── esp32s3_deployment/        # Step 6: Deploy to ESP32
│   ├── esp32s3_deployment.ino
│   ├── receive_wav_files.py
│   └── received_wavs/
└── README.md                  # This file
```

## 🎛️ Configuration Files

### Audio Recording Configuration
- **Sample Rate**: 16,000 Hz
- **Bit Depth**: 16-bit
- **Channels**: Mono
- **Duration**: 5 seconds (customizable 1-60 seconds)

### Model Configuration
- **Architecture**: CNN with 2 conv layers + dense layer
- **Input**: 32×32 spectrogram
- **Classes**: 4 (lehitraoot, shalom, background, unknown)
- **Model Size**: <500KB (ESP32-optimized)

### ESP32 Configuration
- **Window Size**: 1 second
- **Stride**: 500ms
- **Confidence Threshold**: 0.7
- **Memory**: Uses PSRAM for large buffers


## 📊 Expected Performance

- **Overall Accuracy**: >85%
- **Wake Word Detection**: >90%
- **Model Size**: <500KB
- **Training Time**: <30 minutes
- **ESP32 Inference**: <1 second latency

## 🔧 Customization

### Adding New Wake Words
1. Record new samples in `audio_recorder_esp32_pc/recordings/new_word/`
2. Process through audio splitting pipeline
3. Update `config.json` to include new class
4. Retrain model with updated configuration

### Adjusting Model Architecture
Edit `model_training/config.json`:
```json
{
  "model": {
    "conv1_filters": 16,
    "conv2_filters": 32,
    "dense_units": 64
  }
}
```

### Changing Audio Parameters
Modify audio settings in respective configuration files:
- Recording: `audio_recorder_esp32_pc/mic_config.h`
- Training: `model_training/config.json`
- Deployment: `esp32s3_deployment/config.h`


## 🤝 Contributing

1. Follow the pipeline workflow outlined above
2. Test each step thoroughly before proceeding
3. Maintain consistent file naming conventions
4. Update configuration files when making changes

## 📞 Support

For issues or questions:
1. Review configuration files for correctness
2. Verify your hardware setup matches requirements
3. Ensure all dependencies are properly installed

---

**Happy Wake Word Detection! 🎤✨**
