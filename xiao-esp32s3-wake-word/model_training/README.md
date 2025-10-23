# Hebrew Wake Word Detection - Training Pipeline

Complete TensorFlow training pipeline for Hebrew wake word detection (lehitraoot and shaloom) with ESP32-S3 deployment. All configuration centralized in `config.json` with zero magic numbers.

## Overview

This pipeline trains a CNN model on 4 classes:
- **lehitraoot** (wake word 1)
- **shaloom** (wake word 2) 
- **noise** (background noise)
- **unknown** (negative samples)

The trained model is automatically converted to TFLite format and deployed to ESP32-S3 for real-time inference.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run complete pipeline
python run_pipeline.py
```

That's it! The pipeline will:
1. Load and prepare your dataset
2. Train the CNN model
3. Convert to TFLite
4. Generate C headers
5. Deploy to ESP32-S3

## Data Structure

The pipeline expects this data structure:

```
data/
├── augmented/
│   ├── lehitraoot/     # 619 training samples
│   └── shaloom/        # 437 training samples
├── ivrit-ai/          # NEW: Additional Hebrew speech data
│   ├── lehitraoot/     # Extracted from crowd-recital dataset
│   └── shaloom/        # Extracted from crowd-recital dataset
├── lehitraoot/         # 36 validation samples
├── shaloom/           # 26 validation samples
├── noise/             # 128 samples (80/20 split)
└── unknown/           # 140 samples (80/20 split)
```

**Enhanced Training Data:**
- **Existing augmented data**: 619 lehitraoot + 437 shaloom
- **ivrit-ai dataset**: Additional Hebrew speech recordings
- **Total training samples**: Significantly increased for better accuracy
- **Model size**: Reduced to <500KB for ESP32-S3 compatibility

## Configuration

All parameters are in `config.json` - no magic numbers in code:

### Audio Configuration
```json
"audio": {
  "sample_rate": 16000,
  "duration_seconds": 1.0,
  "num_samples": 16000,
  "channels": 1
}
```

### Model Architecture (Reduced for <500KB)
```json
"model": {
  "input_shape": [32, 32, 1],
  "conv1_filters": 16,      // Reduced from 32
  "conv1_kernel": 3,
  "conv2_filters": 32,      // Reduced from 64
  "conv2_kernel": 3,
  "pool_size": 2,
  "dropout1_rate": 0.25,
  "dense_units": 64,        // Reduced from 128
  "dropout2_rate": 0.5,
  "num_classes": 4
}
```

### Training Parameters
```json
"training": {
  "batch_size": 32,
  "epochs": 50,
  "learning_rate": 0.001,
  "early_stopping_patience": 5,
  "validation_split_noise_unknown": 0.2
}
```

## Pipeline Components

### 1. Dataset Preparation (`prepare_dataset.py`)
- Loads 4 classes with proper train/validation split
- Converts audio to spectrograms using STFT
- Applies class weighting for balanced training
- Returns TensorFlow datasets with batching/caching

### 2. Model Training (`train_model.py`)
- Config-driven CNN architecture
- Class-weighted training with early stopping
- Generates training curves and confusion matrix
- Saves best model based on validation accuracy

### 3. TFLite Conversion (`convert_to_tflite.py`)
- Converts Keras model to TFLite (float32, no quantization)
- Verifies model size (<100KB for ESP32-S3)
- Tests inference with sample data

### 4. Header Generation (`generate_header.py`)
- Creates `wake_word_model.h` with model as C array
- Creates `model_config.h` with all configuration parameters
- Ensures training/inference parameter alignment

### 5. ESP32 Deployment (`deploy_esp32.py`)
- Copies headers to `../esp32s3_deployment/`
- Verifies deployment success
- Prints Arduino integration instructions

## Usage

### Single Command Execution
```bash
python run_pipeline.py
```

### Individual Steps
```bash
# Download ivrit-ai dataset (optional, for additional training data)
python download_ivrit_dataset.py

# Prepare dataset
python prepare_dataset.py

# Train model
python train_model.py

# Convert to TFLite
python convert_to_tflite.py

# Generate headers
python generate_header.py

# Deploy to ESP32
python deploy_esp32.py
```

## Output Files

After successful execution:

```
models/
├── wake_word_model.h5          # Trained Keras model
├── wake_word_model.tflite      # TFLite model (<100KB)
├── wake_word_model.h           # C header with model data
├── model_config.h              # C header with configuration
├── training_history.png        # Training curves
└── confusion_matrix.png        # Confusion matrix

../esp32s3_deployment/
├── wake_word_model.h           # Copied for ESP32
├── model_config.h              # Copied for ESP32
└── esp32s3_deployment.ino      # Arduino sketch
```

## Expected Performance

- **Overall accuracy**: >85% (improved with additional ivrit-ai data)
- **Wake word accuracy**: >90% (lehitraoot, shaloom)
- **Model size**: <500KB (reduced architecture)
- **Training time**: <30 minutes
- **ESP32 inference**: <1 second latency

## New Features

### Enhanced Dataset
- **ivrit-ai/crowd-recital integration**: Additional Hebrew speech data
- **Automatic download**: Downloads and processes dataset during training
- **Improved accuracy**: More training data leads to better model performance

### Optimized Model Size
- **Reduced architecture**: Smaller CNN for <500KB model size
- **ESP32-S3 compatible**: Optimized for microcontroller deployment
- **Maintained accuracy**: Smaller model with better training data

## Troubleshooting

### Common Issues

**1. Missing Dependencies**
```bash
pip install -r requirements.txt
```

**2. Data Path Errors**
- Check that all paths in `config.json` exist
- Ensure WAV files are in correct directories

**3. Memory Issues**
- Reduce `batch_size` in config
- Reduce `epochs` for faster training

**4. Poor Accuracy**
- Check class balance in dataset
- Increase `epochs` or adjust `learning_rate`
- Verify audio quality and labeling

**5. Model Too Large**
- Reduce model architecture in config
- Check for unnecessary layers

### Configuration Tips

**For Better Accuracy:**
- Increase `epochs` to 100
- Adjust `learning_rate` (0.0001 to 0.01)
- Increase model complexity

**For Faster Training:**
- Reduce `epochs` to 20
- Increase `batch_size` to 64
- Use fewer training samples

**For ESP32 Compatibility:**
- Keep model size <100KB
- Use simple architecture
- Avoid complex operations

## ESP32-S3 Integration

After training, the generated headers are automatically deployed to `../esp32s3_deployment/`:

### Arduino Setup
1. Install libraries:
   - TensorFlowLite_ESP32
   - ESP_I2S

2. Include headers in your sketch:
```cpp
#include "wake_word_model.h"
#include "model_config.h"
```

3. Use configuration constants:
```cpp
// Audio parameters
#define SAMPLE_RATE 16000
#define NUM_SAMPLES 16000

// Model parameters  
#define NUM_CLASSES 4
#define SPECTROGRAM_HEIGHT 32
#define SPECTROGRAM_WIDTH 32

// Class labels
const char* CLASS_LABELS[NUM_CLASSES] = {
  "lehitraoot", "shaloom", "noise", "unknown"
};
```

### Real-time Inference
The ESP32-S3 continuously:
1. Captures 1-second audio windows
2. Converts to spectrograms
3. Runs inference
4. Outputs predictions with confidence scores
5. Sends audio + predictions to PC

## Advanced Configuration

### Custom Model Architecture
Edit `config.json` model section:
```json
"model": {
  "conv1_filters": 64,      // More filters
  "conv2_filters": 128,      // Deeper network
  "dense_units": 256,        // Larger dense layer
  "dropout1_rate": 0.3,      // More regularization
  "dropout2_rate": 0.6       // Higher dropout
}
```

### Audio Processing
Adjust spectrogram parameters:
```json
"spectrogram": {
  "frame_length": 512,       // Larger FFT
  "frame_step": 256,        // Different overlap
  "target_height": 64,       // Higher resolution
  "target_width": 64
}
```

### Training Optimization
Fine-tune training:
```json
"training": {
  "batch_size": 16,         // Smaller batches
  "epochs": 100,            // More training
  "learning_rate": 0.0001,  // Lower learning rate
  "early_stopping_patience": 10
}
```

## File Structure

```
model_training/
├── config.json              # Central configuration
├── requirements.txt         # Python dependencies
├── run_pipeline.py         # Main pipeline script
├── prepare_dataset.py      # Dataset loading
├── train_model.py          # Model training
├── convert_to_tflite.py    # TFLite conversion
├── generate_header.py      # C header generation
├── deploy_esp32.py         # ESP32 deployment
├── README.md              # This file
└── models/                 # Output directory
    ├── wake_word_model.h5
    ├── wake_word_model.tflite
    ├── wake_word_model.h
    ├── model_config.h
    ├── training_history.png
    └── confusion_matrix.png
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your data structure matches requirements
3. Ensure all dependencies are installed
4. Check that config.json is valid

The pipeline is designed to be robust and provide clear error messages for common issues.
