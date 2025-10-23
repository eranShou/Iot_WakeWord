# Hebrew Wake Word Detection - Training Pipeline

Complete TensorFlow training pipeline for Hebrew wake word detection (lehitraoot and shalom) with ESP32-S3 deployment. All configuration centralized in `config.json` with zero magic numbers.

## Overview

This pipeline trains a CNN model on 4 classes:
- **lehitraoot** (wake word 1)
- **shalom** (wake word 2) 
- **background** (background noise)
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
augmented_dataset/
├── lehitraoot/         # 1728 training samples
├── shalom/            # 1896 training samples  
├── background/        # 120 background noise samples
└── unknown/           # 2140 negative samples
```

**Training Data:**
- **lehitraoot**: 1728 augmented samples
- **shalom**: 1896 augmented samples
- **background**: 120 background noise samples
- **unknown**: 2140 negative samples
- **Model size**: Optimized for ESP32-S3 compatibility

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

### Spectrogram Configuration
```json
"spectrogram": {
  "frame_length": 255,
  "frame_step": 128,
  "fft_length": 256,
  "target_height": 32,
  "target_width": 32
}
```

### Model Architecture (Optimized for ESP32-S3)
```json
"model": {
  "input_shape": [32, 32, 1],
  "conv1_filters": 16,
  "conv1_kernel": 3,
  "conv2_filters": 32,
  "conv2_kernel": 3,
  "pool_size": 2,
  "dropout1_rate": 0.25,
  "dense_units": 64,
  "dropout2_rate": 0.5,
  "num_classes": 4
}
```

### Training Parameters
```json
"training": {
  "batch_size": 32,
  "epochs": 100,
  "learning_rate": 0.0005,
  "early_stopping_patience": 10,
  "validation_split_noise_unknown": 0.2
}
```

### Class Configuration
```json
"classes": {
  "labels": ["lehitraoot", "shalom", "background", "unknown"],
  "label_map": {
    "lehitraoot": 0,
    "shalom": 1,
    "background": 2,
    "unknown": 3
  }
}
```

## Pipeline Components

### 1. Dataset Preparation (`prepare_dataset.py`)
- Loads 4 classes with proper train/validation split
- Converts audio to spectrograms using custom STFT
- Applies class weighting for balanced training
- Returns TensorFlow datasets with batching/caching

### 2. Model Training (`train_model.py`)
- Config-driven CNN architecture
- Class-weighted training with early stopping
- Generates training curves and confusion matrix
- Saves best model based on validation accuracy

### 3. TFLite Conversion (`convert_to_tflite.py`)
- Converts Keras model to TFLite (float32, no quantization)
- Verifies model size for ESP32-S3 compatibility
- Tests inference with sample data

### 4. Header Generation (`generate_header.py`)
- Creates `wake_word_model.h` with model as C array
- Creates `model_config.h` with all configuration parameters
- Ensures training/inference parameter alignment

### 5. ESP32 Deployment (`deploy_esp32.py`)
- Copies headers to `../esp32s3_deployment/`
- Verifies deployment success
- Prints Arduino integration instructions

### 6. Custom STFT (`custom_stft.py`)
- Pure Python FFT implementation matching ESP32 C++ code
- Ensures perfect alignment between training and inference
- Radix-2 FFT with bit-reversal permutation

### 7. Pipeline Testing (`test_pipeline.py`)
- Verifies data paths and file counts
- Tests model architecture optimization
- Validates configuration parameters
- Checks ESP32 compatibility constraints

## Usage

### Single Command Execution
```bash
python run_pipeline.py
```

### Individual Steps
```bash
# Test pipeline configuration
python test_pipeline.py

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

- **Overall accuracy**: >85% (with augmented dataset)
- **Wake word accuracy**: >90% (lehitraoot, shalom)
- **Model size**: Optimized for ESP32-S3
- **Training time**: <30 minutes
- **ESP32 inference**: <1 second latency

## Key Features

### Custom STFT Implementation
- **Perfect alignment**: Python training matches ESP32 C++ inference exactly
- **Radix-2 FFT**: Pure Python implementation with bit-reversal permutation
- **No dependencies**: Custom FFT ensures consistent results across platforms

### Optimized Model Architecture
- **ESP32-S3 compatible**: Designed for microcontroller deployment
- **Efficient spectrograms**: 32x32 input with optimized FFT parameters
- **Balanced training**: Class weighting for imbalanced datasets

### Comprehensive Testing
- **Pipeline validation**: Automated testing of all components
- **Data verification**: Checks file counts and paths
- **Model validation**: Ensures architecture meets ESP32 constraints

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
- Increase `epochs` to 200
- Adjust `learning_rate` (0.0001 to 0.01)
- Increase model complexity

**For Faster Training:**
- Reduce `epochs` to 20
- Increase `batch_size` to 64
- Use fewer training samples

**For ESP32 Compatibility:**
- Keep model size <6MB
- Use simple architecture
- Avoid complex operations

## ESP32-S3 Integration

After training, the generated headers are automatically deployed to `../esp32s3_deployment/`:

### Arduino Setup
1. Install libraries:
   - TensorFlowLite_ESP32


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
  "lehitraoot", "shalom", "background", "unknown"
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
├── test_pipeline.py     # Pipeline testing
├── prepare_dataset.py      # Dataset loading
├── train_model.py          # Model training
├── convert_to_tflite.py    # TFLite conversion
├── generate_header.py      # C header generation
├── deploy_esp32.py         # ESP32 deployment
├── custom_stft.py          # Custom STFT implementation
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
