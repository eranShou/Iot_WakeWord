#!/usr/bin/env python3
"""
Setup Script for Hebrew Wake Word Detection Project

This script automates the initial setup of the project, including:
- Installing Python dependencies
- Setting up the project structure
- Running initial tests
- Generating documentation
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse

def run_command(command, description=""):
    """Run a shell command with error handling."""
    try:
        print(f"Running: {description}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running: {description}")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return None

def install_python_dependencies():
    """Install required Python packages."""
    print("\n=== Installing Python Dependencies ===")

    requirements = [
        "tensorflow>=2.13.0",
        "numpy>=1.21.0",
        "librosa>=0.10.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "jupyter>=1.0.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0"
    ]

    # Create requirements.txt
    with open("requirements.txt", "w") as f:
        f.write("\n".join(requirements))

    # Install packages
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )

    if success:
        print("✓ Python dependencies installed successfully")
    else:
        print("✗ Failed to install Python dependencies")
        return False

    return True

def setup_project_structure():
    """Ensure project structure is correct."""
    print("\n=== Setting Up Project Structure ===")

    directories = [
        "training",
        "microcontroller",
        "models",
        "docs",
        "scripts",
        "processed_data",
        "custom_models"
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

    # Create __init__.py files for Python packages
    init_files = [
        "training/__init__.py",
        "scripts/__init__.py"
    ]

    for init_file in init_files:
        Path(init_file).touch()
        print(f"✓ Created: {init_file}")

    return True

def check_audio_data():
    """Check if audio data is available."""
    print("\n=== Checking Audio Data ===")

    processed_samples = Path("ProcessedSamples")
    if not processed_samples.exists():
        print("✗ ProcessedSamples directory not found")
        print("Please ensure Hebrew audio samples are in ProcessedSamples/ folder")
        return False

    shalom_dir = processed_samples / "Shalom"
    lehitraot_dir = processed_samples / "Lehitraot"

    if not shalom_dir.exists():
        print("✗ Shalom directory not found")
        return False

    if not lehitraot_dir.exists():
        print("✗ Lehitraot directory not found")
        return False

    # Count audio files
    shalom_files = list(shalom_dir.glob("*.wav"))
    lehitraot_files = list(lehitraot_dir.glob("*.wav"))

    print(f"✓ Found {len(shalom_files)} Shalom samples")
    print(f"✓ Found {len(lehitraot_files)} Lehitraot samples")

    if len(shalom_files) < 50 or len(lehitraot_files) < 50:
        print("⚠ Warning: Low number of audio samples detected")
        print("  Recommended: At least 100 samples per wake word for good performance")

    return True

def run_initial_tests():
    """Run initial tests to verify setup."""
    print("\n=== Running Initial Tests ===")

    # Test imports
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")

        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")

        import librosa
        print(f"✓ Librosa {librosa.__version__} imported successfully")

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

    # Test audio processing
    try:
        import librosa
        # Simple test with dummy data
        dummy_audio = np.random.randn(16000)
        mfccs = librosa.feature.mfcc(y=dummy_audio, sr=16000, n_mfcc=40)
        print(f"✓ Audio processing test passed (MFCC shape: {mfccs.shape})")

    except Exception as e:
        print(f"✗ Audio processing test failed: {e}")
        return False

    return True

def generate_documentation():
    """Generate initial documentation files."""
    print("\n=== Generating Documentation ===")

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    # Setup instructions
    setup_content = """# Setup Instructions

## Prerequisites
- Python 3.8 or higher
- Arduino IDE 2.0+ or PlatformIO
- Seeed Studio XIAO ESP32-S3 board

## Installation Steps

### 1. Clone Repository
```bash
git clone <repository-url>
cd tensorflow_wake_word_detection
```

### 2. Run Setup Script
```bash
python scripts/setup.py --full
```

### 3. Train Model (Optional)
If you want to retrain the model:
```bash
cd training
python data_preprocessing.py
python model_training.py
python model_conversion.py --model_path ../models/hebrew_wake_word_model_cnn.h5
```

### 4. Setup Microcontroller
- Open Arduino IDE
- Install ESP32 board support
- Install TensorFlow Lite Micro library
- Open `microcontroller/hebrew_wake_word_detector.ino`
- Upload to XIAO ESP32-S3

## Troubleshooting
- Ensure all Python dependencies are installed
- Check audio sample quality
- Verify ESP32-S3 connections
- Monitor serial output for debug information
"""

    with open(docs_dir / "setup_instructions.md", "w") as f:
        f.write(setup_content)

    # Customization guide
    custom_content = """# Customization Guide

## Adding Custom Wake Words

### Method 1: Transfer Learning (Recommended)
```bash
python training/custom_word_training.py \\
    --base_model models/hebrew_wake_word_model_cnn.h5 \\
    --new_words "ahlan" "toda" \\
    --custom_audio_dir ProcessedSamples
```

### Method 2: Full Retraining
1. Add audio samples to `ProcessedSamples/CustomWord/`
2. Run preprocessing: `python training/data_preprocessing.py`
3. Train model: `python training/model_training.py`
4. Convert model: `python training/model_conversion.py`

## Configuration Options

### Audio Parameters
- `SAMPLE_RATE`: Audio sampling rate (default: 16000)
- `WINDOW_SIZE_MS`: FFT window size (default: 30)
- `FEATURE_BINS`: Number of MFCC bins (default: 40)

### Detection Parameters
- `DETECTION_THRESHOLD`: Confidence threshold (default: 0.8)
- `COOLDOWN_MS`: Minimum time between detections (default: 2000)

### Hardware Pins
- `LED_PIN`: LED output pin (default: 21)
- `MIC_DATA_PIN`: Microphone data pin (default: 23)
- `MIC_CLOCK_PIN`: Microphone clock pin (default: 22)

## Performance Optimization

### Model Optimization
- Use quantization for smaller model size
- Implement model pruning
- Use knowledge distillation

### Hardware Optimization
- Reduce CPU frequency for power savings
- Use light sleep between detections
- Optimize memory usage

## Advanced Features

### Multi-Wake Word Support
The system supports up to 5 custom wake words simultaneously.

### Real-time Adaptation
Implement online learning for continuous model improvement.

### Multi-Language Support
Extend to other languages by collecting appropriate audio datasets.
"""

    with open(docs_dir / "customization_guide.md", "w") as f:
        f.write(custom_content)

    # Troubleshooting guide
    trouble_content = """# Troubleshooting Guide

## Common Issues and Solutions

### 1. Microphone Not Working
**Symptoms:**
- No audio input detected
- Poor audio quality

**Solutions:**
- Check I2S pin connections
- Verify microphone power supply
- Test with simple audio recording sketch
- Check sample rate configuration

### 2. Model Not Loading
**Symptoms:**
- "Failed to setup TensorFlow Lite Micro" error
- Memory allocation errors

**Solutions:**
- Ensure model file is in correct format
- Check available memory on ESP32-S3 (80KB minimum)
- Verify model was converted with correct quantization
- Reduce model size if necessary

### 3. Poor Detection Accuracy
**Symptoms:**
- Wake words not detected consistently
- False positive detections

**Solutions:**
- Check audio quality and background noise
- Adjust DETECTION_THRESHOLD (lower = more sensitive)
- Retrain model with more/better audio samples
- Verify audio preprocessing parameters

### 4. Memory Issues
**Symptoms:**
- Out of memory errors
- System crashes

**Solutions:**
- Reduce TENSOR_ARENA_SIZE if possible
- Optimize audio buffer sizes
- Use lighter model architecture
- Enable memory monitoring in debug mode

### 5. Slow Inference
**Symptoms:**
- High latency in wake word detection
- System feels unresponsive

**Solutions:**
- Reduce CPU frequency for power savings
- Optimize model architecture
- Use quantization for faster inference
- Implement efficient MFCC extraction

## Debug Mode

Enable debug output in `config.h`:
```cpp
#define ENABLE_SERIAL_DEBUG true
#define ENABLE_PERFORMANCE_MONITORING true
```

## Performance Monitoring

Check system performance:
```cpp
print_system_info();
print_tensor_info();
MemoryMonitor::printMemoryInfo();
```

## Getting Help

1. Check serial output for error messages
2. Verify hardware connections
3. Test with known working audio samples
4. Review configuration parameters
5. Check GitHub issues for similar problems

## Diagnostic Commands

### Hardware Diagnostics
- Check ESP32-S3 chip information
- Verify microphone connections
- Test LED functionality

### Software Diagnostics
- Print memory usage statistics
- Monitor inference timing
- Check audio signal quality

### Model Diagnostics
- Verify model file integrity
- Check model input/output shapes
- Validate quantization parameters
"""

    with open(docs_dir / "troubleshooting.md", "w") as f:
        f.write(trouble_content)

    print("✓ Documentation generated successfully")
    return True

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup Hebrew Wake Word Detection project')
    parser.add_argument('--full', action='store_true',
                       help='Run full setup including dependency installation')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip Python dependency installation')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip initial tests')

    args = parser.parse_args()

    print("=== Hebrew Wake Word Detection Setup ===")

    # Setup project structure
    if not setup_project_structure():
        print("✗ Failed to setup project structure")
        return 1

    # Check audio data
    if not check_audio_data():
        print("⚠ Audio data check failed - please ensure audio samples are available")
        # Don't exit - continue with setup

    # Install dependencies
    if not args.skip_deps and (args.full or input("Install Python dependencies? (y/n): ").lower() == 'y'):
        if not install_python_dependencies():
            print("✗ Failed to install Python dependencies")
            return 1

    # Run initial tests
    if not args.skip_tests:
        if not run_initial_tests():
            print("✗ Initial tests failed")
            return 1

    # Generate documentation
    if not generate_documentation():
        print("✗ Failed to generate documentation")
        return 1

    print("\n=== Setup Complete ===")
    print("Next steps:")
    print("1. Review the generated documentation in docs/")
    print("2. Run training: cd training && python data_preprocessing.py")
    print("3. Setup Arduino IDE and upload microcontroller/hebrew_wake_word_detector.ino")
    print("4. Say 'shalom' or 'lehitraot' to test wake word detection!")

    return 0

if __name__ == "__main__":
    sys.exit(main())
