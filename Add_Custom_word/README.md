# Custom Wake Word Training

Add custom wake words to your Hebrew wake word detection model with just a few simple steps!

## 🚀 Quick Start

1. **Add your audio samples**: Create a folder in `CustomSoundSamples/` with 5 WAV recordings
2. **Run the training script**: `python train_custom_word.py YourWordName`
3. **Get your trained model**: Ready-to-use model files in the `models/` folder!

## 📋 Requirements

- Python 3.7+
- 5 WAV audio recordings of your custom wake word
- Audio files should be:
  - 16kHz sample rate
  - Mono channel
  - 1 second duration each
  - WAV format

## 🎯 Step-by-Step Guide

### Step 1: Prepare Your Audio Samples

1. Create a new folder in `CustomSoundSamples/` named after your wake word:
   ```
   CustomSoundSamples/
   └── Hello/
       ├── recording1.wav
       ├── recording2.wav
       ├── recording3.wav
       ├── recording4.wav
       └── recording5.wav
   ```

2. Record 5 different utterances of your wake word:
   - Say the word naturally
   - Vary your pronunciation slightly
   - Record in different environments if possible
   - Each recording should be about 1 second long

### Step 2: Run the Training

```bash
# Basic usage
python train_custom_word.py Hello

# Skip noise addition for faster training
python train_custom_word.py Hello --skip-noise

# Keep intermediate files for debugging
python train_custom_word.py Hello --keep-intermediates
```

### Step 3: Use Your Trained Model

After training completes, you'll find:
- `models/hebrew_wake_word_model_cnn.h5` - Your trained model
- `models/training_results_cnn.json` - Training metrics
- `models/custom_word_Hello_summary.json` - Usage instructions

## 🎵 Audio Recording Tips

### Using the Recording Script

We've included a simple recording script to help you capture audio:

```bash
python record_samples.py Hello 5
```

This will:
- Record 5 audio samples
- Save them as WAV files
- Automatically name them for training

### Manual Recording

If you prefer to record manually:

1. Use any audio recording software (Audacity, etc.)
2. Record at 16kHz, mono
3. Trim each recording to ~1 second
4. Save as WAV format
5. Name files descriptively (e.g., `hello_01.wav`, `hello_02.wav`, etc.)

## 🛠️ How It Works

The training pipeline automatically handles:

1. **Audio Processing** (`audio_processor.py`)
   - Splits long recordings into individual words
   - Detects silence to isolate word segments
   - Creates clean word samples

2. **Noise Addition** (`noise_adder.py`)
   - Adds background noise for robustness
   - Uses multiple noise types (cafe, car, factory, etc.)
   - Creates varied training data

3. **Data Preprocessing** (`data_preprocessing.py`)
   - Extracts MFCC features
   - Prepares data for machine learning
   - Splits into train/validation/test sets

4. **Model Training** (`model_training.py`)
   - Trains a CNN model
   - Uses TensorFlow/Keras
   - Optimizes for microcontroller deployment

## 📁 Project Structure

```
Add_Custom_word/
├── CustomSoundSamples/     # Your audio recordings go here
│   └── YourWordName/
│       ├── sample1.wav
│       ├── sample2.wav
│       └── ...
├── ProcessedSamples/       # Auto-generated: processed audio
├── NoisySamples/          # Auto-generated: noise-augmented audio
├── processed_data/        # Auto-generated: ML-ready data
├── models/                # Output: trained models
├── train_custom_word.py   # Main training script
├── record_samples.py      # Audio recording helper
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Troubleshooting

### "No audio samples found"
- Check that your WAV files are in the correct folder
- Ensure files end with `.wav` extension
- Verify you have exactly 5 recordings

### "Audio processing failed"
- Check that audio files are valid WAV format
- Ensure recordings are 16kHz, mono
- Try shorter recordings (1 second each)

### "Model training failed"
- Ensure you have enough RAM (4GB+ recommended)
- Check that all dependencies are installed
- Try reducing batch size if needed

### "Low accuracy"
- Add more varied recordings
- Include different speakers
- Record in different environments
- Don't skip the noise addition step

## 🎛️ Advanced Options

### Custom Noise Levels

The script automatically uses optimal noise levels, but you can modify `train_custom_word.py` to adjust:

```python
# In train_custom_word.py, modify this line:
"--noise-level", "-25.0"  # More negative = quieter noise
```

### Training Parameters

Adjust training hyperparameters by modifying the training command in `train_custom_word.py`:

```python
"--epochs", "30",        # More epochs = longer training
"--batch_size", "16"     # Smaller batch = less memory usage
```

## 📊 Performance Expectations

With 5 clean recordings:
- Training time: 2-5 minutes
- Expected accuracy: 85-95%
- Model size: ~50KB (suitable for microcontrollers)

With noise augmentation:
- Training time: 5-10 minutes
- Expected accuracy: 90-98%
- Better real-world performance

## 🤝 Contributing

Found issues or want to improve the system? The training pipeline uses the same scripts as the main project, so improvements there will benefit custom word training too!

## 📝 License

This custom word training system is part of the Hebrew Wake Word Detection project.
