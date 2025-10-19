# Wake Word Augmentation Pipeline

This directory contains the audio augmentation pipeline for ESP32 wake word detection. The pipeline processes audio files from the `database/` directory and creates augmented datasets for training wake word models.

## Features

- **Conservative ESP32-focused augmentation**: Preserves device-specific microphone characteristics
- **Intelligent scaling**: Automatically calculates augmentation intensity based on current sample count
- **Time stretching**: ±10% speed variations (0.9x-1.1x) to simulate natural speech rate differences
- **Background noise mixing**: Uses background files at 25-35 dB SNR levels for realistic augmentation
- **Complete dataset processing**: Handles all directory types with appropriate processing strategies

## Directory Structure

```
wake_word_augmentation/
├── process_complete_dataset.py  # Main dataset processing script
├── audiolib.py                  # Audio processing utilities
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Dataset Processing Strategy

The pipeline processes different directory types appropriately:

### Wake Word Directories (Augmented)
- **lehitraoot**: ~36 samples → targets ~2450 total (~68x expansion)
- **shalom**: ~79 samples → targets ~2420 total (~31x expansion)  
- **bait**: ~498 samples → targets ~2000-2500 total (~4x expansion)

Augmentation intensity automatically adapts:
- **High expansion** (lehitraoot): All time stretches + heavy noise combinations
- **Medium expansion** (shalom): Most time stretches + moderate noise combinations  
- **Low expansion** (bait): Minimal combinations (1-2 time stretches + 1-2 noise levels)

### Non-Wake Word Directories (Copied As-Is)
- **unknown/**: All files copied without modification
- **background/**: All files copied without modification

## Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run dataset processing**:
   ```bash
   python process_complete_dataset.py
   ```

3. **Output**: The script creates `augmented_dataset/` in the main project directory with:
   ```
   augmented_dataset/
   ├── lehitraoot/
   │   ├── lehitraoot_001.wav (original)
   │   ├── lehitraoot_001_stretch0.90.wav
   │   ├── lehitraoot_001_noise_background_084_snr25.wav
   │   └── ... (~2450 total files)
   ├── shalom/
   │   └── ... (~2500 total files)
   ├── bait/
   │   └── ... (~2000 total files)
   ├── unknown/
   │   └── ... (copied as-is)
   └── background/
       └── ... (copied as-is)
   ```

## Configuration

Key parameters in `process_complete_dataset.py`:

- `target_samples`: Target number of samples per class (default: 2250)
- `time_stretches`: Speed variation factors [0.9, 0.95, 1.0, 1.05, 1.1]
- `snr_levels`: Signal-to-noise ratios [25, 30, 35] dB
- `data_dir`: Source directory (default: "../database")
- `noise_dir`: Noise files directory (default: "../database/background")

## Noise Source

The augmentation uses background noise files from `database/background/` (120 wav files) for realistic noise mixing. These files are shorter and more appropriate for ESP32 device characteristics compared to external noise datasets.

## Logging

The script generates detailed logs including:
- Per-class progress and sample counts
- Augmentation multiplier being applied for wake word directories
- Copy operations for unknown and background directories
- Final dataset statistics per class
- Processing time and error counts

Logs are displayed in the console during execution.

## Dependencies

- `numpy`: Numerical computations
- `librosa`: Audio processing and time stretching
- `soundfile`: Audio file I/O
- `scipy`: Signal processing utilities
- `pathlib`: Path manipulation (Python 3.4+)

## ESP32 Optimization

The augmentation strategy is specifically designed for ESP32 wake word detection:

- **Preserves device characteristics**: Avoids heavy pitch shifting and aggressive gain changes
- **Realistic noise mixing**: Uses device-recorded background files for authentic augmentation
- **Conservative parameters**: Time stretching limited to ±10% to maintain natural speech patterns
- **Appropriate SNR levels**: 25-35 dB SNR provides subtle but effective noise augmentation

## Notes

- All audio files are processed at 16kHz sample rate
- Original wake word files are copied to the output directory alongside augmented versions
- Unknown and background directories are copied as-is without any modification
- The pipeline automatically handles all directory types appropriately
