# Voice Sample Audio Processor

A comprehensive Python toolkit that automatically processes voice sample recordings, detects individual word segments using advanced algorithms, splits them into separate audio files, and adds background noise for enhanced training data diversity.

## 🚀 Major Improvements (v2.0)

### Enhanced Reliability
- ✅ **Adaptive Parameter Retry**: Automatically tries different silence detection parameters for failed files
- ✅ **Audio Validation**: Detects corrupted, silent, or invalid audio files before processing
- ✅ **Comprehensive Error Recovery**: Graceful handling of edge cases with detailed error categorization

### Advanced Configuration
- ✅ **YAML Configuration System**: Fully configurable processing parameters via `config.yaml`
- ✅ **Parameter Optimization**: Fine-tune silence detection, validation, and filtering settings
- ✅ **Multiple Retry Strategies**: Configurable parameter combinations for difficult audio files

### User Experience & Monitoring
- ✅ **Rich Progress Visualization**: Real-time progress bars with detailed statistics
- ✅ **Comprehensive Reporting**: Detailed processing reports with failure analysis and recommendations
- ✅ **Enhanced Logging**: Structured logging with configurable verbosity levels

### Technical Enhancements
- ✅ **Continuous Numbering**: Sequential numbering across all files (1-5, 6-10, 11-15, etc.)
- ✅ **Memory Optimization**: Efficient processing for large audio files
- ✅ **Modular Architecture**: Clean, maintainable code with clear separation of concerns

## Features

- **Intelligent Detection**: Advanced silence detection with adaptive parameters
- **Batch Processing**: Processes all WAV files in subfolders automatically
- **Continuous Numbering**: Creates sequential numbering across all files in each subfolder
- **Configuration System**: YAML-based configuration for all processing parameters
- **Progress Visualization**: Rich progress bars and real-time statistics
- **Comprehensive Reporting**: Detailed reports with failure analysis
- **Error Recovery**: Automatic retry with different parameters for failed files
- **Audio Validation**: Pre-processing validation to detect corrupted files
- **Flexible Output**: Customizable output directories and naming conventions
- **Noise Addition**: Automatically add background noise to create diverse training data
- **Random Time Offsets**: Uses random starting points in noise files for varied augmentation
- **Multi-format Support**: Processes both WAV and MP3 noise files

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

### Optional Dependencies
- **tqdm**: Progress bars (automatically disabled if not available)
- **pyyaml**: Configuration file support (uses defaults if not available)

## Installation

1. Clone or download the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Additional Setup for Audio Processing

The program uses `pydub` which requires `ffmpeg` for audio processing:

**Windows:**
- Download ffmpeg from https://ffmpeg.org/download.html
- Add ffmpeg to your system PATH

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

## Directory Structure

Your audio files should be organized as follows:
```
SoundSamples/
├── Lehitraot/
│   ├── Amir-L.wav
│   ├── Badash-L.wav
│   └── ...
├── Shalom/
│   ├── Amir-S.wav
│   ├── Badash-S.wav
│   └── ...
└── ProcessedSamples/  # Output directory (created automatically)
```

## Usage

### Basic Usage

```bash
python audio_processor.py SoundSamples/
```

### Advanced Usage

```bash
# Specify custom output directory
python audio_processor.py SoundSamples/ -o MyProcessedSamples/

# Enable verbose logging
python audio_processor.py SoundSamples/ -v

# Combine options
python audio_processor.py SoundSamples/ -o Processed/ -v
```

## Configuration System

The program supports extensive configuration via a `config.yaml` file. Copy and modify the provided `config.yaml` to customize processing parameters.

### Key Configuration Sections

#### Silence Detection
```yaml
silence_detection:
  min_silence_len: 500          # Minimum silence length in ms
  silence_thresh: -40           # Silence threshold in dBFS
  seek_step: 10                 # Step size for scanning
  retry_attempts:               # Different parameter combinations to try
    - min_silence_len: 300
      silence_thresh: -35
      seek_step: 5
    # ... more combinations
```

#### Audio Validation
```yaml
validation:
  min_duration: 1000            # Minimum audio duration in ms
  silence_threshold: -50        # dBFS threshold for silence detection
  max_rms: 1000000             # Maximum RMS value
  sample_rate: 100              # Sample rate for validation
```

#### Segment Filtering
```yaml
segment_filtering:
  min_segment_length: 200       # Minimum segment length in ms
  max_segment_length: 3000      # Maximum segment length in ms
  padding: 100                  # Padding around segments in ms
```

### Configuration File Location

Place `config.yaml` in the same directory as `audio_processor.py`. The program automatically detects and loads the configuration file. If not found, it uses sensible defaults.

### Command Line Options

- `input_dir`: Path to the SoundSamples directory (required)
- `-o, --output`: Output directory for processed files (default: ProcessedSamples)
- `-v, --verbose`: Enable verbose logging for detailed progress information

## Noise Addition Feature

The `noise_adder.py` script enhances your audio dataset by adding background noise to processed samples with random time offsets, creating multiple noisy versions of each audio file for improved machine learning model training. Each sample gets noise from random starting points in the noise files, ensuring varied and realistic audio augmentation.

### Noise Addition Usage

```bash
# Basic usage - add noise to processed samples
python noise_adder.py ProcessedSamples noise

# Advanced usage with custom settings
python noise_adder.py ProcessedSamples noise -o NoisySamples --noise-level -15.0 -v
```

### Noise Types & Naming Convention

The script automatically maps noise files to short names and creates output files with the pattern `{word}_{number}_{noise}.wav`:

| Noise File | Short Name | Example Output |
|------------|------------|----------------|
| `background-factory.wav` | `factory` | `Lehitraot_1_factory.wav` |
| `cafe.wav` | `cafe` | `Lehitraot_1_cafe.wav` |
| `convention-crowd.wav` | `crowd` | `Lehitraot_1_crowd.wav` |
| `inside-car-while-driving.wav` | `car` | `Lehitraot_1_car.wav` |
| `people-talking.wav` | `talking` | `Lehitraot_1_talking.wav` |
| `car-honk.wav` | `honk` | `Lehitraot_1_honk.wav` |
| `rain.mp3` | `rain` | `Lehitraot_1_rain.wav` |

### Noise Addition Options

- `samples_dir`: Path to processed samples directory (required)
- `noise_dir`: Path to directory containing noise files (required)
- `-o, --output`: Output directory for noisy files (default: `NoisySamples`)
- `--noise-level`: Noise volume in dB (default: -20.0, more negative = quieter)
- `-v, --verbose`: Enable detailed logging

### Complete Workflow Example

```bash
# 1. Process original audio files
python audio_processor.py SoundSamples/

# 2. Add noise to create diverse training data
python noise_adder.py ProcessedSamples noise

# Result: Each processed sample gets 7 noisy versions
# Lehitraot_1.wav → Lehitraot_1_factory.wav, Lehitraot_1_cafe.wav, etc.
```

## How It Works

1. **Discovery**: Scans the input directory for subfolders containing WAV files
2. **Analysis**: For each audio file:
   - Loads the WAV file
   - Detects silent periods to identify word boundaries
   - Filters segments by length and quality
   - Selects the best 5 segments if more are detected
3. **Splitting**: Extracts each word segment with padding
4. **Output**: Saves segments as separate files with naming convention

## Output Format

Split files are saved in the output directory with continuous numbering across all files in each subfolder:
```
ProcessedSamples/
├── Lehitraot/
│   ├── Lehitraot_1.wav   (first word from Amir-L.wav)
│   ├── Lehitraot_2.wav   (second word from Amir-L.wav)
│   ├── Lehitraot_3.wav   (third word from Amir-L.wav)
│   ├── Lehitraot_4.wav   (fourth word from Amir-L.wav)
│   ├── Lehitraot_5.wav   (fifth word from Amir-L.wav)
│   ├── Lehitraot_6.wav   (first word from Badash-L.wav)
│   ├── Lehitraot_7.wav   (second word from Badash-L.wav)
│   ├── Lehitraot_8.wav   (third word from Badash-L.wav)
│   ├── Lehitraot_9.wav   (fourth word from Badash-L.wav)
│   ├── Lehitraot_10.wav  (fifth word from Badash-L.wav)
│   └── ... (continues for all files in Lehitraot/)
└── Shalom/
    ├── Shalom_1.wav      (first word from Amir-S.wav)
    ├── Shalom_2.wav      (second word from Amir-S.wav)
    ├── Shalom_3.wav      (third word from Amir-S.wav)
    ├── Shalom_4.wav      (fourth word from Amir-S.wav)
    ├── Shalom_5.wav      (fifth word from Amir-S.wav)
    ├── Shalom_6.wav      (first word from Badash-S.wav)
    ├── Shalom_7.wav      (second word from Badash-S.wav)
    └── ... (continues for all files in Shalom/)
```

## Audio Processing Parameters

The program uses the following default parameters for silence detection:

- **Minimum silence length**: 500ms
- **Silence threshold**: -40dBFS
- **Segment length range**: 200ms - 3000ms
- **Padding**: 100ms around each segment

These parameters work well for typical voice recordings but can be adjusted in the source code if needed.

## Error Handling

The program handles various error conditions:

- **Missing input directory**: Clear error message with path information
- **Invalid audio files**: Skips corrupted files with warning
- **Insufficient segments**: Warns when fewer than 5 segments are detected
- **Permission errors**: Detailed error messages for file access issues
- **Missing dependencies**: Instructions for installing required libraries

## Logging

The program creates a log file `audio_processor.log` with detailed information about:

- Files processed and their results
- Segments detected and their timing
- Any errors or warnings encountered
- Processing statistics and timing

Use the `-v` flag for more detailed console output.

## Troubleshooting

### Common Issues

1. **"pydub not available"**
   - Install pydub: `pip install pydub`

2. **Audio processing fails**
   - Install ffmpeg and ensure it's in your PATH
   - Check that WAV files are not corrupted

3. **Low success rate (<75%)**
   - Adjust silence detection parameters in config.yaml
   - Check audio quality and recording conditions
   - Enable verbose logging: `python audio_processor.py SoundSamples/ -v`

4. **No segments detected**
   - Try different parameter combinations in config.yaml
   - Check if audio files are completely silent or corrupted
   - Verify ffmpeg installation

5. **Permission denied**
   - Ensure write permissions for the output directory
   - Check that input files are not locked by other programs

6. **Configuration not loading**
   - Ensure config.yaml is in the same directory as audio_processor.py
   - Check YAML syntax with a YAML validator
   - Install pyyaml: `pip install pyyaml`

7. **Noise addition fails**
   - Ensure noise files are valid audio formats (WAV/MP3)
   - Check that ProcessedSamples directory exists and contains WAV files
   - Install pydub: `pip install pydub`
   - Verify ffmpeg installation for audio processing

### Performance Tips

- Process smaller batches of files for better monitoring
- Use SSD storage for faster processing
- Close other audio applications during processing

## Technical Details

### Dependencies

- **pydub**: Audio manipulation and silence detection
- **pathlib**: Modern path handling
- **logging**: Comprehensive logging system
- **argparse**: Command-line argument parsing

### Algorithm Overview

1. **Silence Detection**: Uses pydub's `detect_nonsilent` function
2. **Segment Filtering**: Removes segments that are too short/long
3. **Segment Selection**: Chooses best segments when too many are detected
4. **Audio Extraction**: Cuts segments with padding for natural sound

## Contributing

To modify the audio processing parameters or add new features:

1. Edit the relevant parameters in `audio_processor.py`
2. Test with sample files
3. Update documentation as needed

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions:
1. Check the log file for detailed error information
2. Verify your audio files are valid WAV format
3. Ensure all dependencies are properly installed
4. Try running with verbose logging (`-v` flag)
