# Hebrew Dataset Extraction - ivrit-ai Database

This module extracts and processes Hebrew audio data from the `ivrit-ai/crowd-recital` dataset for wake word detection training. It focuses on extracting specific Hebrew words (`shalom`, `lehitraoot`, `bait`) and organizing them for machine learning model training.

## Overview

The ivrit-ai dataset contains Hebrew speech recordings with word-level alignments. This extraction tool processes the dataset to:

- Download the complete dataset from HuggingFace
- Extract specific target Hebrew words with precise timing
- Organize audio files by word category
- Generate vocabulary analysis and word frequency statistics
- Create training-ready datasets for wake word detection

## Target Words

The system extracts the following Hebrew words:
- **שלום** (shalom) - "hello/goodbye"
- **להתראות** (lehitraoot) - "goodbye" 
- **בית** (bait) - "house"

## Files Structure

```
ivrit-ai_DB_extraction/
├── README.md                           # This file
├── config.json                         # Configuration settings
├── requirements.txt                    # Python dependencies
├── download_ivrit_dataset.py          # Download dataset from HuggingFace
├── extract_all_words.py               # Basic word extraction
├── extract_all_words_parallel.py     # Parallel processing version
├── extract_all_words_optimized.py    # Optimized extraction with vocabulary filtering
├── analyze_vocabulary.py             # Vocabulary analysis and statistics
├── data/                              # Output directory
│   ├── shalom/                        # Extracted "shalom" audio files
│   ├── lehitraoot/                    # Extracted "lehitraoot" audio files
│   ├── bait/                          # Extracted "bait" audio files
│   ├── unknown/                       # Other words for negative examples
│   ├── vocabulary_analysis.json       # Complete vocabulary statistics
│   └── word_counts.json              # Simple word frequency counts
├── vocabulary_analysis.json          # Generated vocabulary analysis
└── word_counts.json                  # Generated word frequency data
```

## Configuration

The `config.json` file contains all extraction parameters:

```json
{
  "audio": {
    "sample_rate": 16000,
    "duration_seconds": 1.0,
    "num_samples": 16000,
    "channels": 1
  },
  "dataset_path": "path/to/dataset",
  "output_dir": "./data",
  "target_words": {
    "shalom": "שלום",
    "lehitraoot": "להתראות", 
    "bait": "בית"
  },
  "extraction": {
    "method": "precise_word_level",
    "exact_match_only": true,
    "strip_punctuation": true,
    "pad_to_duration": true,
    "center_word": true
  }
}
```

## Setup

### Prerequisites

1. **Python 3.8+** with pip
2. **FFmpeg** installed and available in PATH
3. **HuggingFace account** with access token

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Create `secret.json` with your HuggingFace credentials:
```json
{
  "huggingface": {
    "token": "your_huggingface_token_here"
  }
}
```

3. Get your HuggingFace token from: https://huggingface.co/settings/tokens

## Usage

### 1. Download Dataset

```bash
python download_ivrit_dataset.py
```

This script:
- Authenticates with HuggingFace
- Downloads the complete `ivrit-ai/crowd-recital` dataset
- Updates the dataset path in `config.json`

### 2. Analyze Vocabulary (Optional but Recommended)

```bash
python analyze_vocabulary.py
```

This creates comprehensive vocabulary statistics:
- Word frequency analysis
- Most/least common words
- Statistical insights about the dataset

### 3. Extract Target Words

Choose one of three extraction methods:

#### Basic Extraction
```bash
python extract_all_words.py
```

#### Parallel Processing (Recommended)
```bash
python extract_all_words_parallel.py
```

#### Optimized Extraction (Best Performance)
```bash
python extract_all_words_optimized.py
```

### Extraction Methods Comparison

| Method | Features | Performance | Use Case |
|--------|----------|-------------|----------|
| `extract_all_words.py` | Basic extraction, single-threaded | Slow | Simple extraction |
| `extract_all_words_parallel.py` | Multi-threaded processing | Fast | Large datasets |
| `extract_all_words_optimized.py` | Vocabulary filtering, smart unknown selection | Fastest | Production use |

## Output

### Audio Files
- **Format**: 16kHz, 16-bit, mono WAV files
- **Duration**: Exactly 1 second (padded or trimmed)
- **Organization**: Separate folders for each word category

### Generated Files
- `vocabulary_analysis.json`: Complete vocabulary statistics
- `word_counts.json`: Simple word frequency dictionary
- `*_extraction_summary.json`: Extraction results and statistics

### Example Output Structure
```
data/
├── shalom/           # 50+ audio files
├── lehitraoot/       # 30+ audio files  
├── bait/             # 500+ audio files
├── unknown/          # 2000+ other words
└── *.json           # Analysis and summary files
```

## Performance

### System Requirements
- **RAM**: 8GB+ recommended
- **CPU**: Multi-core recommended for parallel processing
- **Storage**: 10GB+ for complete dataset
- **Network**: Stable internet for initial download

### Processing Times (Approximate)
- **Dataset Download**: 5-15 minutes (depends on connection)
- **Vocabulary Analysis**: 2-5 minutes (parallel processing)
- **Word Extraction**: 10-30 minutes (depends on dataset size)

## Features

### Advanced Audio Processing
- **Precise Word Timing**: Uses word-level alignments for exact extraction
- **Audio Normalization**: Consistent sample rate and duration
- **Error Handling**: Robust retry logic for failed extractions
- **Format Conversion**: Automatic .mka to .wav conversion

### Parallel Processing
- **Multi-threading**: Utilizes all CPU cores (minus 1)
- **Thread-safe Operations**: Safe concurrent processing
- **Progress Tracking**: Real-time extraction progress
- **Memory Management**: Efficient memory usage for large datasets

### Vocabulary Intelligence
- **Frequency Analysis**: Complete word frequency statistics
- **Smart Filtering**: Intelligent unknown word selection
- **Statistical Insights**: Comprehensive dataset analysis
- **Quality Metrics**: Extraction success/failure tracking

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Install FFmpeg and ensure it's in your PATH
   - Windows: Download from https://ffmpeg.org/download.html

2. **HuggingFace authentication failed**
   - Check your token in `secret.json`
   - Ensure token has dataset access permissions

3. **Memory errors during extraction**
   - Use the optimized extraction script
   - Reduce parallel workers in configuration

4. **Slow processing**
   - Use parallel or optimized extraction methods
   - Ensure sufficient RAM and CPU cores

### Error Handling

The extraction scripts include comprehensive error handling:
- **Retry Logic**: Automatic retry for failed operations
- **Timeout Protection**: Prevents hanging on problematic files
- **Progress Logging**: Detailed progress and error reporting
- **Graceful Degradation**: Continues processing despite individual failures

## Integration

This extracted data integrates with the main wake word detection pipeline:

1. **Audio Processing**: Files are ready for model training
2. **Data Augmentation**: Compatible with augmentation scripts
3. **Model Training**: Direct input for TensorFlow/Keras models
4. **ESP32 Deployment**: Compatible with embedded model conversion

## License

This extraction tool is part of the Hebrew wake word detection project. The underlying `ivrit-ai/crowd-recital` dataset has its own licensing terms - please review before use.

## Contributing

When modifying extraction scripts:
- Maintain backward compatibility with existing config
- Add comprehensive error handling
- Include progress logging
- Test with various dataset sizes
- Update documentation for new features
