# Audio Processing Scripts

This directory contains scripts to process wake word recordings from the ESP32 audio recorder.

## Files

- `copy_recordings.py` - Copies WAV files from `../audio_recorder_esp32_pc/recordings/` to `data/`
- `audio_split.py` - Splits recordings into individual word segments using silence detection
- `requirements.txt` - Python dependencies
- `data/` - Output directory with processed audio files

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Copy Recordings

```bash
python copy_recordings.py
```

This will:
- Scan `../audio_recorder_esp32_pc/recordings/` for subfolders (e.g., `lehitraoot`, `shaloom`)
- Create corresponding folders in `data/`
- Copy all WAV files to the new location

### 3. Split Audio

```bash
python audio_split.py
```

This will:
- Process each WAV file in `data/` subfolders
- Split recordings using silence detection
- Save segments as `{word_name}_001.wav`, `{word_name}_002.wav`, etc.
- Move original files to `backup/` subfolders

## Output Structure

```
audio_prosser/
├── copy_recordings.py
├── audio_split.py
├── requirements.txt
└── data/
    ├── lehitraoot/
    │   ├── lehitraoot_001.wav
    │   ├── lehitraoot_002.wav
    │   ├── ...
    │   └── backup/
    │       ├── recording_20251015_134801.wav
    │       └── ...
    └── shaloom/
        ├── shaloom_001.wav
        ├── shaloom_002.wav
        ├── ...
        └── backup/
            └── ...
```

## Configuration

The silence detection parameters in `audio_split.py` can be adjusted:

- `min_silence_len`: Minimum silence duration to detect splits (default: 300ms)
- `silence_thresh`: Silence threshold in dBFS (default: -40)
- `keep_silence`: Silence to preserve around segments (default: 100ms)

## Features

- Automatic folder structure creation
- Continuous numbering across multiple recording files
- Backup of original recordings
- Progress reporting during processing
- Error handling for corrupted files
