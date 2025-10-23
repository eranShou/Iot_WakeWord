# ESP32 Audio Recorder - Stream to PC

Record 5-second audio samples directly from **XIAO ESP32S3** to your PC via USB. **No SD card or external microphone required!**

## ✨ Features

- 📡 Records audio and streams directly to PC via USB/Serial
- 💾 Saves as standard WAV files on your computer
- 🚀 Automatic port detection
- 📊 Real-time progress indicator
- 🔁 Record multiple samples easily
- ⏱️ Automatic timestamped filenames
- 🎯 Uses XIAO ESP32S3 built-in microphone
- 🔌 **NO WIRING NEEDED!**
- ⏰ **Custom recording duration (1-60 seconds)**

## 🔧 Hardware Requirements

- **Seeed Studio XIAO ESP32S3** (with built-in microphone)
- **USB-C Cable** (for data, not just power)

**That's it!** No SD card, no external microphone! 🎉

## 🎤 Built-in Microphone

The XIAO ESP32S3 has a built-in **MSM261D3526H1CPM** PDM microphone.

**✅ NO WIRING REQUIRED** - Just plug in USB and upload the sketch!

The microphone is located on the board (small hole on the side). Speak towards this hole for best results.

## 🚀 Setup Instructions

### 1. Install Python Requirements

```bash
pip install pyserial
```

### 2. Upload Arduino Sketch

1. Open `audio_recorder_esp32_pc.ino` in Arduino IDE
2. **Install ESP32 Board Manager 3.3.0:** `Tools` → `Board` → `Boards Manager` → Search "ESP32" → Install **version 3.3.0**
3. Select board: `Tools` → `Board` → `ESP32 Arduino` → **`XIAO_ESP32S3``
4. **Enable PSRAM:** `Tools` → `PSRAM` → **`OPI PSRAM`** (required for audio recording)
5. Select the correct port: `Tools` → `Port` → (your COM port)
6. Click Upload ⬆️ (no need to hold BOOT button on XIAO!)

### 3. Run the Python Script

```bash
python record_audio.py
```

The script will:
- Automatically detect your ESP32 (or ask you to select the port)
- Ask for output directory (default: `recordings/`)
- Connect to ESP32 and start recording

## 📝 Usage

### Basic Usage

```bash
python record_audio.py
```

The script will guide you through:
1. Detecting/selecting the ESP32 COM port
2. Choosing output directory
3. **Setting custom recording duration (1-60 seconds)**
4. Recording audio
5. Option to record more samples

### Example Output

```
==================================================
ESP32 Audio Recorder - PC Client
==================================================
Scanning for ESP32...
Found potential ESP32 at: COM3 (USB-SERIAL CH340)

Output directory (press Enter for 'recordings'): 

Recording duration in seconds (1-60, press Enter for 5): 10

Connecting to COM3 at 921600 baud...
Waiting for ESP32 to initialize...
ESP32: READY
ESP32: I2S_OK
ESP32: WAITING

==================================================
ESP32 Ready! Starting recording for 10 seconds...
==================================================
ESP32: RECORDING_START
ESP32: SAMPLE_RATE:16000
ESP32: BITS:16
ESP32: CHANNELS:1
ESP32: DATA_SIZE:320000
Receiving 320000 bytes of audio data...
(ESP32 is recording for 10 seconds, please wait...)
Start marker received ✓
Progress: 10%
Progress: 20%
...
Progress: 100%
End marker received ✓
Received 320000 bytes
Saving to recordings/recording_20251011_143052.wav...
✓ File saved successfully!
  Size: 320,044 bytes
  Duration: 10.00 seconds
  Sample Rate: 16000 Hz
  Bit Depth: 16 bits
  Channels: 1

✓ Recording complete: recordings/recording_20251011_143052.wav

Record another? (y/n): 
```

## ⚙️ Configuration

### Custom Recording Duration

**NEW FEATURE:** You can now set custom recording duration (1-60 seconds) directly from the Python script!

**Interactive Mode:**
```bash
python record_audio.py
# When prompted: "Recording duration in seconds (1-60, press Enter for 5): 15"
```

**Programmatic Usage:**
```python
from record_audio import record_audio
record_audio("COM3", "recordings", duration=15)  # 15 seconds
```

**Legacy Method (Arduino sketch):**
```cpp
#define DEFAULT_RECORD_TIME 5  // Default seconds
#define MIN_RECORD_TIME 1     // Minimum seconds  
#define MAX_RECORD_TIME 60    // Maximum seconds
```

### Change Sample Rate

```cpp
#define SAMPLE_RATE 16000  // Try 8000, 22050, 44100, etc.
```

### Built-in Microphone Configuration

The sketch is configured to use the XIAO ESP32S3's built-in PDM microphone:

```cpp
#define I2S_MIC_SERIAL_CLOCK GPIO_NUM_42  // PDM CLK (built-in)
#define I2S_MIC_SERIAL_DATA GPIO_NUM_41   // PDM DATA (built-in)
```

**Note:** These are internal connections - no external wiring needed!

### Change Baud Rate

If you experience data corruption, try a lower baud rate:

**Arduino sketch:**
```cpp
Serial.begin(921600);  // Try 460800 or 115200
```

**Python script:**
```python
BAUD_RATE = 921600  # Match the Arduino sketch
```

## 🎵 Recording Specifications

**Default Settings:**
- **Sample Rate:** 16,000 Hz (optimal for speech)
- **Bit Depth:** 16-bit
- **Channels:** Mono (1 channel)
- **Duration:** 5 seconds (customizable: 1-60 seconds)
- **Format:** WAV (PCM)
- **File Size:** ~160 KB per 5-second recording (~320 KB for 10 seconds)

## 🐛 Troubleshooting

### ESP32 Not Detected

**Windows:**
- Install CH340 or CP210x drivers (depending on your ESP32)
- Check Device Manager for COM port

**Linux:**
- Check: `ls /dev/ttyUSB*` or `ls /dev/ttyACM*`
- Add user to dialout group: `sudo usermod -a -G dialout $USER`
- May need to logout/login

**macOS:**
- Check: `ls /dev/cu.*`
- Install drivers if needed

### Recording Fails or Corrupted Audio

1. **Try lower baud rate** (460800 or 115200)
2. **Check USB cable** - must support data transfer
3. **Close Serial Monitor** in Arduino IDE before running script
4. **Restart ESP32** (press reset button)
5. **Check microphone connections**

### Timeout Errors

- Make sure only one program is using the serial port
- Close Arduino Serial Monitor
- Try increasing timeout in Python script:
  ```python
  TIMEOUT = 20  # Increase from 10
  ```

### Silent Recording / No Audio

- Make sure you're speaking towards the microphone hole on the XIAO board
- The built-in mic is less sensitive than external mics - speak louder or closer
- Tap the board gently or clap near it to test
- Re-upload the sketch with correct board selected (XIAO_ESP32S3)
- Check Serial Monitor shows "I2S_OK" message

### Permission Denied (Linux)

```bash
sudo usermod -a -G dialout $USER
# Logout and login again
```

## 📊 File Management

### Output Files

Files are automatically saved with timestamps:
```
recordings/
  ├── recording_20251011_143052.wav
  ├── recording_20251011_143125.wav
  └── recording_20251011_143158.wav
```

### Organize by Wake Word

For wake word training, organize recordings:

```bash
mkdir -p data/raw/Shalom
mkdir -p data/raw/Lehitraot
mkdir -p data/raw/Background

# Move recordings to appropriate folders
mv recordings/recording_*.wav data/raw/Shalom/
```

Then use the preprocessing pipeline:
```bash
python scripts/preprocess.py
```

## 🔬 Advanced Options

### Record Specific Number of Samples

Modify the Python script to loop automatically:

```python
for i in range(10):  # Record 10 samples
    print(f"\nRecording {i+1}/10...")
    record_audio(port, f"data/raw/WakeWord1")
    time.sleep(1)  # Pause between recordings
```

### Change Output Format

The script saves as WAV by default. To convert to other formats, use ffmpeg:

```bash
# Convert to MP3
ffmpeg -i recording.wav -acodec mp3 recording.mp3

# Convert to FLAC (lossless)
ffmpeg -i recording.wav recording.flac
```

## 💡 Tips

1. **Record in quiet environment** for best results
2. **Speak clearly** and at consistent distance from mic
3. **Collect diverse samples** (different speakers, tones, speeds)
4. **Include background noise samples** for robust model training
5. **Use consistent distance** from microphone (~20-30cm)

## 🔗 Integration with Wake Word Pipeline

These recordings are fully compatible with your training pipeline:

1. **Record samples:**
   ```bash
   python record_audio.py
   ```

2. **Organize into directories:**
   ```
   data/raw/
     ├── Shalom/
     ├── Lehitraot/
     └── Background/
   ```

3. **Preprocess:**
   ```bash
   python scripts/preprocess.py
   ```

4. **Train model:**
   ```bash
   python scripts/train_model.py
   ```

## 📈 Performance

- **Transfer speed:** ~920 Kbps (high-speed USB)
- **Recording latency:** ~5-6 seconds total
- **Memory usage (ESP32):** ~1-2 KB RAM
- **Memory usage (PC):** ~1 MB per recording

## ⚡ Why Stream to PC?

**Advantages over SD card:**
- ✅ No SD card module needed (saves cost & space)
- ✅ Files immediately accessible on PC
- ✅ No need to remove SD card
- ✅ Easy to organize and rename files
- ✅ Faster workflow for training data collection

**Disadvantages:**
- ❌ Must be connected to PC
- ❌ Slightly more complex setup

## 📄 License

Part of the ESP32 Wake Word Detection project.

