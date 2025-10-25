# Wake Word Detection Statistics Tool

A Python tool for collecting detailed statistics on wake word detection performance from the ESP32-S3 standalone system. Monitors serial output in real-time, tracks detections with distance categories, measures false positives, and exports comprehensive results to Excel.

## Features

- **Real-time Serial Monitoring**: Connects to ESP32-S3 via USB serial port
- **Session Management**: Organize tests by wake word and distance (close/far)
- **False Positive Tracking**: Mark incorrect detections during testing
- **Confidence Score Tracking**: Records confidence scores for all detections
- **Detailed Statistics**: Calculates detection rates, accuracy, and confidence metrics
- **Excel Export**: Saves results to timestamped Excel files with multiple sheets
- **Color-coded Output**: Visual feedback for correct detections and errors
- **Automatic Port Detection**: Auto-detects ESP32-S3 COM port

## Setup

### Prerequisites

- Python 3.7 or higher
- ESP32-S3 with wake word detection firmware loaded
- USB cable to connect ESP32-S3 to computer
- Seeed Studio XIAO ESP32S3 (or compatible)

### Installation

1. Navigate to the statistics directory:
```bash
cd xiao-esp32s3-wake-word/statistics
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

This will install:
- `pyserial` - Serial communication
- `openpyxl` - Excel file generation
- `pandas` - Data manipulation
- `colorama` - Colored terminal output

## Usage

### Basic Usage

1. Connect your ESP32-S3 to your computer via USB
2. Upload and run the `esp32s3_standalone.ino` firmware
3. Run the statistics tool:
```bash
python wake_word_statistics.py
```

4. Follow the interactive prompts:
   - Auto-detect or manually enter COM port
   - Enter distance category (close/far)
   - Select wake word to test (lehitraoot/shalom/both)
   - Enter expected number of times to say the wake word
   - Say the wake words while monitoring detections
   - Press `f` to mark any false positives
   - Press `q` to end the session
   - Repeat for additional sessions
   - Export results to Excel

### Interactive Commands

During monitoring:
- **`f`** - Mark the last detection as a false positive
- **`q`** - End the current session

After session:
- **`y`** - Start another session
- **`n`** - Exit and export results to Excel

### Excel Output

Results are saved in the `results/` directory with timestamped filenames:
```
results/wake_word_stats_20251025_143000.xlsx
```

The Excel file contains three sheets:

#### 1. Summary Sheet
Aggregated statistics grouped by wake word and distance:
- Number of sessions
- Total expected vs. detected
- False positives and true positives
- Detection rate percentage
- Average confidence

#### 2. Sessions Sheet
Detailed per-session information:
- Session ID and wake word
- Distance category
- Expected vs. actual detections
- False positive count
- Detection rate and average confidence
- Session start time and duration

#### 3. Detections Sheet
Individual detection records:
- Detection number
- Session ID and parameters
- Detected class
- Confidence score
- All class confidence scores
- False positive flag
- Timestamp

## Example Workflow

### Testing "shalom" at Close Distance

```bash
$ python wake_word_statistics.py

Wake Word Detection Statistics Tool
===============================================================

Auto-detected port: COM3
Use this port? (y/n): y
Connected to COM3 at 921600 baud

==============================================================
Session 1 Configuration
==============================================================

Enter distance category (close/far): close
Which wake word are you testing? (lehitraoot/shalom/both): shalom
How many times will you say it? 10

[Session 1] Started: shalom at close distance
Expected count: 10
Press 'f' to mark last detection as false positive
Press 'q' to end session

Monitoring... (Press 'f' for false positive, 'q' to end session)

[#1] shalom - Confidence: 0.847 [CORRECT]
[#2] shalom - Confidence: 0.892 [CORRECT]
[#3] shalom - Confidence: 0.756 [CORRECT]
[#4] unknown - Confidence: 0.723 [WRONG CLASS]
[#5] shalom - Confidence: 0.834 [CORRECT]
...

Press 'q' to end session
[#15] shalom - Confidence: 0.901 [CORRECT]

============================================================
Session 1 Summary
============================================================
Wake word: shalom
Distance: close
Expected: 10
Actual detections: 15
False positives: 2
True positives: 13
Detection rate: 130.0%
Average confidence: 0.824
============================================================

Start another session? (y/n): n

Exporting results to Excel...
Results saved to: results/wake_word_stats_20251025_143000.xlsx
Exiting. Goodbye!
```

## Troubleshooting

### Port Not Detected
If automatic port detection fails:
- Manually enter the COM port (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)
- Check Device Manager (Windows) or `lsusb` (Linux) to find the port
- Ensure only one USB-to-serial device is connected

### Connection Errors
- Ensure ESP32-S3 is powered and firmware is running
- Close other serial monitor applications (Arduino IDE Serial Monitor)
- Try disconnecting and reconnecting the USB cable
- Check that baud rate is 921600 (defined in `config.h`)

### No Detections Received
- Verify that wake word detection output is enabled in `esp32s3_standalone.ino`
- Check serial connection is active and data is flowing
- Look for "*** WAKE WORD DETECTED #X ***" messages in raw serial output

### Excel Export Fails
- Ensure `openpyxl` package is installed: `pip install openpyxl`
- Check write permissions in the `results/` directory
- Verify Excel is not currently open with the same filename

## Configuration

### Serial Settings
Default settings match `esp32s3_standalone/config.h`:
- Baud Rate: 921600
- Parity: None
- Stop Bits: 1
- Data Bits: 8

### Detection Parsing
The tool parses these output formats from ESP32-S3:

```
*** WAKE WORD DETECTED #15 ***
Class: shalom, Confidence: 0.892 (threshold: 0.700)
All confidences: [0.123, 0.892, 0.021, 0.064]
```

### Wake Word Classes
From `model_config.h`:
- Index 0: "lehitraoot"
- Index 1: "shalom"
- Index 2: "background"
- Index 3: "unknown"

## Data Analysis Tips

### Detection Rate Analysis
- **>100%**: Model detected more times than expected (possible false positives)
- **<100%**: Model missed some detections (possibly too strict threshold)
- Use false positive marking to get accurate true positive counts

### Confidence Score Trends
- Track average confidence over sessions
- Lower confidence at far distances is expected
- Sudden drops may indicate environmental changes

### False Positive Patterns
- Check which classes are incorrectly detected
- "unknown" class often captures non-wake-word speech
- High false positive rate may indicate threshold adjustment needed

## Advanced Usage

### Multiple Sessions Comparison
Run multiple sessions with different parameters to compare:
- Performance at different distances
- Different wake words
- Different environments
- Before/after model retraining

### Batch Testing
To test systematically:
1. Create a test plan with specific parameters
2. Run multiple sessions without closing the tool
3. Export single Excel file with all sessions
4. Analyze aggregated results in Summary sheet

### Custom Export
Modify the Excel export functions in `wake_word_statistics.py` to:
- Add custom charts or visualizations
- Include additional metrics
- Format for specific analysis tools
- Integrate with other data sources

## Technical Details

### Class Structure
- **Detection**: Single wake word detection event
- **Session**: Test session with parameters and detections
- **StatisticsCollector**: Manages sessions and exports
- **SerialMonitor**: Handles ESP32-S3 communication

### Memory Management
- Sessions are stored in memory during execution
- Data is only written to disk on exit
- Large sessions may consume significant RAM

### Error Handling
- Graceful handling of serial disconnections
- Keyboard interrupt saves current session data
- Invalid input validation with retry prompts

## License

Part of the XIAO ESP32S3 Wake Word Detection project.
