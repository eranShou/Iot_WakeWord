# ESP32 Model Upload Tool

This tool helps you upload your trained Hebrew wake word detection model to your ESP32 microcontroller.

## Files in this directory

- `model_converter.py` - Python script to convert TensorFlow Lite model data
- `convert_and_upload.bat` - Windows batch script to run the converter
- `model_data.h` - Generated model data file (created by the converter)
- `tflm.ino` - ESP32 Arduino sketch for wake word detection

## Quick Start
Manual Python Execution

```bash
# Navigate to ESP32 directory
cd ESP32

#Run in tensorflow_wake_word_detection\training\models\
python model_conversion.py --model_path models/hebrew_wake_word_model_cnn.h5

# Run the converter 
python model_converter.py
```

## Upload to ESP32

After running the converter, follow these steps:

### Using Arduino IDE

1. Open Arduino IDE
2. Load the `tflm.ino` file
3. Select your ESP32 board (Tools → Board → ESP32 Dev Module)
4. Select the correct COM port (Tools → Port)
5. Click **Upload** (→ button)

### Using PlatformIO

If you have PlatformIO installed:

```bash
# Navigate to ESP32 directory
cd ESP32

# Upload
pio run -t upload
```

### Using Arduino CLI

If you have Arduino CLI installed:

```bash
# Compile
arduino-cli compile --fqbn esp32:esp32:esp32 ESP32

# Upload (replace COM_PORT with your actual port)
arduino-cli upload -p COM_PORT --fqbn esp32:esp32:esp32 ESP32
```

## Model Information

The converted model includes:
- **Model Size**: ~4MB of model data
- **Input**: 13 MFCC features (processed audio)
- **Output**: 4 classes (shalom, unknown, noise, lehit)
- **Quantization**: INT8 for efficient ESP32 execution

## Troubleshooting

### Python Not Found
- Install Python 3.x from https://python.org
- Make sure Python is added to your PATH

### Model File Not Found
- Ensure your trained model exists at:
  `../tensorflow_wake_word_detection/training/models/hebrew_wake_word_model_cnn_int8.cc`

### ESP32 Upload Issues
- Check that your ESP32 is properly connected
- Verify the correct COM port is selected
- Ensure no other programs are using the serial port
- Try pressing the ESP32 boot button during upload

### Model Not Working
- Check the serial output for error messages
- Verify the model was trained correctly
- Ensure MFCC parameters match between training and inference
- Check that the audio input is working (microphone, I2S setup)

## Technical Details

The converter extracts the model data from the TensorFlow Lite C++ file and formats it for ESP32 use. The model data is stored as a byte array that gets loaded into TensorFlow Lite Micro runtime on the ESP32.

### Model Architecture
- CNN-based wake word detection
- Optimized for Hebrew language wake words
- INT8 quantization for memory efficiency
- Designed for real-time inference on microcontrollers

## Requirements

- Python 3.x
- Trained model file (`hebrew_wake_word_model_cnn_int8.cc`)
- ESP32 development board
- Arduino IDE or PlatformIO (optional, for uploading)

## License

This tool is part of the Hebrew Wake Word Detection project.
