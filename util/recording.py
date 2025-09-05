import serial
import wave
import struct
import time
import sys
import numpy as np

# --- Configuration ---
# The serial port of your ESP32.
# On Windows, this will look like 'COM3' or 'COM4'
# On macOS/Linux, it will look like '/dev/ttyUSB0' or '/dev/cu.usbserial-...'
# You may need to change this to match your system.
SERIAL_PORT = 'COM4'

# The baud rate must match the one in your Arduino code.
BAUD_RATE = 115200

# Audio settings from your Arduino code
SAMPLE_RATE = 16000  # 16 kHz, a good balance for speech
BIT_DEPTH_INPUT = 32 # I2S.read() returns 32-bit data
BIT_DEPTH_OUTPUT = 32# We will save as 32-bit for wider range and less distortion
DURATION = 5         # seconds
GAIN = 500           # A gain factor to amplify the audio signal. Adjust as needed.

# Filter settings
FILTER_CUTOFF_HZ = 150 # High-pass filter cutoff frequency in Hz to remove low-end noise

# Output file name
OUTPUT_WAV_FILE = 'recording.wav'

# --- Digital Signal Processing ---
def iir_filter(data, cutoff, sample_rate):
    """
    Applies a simple IIR high-pass filter to the audio data.
    This helps to remove low-frequency noise like hums.
    """
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff / nyquist
    alpha = 1 - normalized_cutoff
    
    y = np.zeros_like(data)
    y[0] = data[0]
    
    for i in range(1, len(data)):
        y[i] = alpha * y[i-1] + alpha * (data[i] - data[i-1])
        
    return y

# --- Main script ---
def main():
    """
    Reads audio data from the serial port for a fixed duration and saves it as a WAV file.
    """
    print(f"Connecting to serial port {SERIAL_PORT}...")
    try:
        # Open the serial port
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=None) # timeout is set to None for binary data
    except serial.SerialException as e:
        print(f"Error: Could not open serial port {SERIAL_PORT}. Please check the port name and if the device is connected.")
        print(f"Error details: {e}")
        sys.exit(1)

    print(f"Successfully connected to {ser.name}")
    print(f"Recording for {DURATION} seconds at {SAMPLE_RATE} Hz...")
    
    # Calculate the number of samples to read
    num_samples = int(SAMPLE_RATE * DURATION)
    recorded_samples = []
    
    # Get start time to track duration
    start_time = time.time()
    
    # Read raw binary data from the serial port
    try:
        while len(recorded_samples) < num_samples:
            if time.time() - start_time > DURATION + 2: # Add a small buffer for reading
                print("Recording timeout. Stopping.")
                break
            
            # Read 4 bytes at a time for each 32-bit integer sample
            raw_sample = ser.read(4)
            if len(raw_sample) == 4:
                # Unpack the raw bytes into a signed 32-bit integer
                sample = struct.unpack('<i', raw_sample)[0]
                
                # Apply a gain factor to the sample to make the audio louder
                amplified_sample = int(sample * GAIN)
                recorded_samples.append(amplified_sample)

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        ser.close()
        print("Serial port closed.")

    # Check if any samples were recorded
    if not recorded_samples:
        print("No samples were recorded. The serial monitor might be empty.")
        print("Please ensure your ESP32 is running and sending data.")
        sys.exit(1)

    print(f"Recorded {len(recorded_samples)} samples.")
    
    # Apply the high-pass filter to the raw data
    print("Applying high-pass filter to reduce noise...")
    # Convert recorded_samples to a numpy array for filtering
    samples_numpy = np.array(recorded_samples, dtype=np.int32)
    filtered_samples = iir_filter(samples_numpy, FILTER_CUTOFF_HZ, SAMPLE_RATE)

    print(f"Saving data to {OUTPUT_WAV_FILE} as 32-bit audio...")
    try:
        with wave.open(OUTPUT_WAV_FILE, 'w') as wav_file:
            # Set WAV file parameters for 32-bit output
            wav_file.setnchannels(1)  # Mono audio
            wav_file.setsampwidth(BIT_DEPTH_OUTPUT // 8) # 4 bytes for 32-bit
            wav_file.setframerate(SAMPLE_RATE)
            
            # Write the 32-bit samples directly
            wav_file.writeframes(filtered_samples.tobytes())
            
        print(f"File '{OUTPUT_WAV_FILE}' saved successfully!")
    except Exception as e:
        print(f"An error occurred while writing the WAV file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
