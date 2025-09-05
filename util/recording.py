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
SAMPLE_RATE = 16000  # 16 kHz
BIT_DEPTH_INPUT = 32 # I2S.read() returns 32-bit data
BIT_DEPTH_OUTPUT = 16# We will save as 16-bit for compatibility
DURATION = 5         # seconds
GAIN = 500           # A gain factor to amplify the audio signal. I have increased this value. Adjust as needed.

# Output file name
OUTPUT_WAV_FILE = 'recording.wav'

# --- Main script ---
def main():
    """
    Reads audio data from the serial port for a fixed duration and saves it as a WAV file.
    """
    print(f"Connecting to serial port {SERIAL_PORT}...")
    try:
        # Open the serial port
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
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
    
    # Read data line by line from the serial port
    try:
        while len(recorded_samples) < num_samples:
            if time.time() - start_time > DURATION + 2: # Add a small buffer for reading
                print("Recording timeout. Stopping.")
                break
                
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    # Convert the string to a 32-bit integer sample
                    sample = int(line)
                    # Apply a gain factor to the sample to make the audio louder
                    amplified_sample = int(sample * GAIN)
                    recorded_samples.append(amplified_sample)
                except (ValueError, IndexError):
                    # Ignore lines that are not valid integers
                    pass
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
    
    # Convert the 32-bit integer samples to 16-bit samples
    # This is done by scaling the values down
    max_val_32bit = 2**31 - 1
    max_val_16bit = 2**15 - 1
    
    samples_32bit_numpy = np.array(recorded_samples, dtype=np.int32)
    # Clamp values to prevent overflow during conversion
    samples_32bit_clamped = np.clip(samples_32bit_numpy, -max_val_16bit * (max_val_32bit / max_val_16bit), max_val_16bit * (max_val_32bit / max_val_16bit))
    
    # Scale down to 16-bit range
    samples_16bit_numpy = (samples_32bit_clamped / (max_val_32bit / max_val_16bit)).astype(np.int16)
    
    print(f"Saving data to {OUTPUT_WAV_FILE} as 16-bit audio...")
    try:
        with wave.open(OUTPUT_WAV_FILE, 'w') as wav_file:
            # Set WAV file parameters for 16-bit output
            wav_file.setnchannels(1)  # Mono audio
            wav_file.setsampwidth(BIT_DEPTH_OUTPUT // 8) # 2 bytes for 16-bit
            wav_file.setframerate(SAMPLE_RATE)
            
            # Write the 16-bit samples
            wav_file.writeframes(samples_16bit_numpy.tobytes())
            
        print(f"File '{OUTPUT_WAV_FILE}' saved successfully!")
    except Exception as e:
        print(f"An error occurred while writing the WAV file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
