"""
ESP32 Audio Recorder - PC Client
Receives audio data from ESP32 via USB/Serial and saves as WAV file
"""

import serial
import struct
import wave
import time
import os
from datetime import datetime

# Constants (matching mic_config.h)
START_MARKER = 0xAA55AA55
END_MARKER = 0x55AA55AA
BAUD_RATE = 921600
TIMEOUT = 10

# Audio Configuration
SAMPLE_RATE = 16000
BITS_PER_SAMPLE = 16
NUM_CHANNELS = 1
BYTES_PER_SAMPLE = 2

# Recording Limits
MIN_RECORD_TIME = 1
MAX_RECORD_TIME = 60
DEFAULT_RECORD_TIME = 15

# Gain Configuration
MIN_GAIN = 0.1
MAX_GAIN = 20.0
DEFAULT_GAIN = 8.0

def find_esp32_port():
    """Try to find the ESP32 COM port automatically"""
    import serial.tools.list_ports
    
    print("Scanning for ESP32...")
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        # Look for common ESP32 USB chip identifiers
        if any(x in port.description.lower() for x in ['ch340', 'cp210', 'usb-serial', 'uart', 'silicon labs']):
            print(f"Found potential ESP32 at: {port.device} ({port.description})")
            return port.device
    
    # Show all available ports
    if ports:
        print("\nAvailable ports:")
        for i, port in enumerate(ports):
            print(f"  [{i}] {port.device} - {port.description}")
        
        choice = input("\nEnter port number or full port name (e.g., COM3): ").strip()
        if choice.isdigit() and int(choice) < len(ports):
            return ports[int(choice)].device
        else:
            return choice
    
    return None

def wait_for_response(ser, expected, timeout=5):
    """Wait for a specific response from ESP32"""
    start_time = time.time()
    all_responses = []
    while time.time() - start_time < timeout:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"ESP32: {line}")
                all_responses.append(line)
                if expected in line:
                    return True
        time.sleep(0.01)  # Small delay to prevent tight loop
    
    # If we didn't find the expected message but got some responses, 
    # check if any of them contain the expected string
    for response in all_responses:
        if expected.lower() in response.lower():
            return True
    
    return False

def receive_audio_data(ser, expected_size, duration=5):
    """Receive binary audio data from ESP32"""
    print(f"Receiving {expected_size} bytes of audio data...")
    print(f"(ESP32 is recording for {duration} seconds, please wait...)")
    
    data = bytearray()
    start_time = time.time()
    bytes_received = 0  # Initialize bytes_received variable
    last_countdown = 0  # Track last countdown to avoid spam
    
    # Wait for start marker (recording takes duration+ seconds)
    print("Waiting for start marker...")
    
    # Give ESP32 time to start recording, but don't clear buffer yet
    # Scale delay with duration - longer recordings need more setup time
    setup_delay = max(1.0, duration * 0.2)  # At least 1 second, or 20% of duration
    time.sleep(setup_delay)
    
    marker_bytes = bytearray()
    wait_start = time.time()
    marker_found = False
    
    # Calculate dynamic timeout for start marker based on recording duration
    # Longer recordings need more time to start
    marker_timeout = max(15, duration + 10)  # At least 15 seconds, or duration + 10
    
    # Try to find the start marker with more robust approach
    while time.time() - wait_start < marker_timeout:
        if ser.in_waiting:
            chunk = ser.read(ser.in_waiting)
            marker_bytes.extend(chunk)
            
            
            # Look for start marker in the accumulated data
            for i in range(len(marker_bytes) - 3):
                if len(marker_bytes) >= i + 4:
                    marker = struct.unpack('<I', marker_bytes[i:i+4])[0]
                    if marker == START_MARKER:
                        # Remove everything before the marker
                        marker_bytes = marker_bytes[i+4:]
                        marker_found = True
                        break
            
            if marker_found:
                break
        
        time.sleep(0.01)
    
    if not marker_found:
        print(f"Error: Could not find start marker in stream after {marker_timeout}s")
        print(f"This might happen with longer recordings ({duration}s).")
        print("Try:")
        print("  1. Reducing recording duration")
        print("  2. Checking ESP32 power supply")
        print("  3. Using a shorter USB cable")
        return None
    
    print("Start marker received ✓")
    
    # Start with any remaining bytes from marker search
    data.extend(marker_bytes)
    bytes_received = len(marker_bytes)
    last_progress = 0
    last_data_time = time.time()
    
    while bytes_received < expected_size:
        # Check completion ratio for dynamic timeout and chunk size
        completion_ratio = bytes_received / expected_size
        
        # Use longer timeout when we're getting close to completion
        # Scale timeouts with recording duration for better reliability
        base_timeout = max(5, duration * 0.5)  # Base timeout scales with duration
        if completion_ratio > 0.9:
            # When we're past 90%, be much more patient
            timeout_duration = max(15, duration * 1.5)  # Scale with duration
        elif completion_ratio > 0.8:
            # When we're past 80%, be more patient
            timeout_duration = max(10, duration * 1.0)  # Scale with duration
        else:
            timeout_duration = base_timeout   # Base timeout for normal operation
        
        # Use smaller chunks when close to completion to grab whatever is available
        # Adjust chunk sizes based on data size and completion
        max_chunk = min(4096, expected_size // 100)  # Scale chunk size with data size
        if completion_ratio > 0.8:
            chunk_size = min(256, expected_size - bytes_received)  # Very small chunks near end
        elif completion_ratio > 0.6:
            chunk_size = min(512, expected_size - bytes_received)  # Small chunks
        else:
            chunk_size = min(max_chunk, expected_size - bytes_received)  # Scaled chunk size
        
        chunk = ser.read(chunk_size)
        
        if chunk:
            data.extend(chunk)
            bytes_received += len(chunk)
            last_data_time = time.time()  # Reset timeout when we get data
            
            # Show progress
            progress = int((bytes_received / expected_size) * 100)
            if progress >= last_progress + 10:
                print(f"Progress: {progress}%")
                last_progress = progress
            
            # Show countdown for last 5 seconds of recording
            elapsed_time = time.time() - start_time
            remaining_time = duration - elapsed_time
            if remaining_time <= 5 and remaining_time > 0:
                # Only show countdown once per second to avoid spam
                countdown_seconds = int(remaining_time) + 1
                if countdown_seconds != last_countdown:
                    print(f"Recording ends in {countdown_seconds} seconds...")
                    last_countdown = countdown_seconds
            
        else:
            # Check for timeout with dynamic duration
            if time.time() - last_data_time > timeout_duration:
                print(f"\nError: Timeout waiting for data (received {bytes_received}/{expected_size} bytes)")
                return None
            time.sleep(0.01)  # Small delay to prevent busy waiting
    
    # If we got at least 99% of the data, consider it successful
    if bytes_received >= expected_size * 0.99:
        print(f"Progress: 100%")
        print("🎙️ Recording complete!")
        
        # Try to read any remaining bytes
        while bytes_received < expected_size and ser.in_waiting:
            remaining = expected_size - bytes_received
            chunk = ser.read(remaining)
            if chunk:
                data.extend(chunk)
                bytes_received += len(chunk)
        
        # Read end marker
        time.sleep(0.1)  # Give time for end marker
        marker_bytes = ser.read(4)
        if len(marker_bytes) == 4:
            marker = struct.unpack('<I', marker_bytes)[0]
            if marker == END_MARKER:
                print("End marker received ✓")
        
        print(f"Received {len(data)} bytes ({100*len(data)/expected_size:.1f}%)")
        
        # Pad with silence if slightly short
        if len(data) < expected_size:
            padding = expected_size - len(data)
            data.extend(bytes(padding))  # Add zeros
            print(f"Padded {padding} bytes with silence")
        
        return bytes(data)
    
    print(f"Received {len(data)} bytes (not enough)")
    return None

def save_wav_file(filename, audio_data, sample_rate, bits_per_sample, channels):
    """Save audio data as WAV file"""
    print(f"Saving to {filename}...")
    
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bits_per_sample // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    
    file_size = os.path.getsize(filename)
    duration = len(audio_data) / (sample_rate * channels * (bits_per_sample // 8))
    
    print(f"✓ File saved successfully!")
    print(f"  Size: {file_size:,} bytes")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Sample Rate: {sample_rate} Hz")
    print(f"  Bit Depth: {bits_per_sample} bits")
    print(f"  Channels: {channels}")

def record_audio(port, output_dir="recordings", duration=5):
    """Main recording function"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Connect to ESP32
    print(f"\nConnecting to {port} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        print("Waiting for ESP32 to boot...")
        time.sleep(3)  # Wait longer for ESP32 to reset and boot
    except serial.SerialException as e:
        print(f"\n❌ Error: Could not open port {port}")
        print(f"Details: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("  1. Close Arduino IDE Serial Monitor (most common issue!)")
        print("  2. Close any other programs using this port")
        print("  3. Unplug and replug the ESP32")
        print("  4. Try a different USB port")
        print("  5. Restart your computer if issue persists")
        print("\n💡 TIP: Make sure ONLY this Python script is accessing the port")
        return False
    
    # Clear any pending data
    ser.reset_input_buffer()
    
    # Set larger buffer size for longer recordings
    if duration > 15:
        print(f"Setting up for long recording ({duration}s) - optimizing buffer...")
        # Increase buffer size for longer recordings
        ser.reset_input_buffer()
        time.sleep(0.5)  # Give time for buffer reset
    
    # Wait for ESP32 to be ready
    print("Waiting for ESP32 to initialize...")
    print("(Watching for READY, I2S_OK, WAITING messages...)")
    
    # Try to find WAITING message, but also accept READY or I2S_OK as signs it's working
    found_waiting = wait_for_response(ser, "WAITING", timeout=10)
    
    if not found_waiting:
        # Maybe we missed the initial messages, send a status command
        print("\nSending STATUS command to check if ESP32 is responsive...")
        ser.write(b"STATUS\n")
        time.sleep(0.5)
        
        # Check for any response
        if ser.in_waiting > 0:
            while ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"ESP32: {line}")
                    if "READY" in line or "WAITING" in line:
                        found_waiting = True
                        break
        
        if not found_waiting:
            print("\n❌ Error: ESP32 did not respond with expected messages.")
            print("\n🔧 TROUBLESHOOTING:")
            print("  1. Check if sketch is uploaded:")
            print("     - Open Arduino IDE")
            print("     - Select Tools → Board → XIAO_ESP32S3")
            print("     - Select Tools → Port → " + port)
            print("     - Click Upload (→) button")
            print("  2. Press RESET button on ESP32")
            print("  3. Run test_connection.py to diagnose")
            ser.close()
            return False
    
    print("\n" + "="*50)
    print(f"ESP32 Ready! Starting recording for {duration} seconds...")
    print("="*50)
    
    # Wait 3 seconds before starting recording
    print("Get ready! Recording starts in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("Recording NOW!")
    
    # Send record command with custom duration
    if duration == 5:
        # Use default command for 5 seconds (backward compatibility)
        ser.write(b"RECORD\n")
    else:
        # Send custom duration command
        command = f"RECORD:{duration}\n"
        ser.write(command.encode())
    
    # Wait for recording to start - scale timeout with duration
    start_timeout = max(5, duration * 0.5)  # At least 5 seconds, or half the recording duration
    if not wait_for_response(ser, "RECORDING_START", timeout=start_timeout):
        print("Error: Recording did not start")
        ser.close()
        return False
    
    # Read audio configuration
    config = {}
    for _ in range(4):
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        print(f"ESP32: {line}")
        if ':' in line:
            key, value = line.split(':', 1)
            config[key] = int(value)
    
    # Receive audio data
    sample_rate = config.get('SAMPLE_RATE', 16000)
    bits_per_sample = config.get('BITS', 16)
    channels = config.get('CHANNELS', 1)
    data_size = config.get('DATA_SIZE', 160000)
    
    audio_data = receive_audio_data(ser, data_size, duration)
    
    if audio_data:
        # Wait for completion message
        time.sleep(0.5)
        while ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"ESP32: {line}")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"recording_{timestamp}.wav")
        
        # Save WAV file
        save_wav_file(filename, audio_data, sample_rate, bits_per_sample, channels)
        print(f"\n✓ Recording complete: {filename}")
    else:
        print("\n✗ Recording failed")
    
    ser.close()
    return True

def adjust_audio_settings(ser):
    """Allow user to adjust gain and compression settings"""
    print("\n" + "="*40)
    print("AUDIO PROCESSING SETTINGS")
    print("="*40)
    
    # Check current status
    ser.write(b"GAIN_STATUS\n")
    time.sleep(0.5)
    ser.write(b"COMPRESSION_STATUS\n")
    time.sleep(0.5)
    
    while True:
        print("\nAudio Control Options:")
        print("1. Set gain level (0.1-20.0x)")
        print("2. Enable/disable gain")
        print("3. Set compression threshold (0.0-1.0)")
        print("4. Enable/disable compression")
        print("5. Check current settings")
        print("6. Continue to recording")
        
        choice = input("\nChoose option (1-6): ").strip()
        
        if choice == "1":
            try:
                gain = float(input(f"Enter gain level ({MIN_GAIN}-{MAX_GAIN}): "))
                if MIN_GAIN <= gain <= MAX_GAIN:
                    command = f"GAIN:{gain}\n"
                    ser.write(command.encode())
                    time.sleep(0.5)
                    print(f"Gain set to {gain}x")
                else:
                    print(f"Gain must be between {MIN_GAIN} and {MAX_GAIN}")
            except ValueError:
                print("Please enter a valid number")
        
        elif choice == "2":
            enable = input("Enable gain? (y/n): ").strip().lower()
            if enable == 'y':
                ser.write(b"GAIN_ON\n")
                print("Gain enabled")
            else:
                ser.write(b"GAIN_OFF\n")
                print("Gain disabled")
            time.sleep(0.5)
        
        elif choice == "3":
            try:
                threshold = float(input("Enter compression threshold (0.0-1.0): "))
                if 0.0 <= threshold <= 1.0:
                    command = f"COMPRESSION:{threshold}\n"
                    ser.write(command.encode())
                    time.sleep(0.5)
                    print(f"Compression threshold set to {threshold}")
                else:
                    print("Threshold must be between 0.0 and 1.0")
            except ValueError:
                print("Please enter a valid number")
        
        elif choice == "4":
            enable = input("Enable compression? (y/n): ").strip().lower()
            if enable == 'y':
                ser.write(b"COMPRESSION_ON\n")
                print("Compression enabled")
            else:
                ser.write(b"COMPRESSION_OFF\n")
                print("Compression disabled")
            time.sleep(0.5)
        
        elif choice == "5":
            ser.write(b"GAIN_STATUS\n")
            time.sleep(0.5)
            ser.write(b"COMPRESSION_STATUS\n")
            time.sleep(0.5)
        
        elif choice == "6":
            break
        
        else:
            print("Invalid choice")

def main():
    """Main entry point"""
    print("="*50)
    print("ESP32 Audio Recorder - PC Client")
    print("="*50)
    
    # Find ESP32 port
    port = find_esp32_port()
    
    if not port:
        print("\nNo ESP32 found. Please enter the COM port manually.")
        port = input("Port (e.g., COM3 or /dev/ttyUSB0): ").strip()
    
    if not port:
        print("Error: No port specified")
        return
    
    # Connect to ESP32 first to check gain settings
    print(f"\nConnecting to {port} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        print("Connected to ESP32!")
        time.sleep(2)  # Wait for ESP32 to boot
        
        # Clear any pending data
        ser.reset_input_buffer()
        
        # Ask if user wants to adjust audio settings
        adjust_audio = input("\nAdjust audio processing settings? (y/n, press Enter for 'n'): ").strip().lower()
        if adjust_audio == 'y':
            adjust_audio_settings(ser)
        
        ser.close()
        
    except serial.SerialException as e:
        print(f"\n❌ Error: Could not open port {port}")
        print(f"Details: {e}")
        return
    
    # Get output directory
    output_dir = input("\nOutput directory (press Enter for 'test'): ").strip()
    if not output_dir:
        output_dir = "test"
    
    # Always put the directory inside recordings/ folder
    if not output_dir.startswith("recordings/"):
        output_dir = f"recordings/{output_dir}"
    
    # Get recording duration
    while True:
        duration_input = input(f"\nRecording duration in seconds ({MIN_RECORD_TIME}-{MAX_RECORD_TIME}, press Enter for {DEFAULT_RECORD_TIME}): ").strip()
        if not duration_input:
            duration = DEFAULT_RECORD_TIME
            break
        try:
            duration = int(duration_input)
            if MIN_RECORD_TIME <= duration <= MAX_RECORD_TIME:
                break
            else:
                print(f"Duration must be between {MIN_RECORD_TIME} and {MAX_RECORD_TIME} seconds")
        except ValueError:
            print("Please enter a valid number")
    
    # Start recording
    print()
    if record_audio(port, output_dir, duration):
        # Ask if user wants to record again
        while True:
            print("\n" + "-"*50)
            response = input("Record another? (y/n): ").strip().lower()
            if response == 'y':
                # Ask for duration again for subsequent recordings
                while True:
                    duration_input = input(f"Recording duration in seconds ({MIN_RECORD_TIME}-{MAX_RECORD_TIME}, press Enter for {DEFAULT_RECORD_TIME}): ").strip()
                    if not duration_input:
                        duration = DEFAULT_RECORD_TIME
                        break
                    try:
                        duration = int(duration_input)
                        if MIN_RECORD_TIME <= duration <= MAX_RECORD_TIME:
                            break
                        else:
                            print(f"Duration must be between {MIN_RECORD_TIME} and {MAX_RECORD_TIME} seconds")
                    except ValueError:
                        print("Please enter a valid number")
                record_audio(port, output_dir, duration)
            else:
                break
    
    print("\nDone!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

