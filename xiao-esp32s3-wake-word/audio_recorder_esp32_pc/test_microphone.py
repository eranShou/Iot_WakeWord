"""
Microphone Test Tool
Records a sample and analyzes the audio data to diagnose issues
"""

import serial
import struct
import wave
import time
import os

def test_microphone(port, baud=921600):
    """Test microphone and show what's being recorded"""
    
    print("="*60)
    print("XIAO ESP32S3 Microphone Test")
    print("="*60)
    
    print(f"\nConnecting to {port}...")
    ser = serial.Serial(port, baud, timeout=2)
    time.sleep(3)
    
    # Clear buffer
    ser.reset_input_buffer()
    
    # Show ESP32 status
    print("\nESP32 Status:")
    start = time.time()
    while time.time() - start < 2:
        if ser.in_waiting:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"  {line}")
            except:
                pass
    
    # Send record command
    print("\n" + "-"*60)
    print("Starting test recording...")
    print("-"*60)
    ser.write(b"RECORD\n")
    time.sleep(0.5)
    
    # Skip text responses until we get to binary data
    config = {}
    while True:
        if ser.in_waiting:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"  {line}")
                    if ':' in line:
                        key, value = line.split(':', 1)
                        try:
                            config[key] = int(value)
                        except:
                            pass
                    if "DATA_SIZE" in line:
                        time.sleep(0.5)
                        break
            except:
                break
    
    # Read start marker
    marker_bytes = ser.read(4)
    if len(marker_bytes) == 4:
        marker = struct.unpack('<I', marker_bytes)[0]
        if marker == 0xAA55AA55:
            print("✓ Start marker OK")
        else:
            print(f"⚠ Wrong marker: 0x{marker:08X}")
    
    # Read audio data
    print("\nRecording 5 seconds...")
    data_size = config.get('DATA_SIZE', 160000)
    audio_data = ser.read(data_size)
    
    # Read end marker
    marker_bytes = ser.read(4)
    
    print(f"✓ Received {len(audio_data)} bytes")
    
    ser.close()
    
    # Analyze the data
    print("\n" + "="*60)
    print("AUDIO DATA ANALYSIS")
    print("="*60)
    
    if len(audio_data) == 0:
        print("✗ ERROR: No data received!")
        return False
    
    # Convert to 16-bit samples
    samples = []
    for i in range(0, len(audio_data)-1, 2):
        sample = struct.unpack('<h', audio_data[i:i+2])[0]
        samples.append(sample)
    
    # Calculate statistics
    min_val = min(samples)
    max_val = max(samples)
    avg_val = sum(samples) / len(samples)
    
    # Check for variations
    unique_values = len(set(samples[:1000]))  # Check first 1000 samples
    all_unique = len(set(samples))
    
    # Calculate standard deviation
    variance = sum((x - avg_val) ** 2 for x in samples[:1000]) / len(samples[:1000])
    std_dev = variance ** 0.5
    
    print(f"\nSample Statistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Min value: {min_val}")
    print(f"  Max value: {max_val}")
    print(f"  Average: {avg_val:.2f}")
    print(f"  Range: {max_val - min_val}")
    print(f"  Unique values (first 1000): {unique_values}")
    print(f"  Total unique values: {all_unique}")
    print(f"  Standard deviation: {std_dev:.2f}")
    
    # Show first 50 samples to see pattern
    print(f"\nFirst 50 samples:")
    print(samples[:50])
    
    print("\n" + "-"*60)
    print("DIAGNOSIS:")
    print("-"*60)
    
    # Diagnose issues
    if min_val == 0 and max_val == 0:
        print("✗ PROBLEM: All samples are ZERO")
        print("\n🔧 LIKELY CAUSES:")
        print("  1. Microphone not connected")
        print("  2. Wrong pins - check wiring")
        print("  3. Microphone not powered (VDD → 3V3)")
        print("  4. SD/DOUT pin loose or broken")
        return False
    
    elif all_unique < 5:
        print("✗ PROBLEM: Only a few unique values (stuck or oscillating)")
        print("\n🔧 LIKELY CAUSES:")
        print("  1. SD/DOUT pin not connected properly")
        print("  2. Wrong SD pin in code")
        print("  3. Clock issue - SCK/WS pins swapped?")
        return False
    
    elif std_dev < 10 and (max_val - min_val) < 100:
        print("⚠ WARNING: Very quiet signal")
        print("\n🔧 LIKELY CAUSES:")
        print("  1. L/R pin not connected (should be GND for left)")
        print("  2. Microphone too far away")
        print("  3. Low gain - try speaking louder")
        print("  4. SCK or WS pins swapped")
        print("\n💡 TRY:")
        print("  - Connect L/R pin to GND firmly")
        print("  - Tap microphone or clap near it")
        print("  - Check all connections are secure")
        return False
    
    else:
        # We have data with variation
        print("✓ MICROPHONE IS READING DATA!")
        print("\nℹ Signal Analysis:")
        
        if std_dev > 1000:
            print("  🔊 Strong signal - microphone working well!")
        elif std_dev > 100:
            print("  🔉 Moderate signal - acceptable for speech")
        elif std_dev > 10:
            print("  🔈 Weak signal - try speaking louder or closer")
        else:
            print("  ⚠ Very weak - may be noise floor only")
        
        print(f"\n  Range: {max_val - min_val}")
        print(f"  Std Dev: {std_dev:.2f}")
        print(f"  Unique values: {all_unique}")
        
        # Save test file
        filename = "test_recording.wav"
        print(f"\nSaving test file: {filename}")
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_data)
        
        print("✓ Test file saved!")
        print(f"\n💡 NEXT STEPS:")
        print(f"  1. Play test_recording.wav to hear if it captured sound")
        print(f"  2. If silent, try recording again with LOUDER noise")
        print(f"  3. If you hear audio, microphone is working!")
        return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("This tool will test your microphone and diagnose issues")
    print("="*60)
    print("\n💡 TIP: Make some noise during the recording!")
    print("   Try: clapping, speaking, tapping the microphone")
    
    port = input("\nEnter COM port (e.g., COM5): ").strip()
    if not port:
        port = "COM5"
    
    input("\nPress ENTER to start test...")
    
    try:
        test_microphone(port)
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    input("Press ENTER to exit...")

