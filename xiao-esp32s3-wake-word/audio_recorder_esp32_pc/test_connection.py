"""
ESP32 Connection Test - Diagnostic Tool
Tests connection and shows what the ESP32 is sending
"""

import serial
import serial.tools.list_ports
import time
import sys

def find_ports():
    """List all available ports"""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found!")
        return None
    
    print("Available ports:")
    for i, port in enumerate(ports):
        print(f"  [{i}] {port.device} - {port.description}")
    
    choice = input("\nSelect port number: ").strip()
    if choice.isdigit() and int(choice) < len(ports):
        return ports[int(choice)].device
    return choice

def test_baud_rate(port, baud):
    """Test a specific baud rate"""
    print(f"\nTesting {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)  # Wait for ESP32 to reset
        
        # Clear buffer
        ser.reset_input_buffer()
        
        # Send status command
        ser.write(b"STATUS\n")
        time.sleep(0.5)
        
        # Read any responses
        responses = []
        start_time = time.time()
        while time.time() - start_time < 3:
            if ser.in_waiting:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        responses.append(line)
                        print(f"  Received: {line}")
                except:
                    pass
        
        ser.close()
        
        if responses:
            return True, responses
        else:
            print("  No response")
            return False, []
    except Exception as e:
        print(f"  Error: {e}")
        return False, []

def main():
    print("="*60)
    print("ESP32 Connection Diagnostic Tool")
    print("="*60)
    print("\nThis tool will help diagnose connection issues.")
    print("Make sure the Arduino sketch is uploaded to your ESP32!\n")
    
    # Find port
    port = find_ports()
    if not port:
        return
    
    print(f"\nTesting connection to {port}...")
    print("-"*60)
    
    # Test common baud rates
    baud_rates = [921600, 115200, 460800, 230400, 9600]
    
    working_baud = None
    for baud in baud_rates:
        success, responses = test_baud_rate(port, baud)
        if success:
            print(f"  ✓ Working at {baud} baud!")
            working_baud = baud
            
            # Check for expected responses
            expected = ["READY", "I2S_OK", "WAITING"]
            found = sum(1 for exp in expected if any(exp in r for r in responses))
            
            if found >= 2:
                print(f"  ✓ ESP32 responding correctly!")
                break
            else:
                print(f"  ⚠ ESP32 responding but may not have correct sketch")
        time.sleep(0.5)
    
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)
    
    if working_baud:
        print(f"✓ ESP32 is connected and responding at {working_baud} baud")
        print(f"\nUpdate record_audio.py with:")
        print(f"  BAUD_RATE = {working_baud}")
        print(f"\nAnd audio_recorder_esp32_pc.ino with:")
        print(f"  Serial.begin({working_baud});")
    else:
        print("✗ ESP32 is not responding")
        print("\nTROUBLESHOOTING STEPS:")
        print("  1. Make sure the Arduino sketch is uploaded")
        print("  2. Check in Arduino IDE:")
        print("     - Tools → Board → Select your ESP32 board")
        print("     - Tools → Port → Select the correct port")
        print("     - Verify → Sketch → Upload")
        print("  3. Press the RESET button on ESP32")
        print("  4. Try unplugging and replugging USB")
        print("  5. Open Serial Monitor (115200) to see if anything appears")
    
    print("\n" + "="*60)
    
    # Try reading raw data
    print("\nReading raw data for 5 seconds...")
    print("(Press Ctrl+C to stop early)")
    print("-"*60)
    
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)
        start_time = time.time()
        
        while time.time() - start_time < 5:
            if ser.in_waiting:
                data = ser.read(ser.in_waiting)
                print(f"Raw: {data}")
                # Try to decode
                try:
                    text = data.decode('utf-8', errors='ignore')
                    if text.strip():
                        print(f"Text: {text.strip()}")
                except:
                    pass
        
        ser.close()
        print("-"*60)
        print("If you see 'READY', 'I2S_OK', or 'WAITING' above,")
        print("the sketch is working!")
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user")



