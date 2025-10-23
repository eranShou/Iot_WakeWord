#!/usr/bin/env python3
"""
ESP32-S3 Wake Word Detection - WAV File Receiver
Receives WAV files from ESP32-S3 via serial communication
Handles the protocol: START_MARKER -> FILENAME -> DATA_SIZE -> WAV_DATA -> END_MARKER
"""

import serial
import struct
import os
import time
from datetime import datetime
import argparse

# Protocol constants (matching ESP32 code)
START_MARKER = 0xAA55AA55
END_MARKER = 0x55AA55AA

class WAVReceiver:
    def __init__(self, port, baud_rate=921600, output_dir="received_wavs"):
        self.ser = serial.Serial(port, baud_rate, timeout=10)
        self.output_dir = output_dir
        self.file_count = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"WAV Receiver initialized")
        print(f"Port: {port}")
        print(f"Baud rate: {baud_rate}")
        print(f"Output directory: {output_dir}")
        print(f"Waiting for WAV files from ESP32-S3...")
        print("-" * 50)
    
    def wait_for_start_marker(self):
        """Wait for START_MARKER from ESP32"""
        print("DEBUG: Waiting for START_MARKER...")
        attempts = 0
        while True:
            if self.ser.in_waiting >= 4:
                data = self.ser.read(4)
                marker = struct.unpack('<I', data)[0]  # Little-endian 32-bit unsigned int
                if marker == START_MARKER:
                    print("DEBUG: Found START_MARKER!")
                    return True
                else:
                    # If not the start marker, look for it in the data stream
                    # The ESP32 might be sending text data first
                    if attempts % 100 == 0:  # Print every 100 attempts
                        print(f"DEBUG: Read marker: {hex(marker)} (expected: {hex(START_MARKER)})")
            attempts += 1
            if attempts % 1000 == 0:  # Print every 1000 attempts
                print(f"DEBUG: Still waiting for START_MARKER... (attempts: {attempts})")
            time.sleep(0.001)  # Small delay to prevent excessive CPU usage
    
    def read_line(self):
        """Read a line from serial (until newline)"""
        line = b""
        while True:
            if self.ser.in_waiting > 0:
                char = self.ser.read(1)
                if char == b'\n':
                    return line.decode('utf-8').strip()
                line += char
            time.sleep(0.001)
    
    def receive_wav_file(self):
        """Receive a single WAV file from ESP32"""
        try:
            # Wait for start marker
            if not self.wait_for_start_marker():
                return False
            
            # Read filename
            filename_line = self.read_line()
            if not filename_line.startswith("FILENAME:"):
                print(f"ERROR: Expected FILENAME, got: {filename_line}")
                return False
            
            filename = filename_line[9:]  # Remove "FILENAME:" prefix
            print(f"Receiving: {filename}")
            
            # Read data size
            size_line = self.read_line()
            if not size_line.startswith("DATA_SIZE:"):
                print(f"ERROR: Expected DATA_SIZE, got: {size_line}")
                return False
            
            data_size = int(size_line[10:])  # Remove "DATA_SIZE:" prefix
            print(f"Data size: {data_size} bytes")
            
            # Wait a moment for ESP32 to send the data
            time.sleep(0.1)
            
            # Read WAV data
            wav_data = b""
            bytes_received = 0
            
            while bytes_received < data_size:
                if self.ser.in_waiting > 0:
                    chunk_size = min(1024, data_size - bytes_received)
                    chunk = self.ser.read(chunk_size)
                    wav_data += chunk
                    bytes_received += len(chunk)
                    
                    # Progress indicator
                    if bytes_received % 8192 == 0:  # Every 8KB
                        progress = (bytes_received / data_size) * 100
                        print(f"Progress: {progress:.1f}% ({bytes_received}/{data_size} bytes)")
                else:
                    time.sleep(0.001)
            
            # Wait for end marker
            if self.ser.in_waiting >= 4:
                end_data = self.ser.read(4)
                end_marker = struct.unpack('<I', end_data)[0]
                if end_marker != END_MARKER:
                    print(f"WARNING: End marker mismatch. Expected {hex(END_MARKER)}, got {hex(end_marker)}")
            
            # Save WAV file
            output_path = os.path.join(self.output_dir, filename)
            with open(output_path, 'wb') as f:
                f.write(wav_data)
            
            self.file_count += 1
            print(f"✓ Saved: {output_path}")
            print(f"Total files received: {self.file_count}")
            print("-" * 50)
            
            return True
            
        except Exception as e:
            print(f"ERROR receiving WAV file: {e}")
            return False
    
    def run_continuous(self):
        """Continuously receive WAV files"""
        print("Starting continuous WAV file reception...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                if self.ser.in_waiting > 0:
                    # Debug: Show what data is available without consuming it
                    available = self.ser.in_waiting
                    print(f"DEBUG: {available} bytes available in serial buffer")
                    
                    # Try to receive WAV file
                    self.receive_wav_file()
                else:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nStopping WAV receiver...")
            print(f"Total files received: {self.file_count}")
    
    def close(self):
        """Close serial connection"""
        if self.ser.is_open:
            self.ser.close()
            print("Serial connection closed")

def list_serial_ports():
    """List available serial ports"""
    import serial.tools.list_ports
    
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found")
        return []
    
    print("Available serial ports:")
    for i, port in enumerate(ports):
        print(f"  {i}: {port.device} - {port.description}")
    
    return [port.device for port in ports]

def interactive_port_selection():
    """Interactive port selection menu"""
    ports = list_serial_ports()
    if not ports:
        return None
    
    print("\nSelect a serial port:")
    while True:
        try:
            choice = input(f"Enter port number (0-{len(ports)-1}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            port_index = int(choice)
            if 0 <= port_index < len(ports):
                return ports[port_index]
            else:
                print(f"Invalid choice. Please enter 0-{len(ports)-1}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")
        except KeyboardInterrupt:
            print("\nCancelled by user")
            return None

def show_welcome_menu():
    """Display welcome menu and instructions"""
    print("=" * 60)
    print("🎤 ESP32-S3 Wake Word Detection - WAV File Receiver")
    print("=" * 60)
    print()
    print("This script receives WAV files from your ESP32-S3 wake word")
    print("detection system via serial communication.")
    print()
    print("Features:")
    print("• Automatic WAV file reception and saving")
    print("• Detailed filenames with confidence scores")
    print("• Real-time progress monitoring")
    print("• Organized file storage")
    print()
    print("Requirements:")
    print("• ESP32-S3 connected via USB")
    print("• Wake word detection system running")
    print("• Serial communication at 921600 baud")
    print()

def show_configuration_menu():
    """Interactive configuration menu"""
    print("Configuration Options:")
    print("-" * 30)
    
    # Baud rate selection
    print("1. Baud Rate:")
    baud_options = [115200, 230400, 460800, 921600, 1843200]
    print("   Available options:")
    for i, baud in enumerate(baud_options):
        print(f"     {i}: {baud} (recommended: {921600 if baud == 921600 else ''})")
    
    while True:
        try:
            baud_choice = input(f"Select baud rate (0-{len(baud_options)-1}) [3 for 921600]: ").strip()
            if baud_choice == "":
                baud_rate = 921600  # Default
                break
            baud_index = int(baud_choice)
            if 0 <= baud_index < len(baud_options):
                baud_rate = baud_options[baud_index]
                break
            else:
                print(f"Invalid choice. Please enter 0-{len(baud_options)-1}")
        except ValueError:
            print("Invalid input. Please enter a number")
    
    # Output directory
    print(f"\n2. Output Directory:")
    default_output = "received_wavs"
    output_dir = input(f"Enter output directory [{default_output}]: ").strip()
    if not output_dir:
        output_dir = default_output
    
    return baud_rate, output_dir

def show_help_menu():
    """Display help information"""
    print("\n" + "=" * 60)
    print("📖 HELP - ESP32-S3 WAV Receiver")
    print("=" * 60)
    print()
    print("Troubleshooting:")
    print()
    print("1. No ports found:")
    print("   • Ensure ESP32-S3 is connected via USB")
    print("   • Install USB drivers if needed")
    print("   • Try different USB cable/port")
    print()
    print("2. Connection failed:")
    print("   • Check ESP32-S3 is powered and running")
    print("   • Verify correct COM port selection")
    print("   • Ensure no other program is using the port")
    print()
    print("3. No data received:")
    print("   • Confirm wake word detection system is running")
    print("   • Check baud rate matches ESP32-S3 (921600)")
    print("   • Verify ESP32-S3 serial output shows detections")
    print()
    print("4. File format issues:")
    print("   • WAV files are standard format, playable in any audio player")
    print("   • Filenames contain confidence scores for all 5 classes")
    print("   • Files are saved with complete WAV headers")
    print()
    print("5. Performance tips:")
    print("   • Close other serial terminal programs")
    print("   • Use USB 3.0 ports for better speed")
    print("   • Ensure stable USB connection")
    print()
    print("Command line usage:")
    print("  python receive_wav_files.py --port COM3 --baud 921600")
    print("  python receive_wav_files.py --list-ports")
    print("  python receive_wav_files.py --help")
    print()

def interactive_menu():
    """Main interactive menu"""
    while True:
        print("\n" + "=" * 40)
        print("🚀 ESP32-S3 WAV Receiver - Main Menu")
        print("=" * 40)
        print("1. Start receiving WAV files")
        print("2. List available serial ports")
        print("3. Configuration options")
        print("4. Help & Troubleshooting")
        print("5. Exit")
        print()
        
        choice = input("Select an option (1-5): ").strip()
        
        if choice == "1":
            # Start receiving
            port = interactive_port_selection()
            if port:
                baud_rate, output_dir = show_configuration_menu()
                
                print(f"\n🎯 Starting WAV receiver...")
                print(f"   Port: {port}")
                print(f"   Baud rate: {baud_rate}")
                print(f"   Output: {output_dir}")
                print()
                
                try:
                    receiver = WAVReceiver(port, baud_rate, output_dir)
                    receiver.run_continuous()
                except serial.SerialException as e:
                    print(f"❌ Serial error: {e}")
                    print("Make sure the ESP32-S3 is connected and the port is correct")
                except Exception as e:
                    print(f"❌ Error: {e}")
                finally:
                    if 'receiver' in locals():
                        receiver.close()
            
        elif choice == "2":
            # List ports
            list_serial_ports()
            
        elif choice == "3":
            # Configuration
            baud_rate, output_dir = show_configuration_menu()
            print(f"\n✅ Configuration saved:")
            print(f"   Baud rate: {baud_rate}")
            print(f"   Output directory: {output_dir}")
            
        elif choice == "4":
            # Help
            show_help_menu()
            
        elif choice == "5":
            # Exit
            print("\n👋 Goodbye! Happy wake word detecting!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-5.")

def main():
    parser = argparse.ArgumentParser(description="Receive WAV files from ESP32-S3 Wake Word Detection")
    parser.add_argument("--port", "-p", help="Serial port (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)")
    parser.add_argument("--baud", "-b", type=int, default=921600, help="Baud rate (default: 921600)")
    parser.add_argument("--output", "-o", default="received_wavs", help="Output directory (default: received_wavs)")
    parser.add_argument("--list-ports", "-l", action="store_true", help="List available serial ports")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run interactive menu mode")
    
    args = parser.parse_args()
    
    # Show welcome menu for interactive mode or when no arguments provided
    if args.interactive or (not args.port and not args.list_ports):
        show_welcome_menu()
        interactive_menu()
        return
    
    if args.list_ports:
        list_serial_ports()
        return
    
    if not args.port:
        print("ERROR: Serial port required")
        print("Use --list-ports to see available ports")
        print("Use --interactive for guided setup")
        print("Example: python receive_wav_files.py --port COM3")
        return
    
    try:
        receiver = WAVReceiver(args.port, args.baud, args.output)
        receiver.run_continuous()
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        print("Make sure the ESP32-S3 is connected and the port is correct")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'receiver' in locals():
            receiver.close()

if __name__ == "__main__":
    main()
