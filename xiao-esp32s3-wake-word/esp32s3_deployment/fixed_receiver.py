#!/usr/bin/env python3
"""
Fixed WAV file receiver that properly handles mixed text and binary data
"""

import serial
import struct
import os
import time
import re

# Protocol constants
START_MARKER = 0xAA55AA55
END_MARKER = 0x55AA55AA

def fixed_receive_wav(port, baud_rate=921600, output_dir="received_wavs"):
    """Fixed WAV file receiver"""
    print(f"Fixed WAV Receiver on {port} at {baud_rate} baud")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        print("✓ Serial connection opened")
        
        file_count = 0
        buffer = b""
        
        while True:
            if ser.in_waiting > 0:
                # Read available data
                data = ser.read(ser.in_waiting)
                buffer += data
                
                # Look for "Preparing to send audio file..." in the buffer
                if b"Preparing to send audio file..." in buffer:
                    print("✓ Found 'Preparing to send audio file...'")
                    
                    # Look for "Sending: " pattern
                    sending_match = re.search(rb"Sending: ([^\r\n]+)", buffer)
                    if sending_match:
                        filename = sending_match.group(1).decode('utf-8')
                        print(f"✓ Found filename: {filename}")
                        
                        # Look for "FILENAME:" pattern
                        filename_match = re.search(rb"FILENAME:([^\r\n]+)", buffer)
                        if filename_match:
                            actual_filename = filename_match.group(1).decode('utf-8')
                            print(f"✓ Found FILENAME: {actual_filename}")
                            
                            # Look for "DATA_SIZE:" pattern
                            size_match = re.search(rb"DATA_SIZE:(\d+)", buffer)
                            if size_match:
                                data_size = int(size_match.group(1))
                                print(f"✓ Found DATA_SIZE: {data_size}")
                                
                                # Find the position after "DATA_SIZE:" in the buffer
                                size_pos = buffer.find(b"DATA_SIZE:") + len(b"DATA_SIZE:")
                                # Find the end of the line
                                line_end = buffer.find(b'\n', size_pos)
                                if line_end != -1:
                                    # Start reading WAV data from after the newline
                                    wav_start = line_end + 1
                                    print(f"✓ WAV data starts at position {wav_start}")
                                    
                                    # Try to receive the WAV data
                                    if receive_wav_data_fixed(ser, buffer[wav_start:], actual_filename, data_size, output_dir):
                                        file_count += 1
                                        print(f"✓ Successfully received file #{file_count}")
                                    
                                    # Clear buffer after successful reception
                                    buffer = b""
                                else:
                                    print("Waiting for complete DATA_SIZE line...")
                            else:
                                print("Waiting for DATA_SIZE...")
                        else:
                            print("Waiting for FILENAME...")
                    else:
                        print("Waiting for 'Sending:' pattern...")
                else:
                    # Show some of the buffer content
                    if len(buffer) > 500:
                        print(f"Buffer preview: {buffer[-100:]}")
                        # Keep only the last 2000 bytes to prevent memory issues
                        buffer = buffer[-2000:]
            else:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print(f"\nStopped. Total files received: {file_count}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'ser' in locals():
            ser.close()

def receive_wav_data_fixed(ser, initial_data, filename, data_size, output_dir):
    """Receive WAV data with proper handling"""
    try:
        print(f"Receiving WAV data: {filename} ({data_size} bytes)")
        
        # Start with any initial data we have
        wav_data = initial_data
        bytes_received = len(initial_data)
        
        # Read remaining data
        start_time = time.time()
        
        while bytes_received < data_size:
            if ser.in_waiting > 0:
                chunk_size = min(1024, data_size - bytes_received)
                chunk = ser.read(chunk_size)
                wav_data += chunk
                bytes_received += len(chunk)
                
                progress = (bytes_received / data_size) * 100
                if bytes_received % 8192 == 0:
                    print(f"Progress: {progress:.1f}% ({bytes_received}/{data_size})")
            else:
                time.sleep(0.001)
                
            # Timeout after 10 seconds
            if time.time() - start_time > 10:
                print("Timeout waiting for WAV data")
                return False
        
        # Look for END_MARKER
        if ser.in_waiting >= 4:
            end_data = ser.read(4)
            end_marker = struct.unpack('<I', end_data)[0]
            if end_marker == END_MARKER:
                print("✓ Found END_MARKER")
            else:
                print(f"WARNING: End marker mismatch. Got {hex(end_marker)}")
        
        # Save WAV file
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'wb') as f:
            f.write(wav_data)
        
        print(f"✓ Saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"ERROR receiving WAV data: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = "COM5"
    
    fixed_receive_wav(port)
