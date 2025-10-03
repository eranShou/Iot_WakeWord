#!/usr/bin/env python3
"""
ESP32 Model Converter for Hebrew Wake Word Detection

This script extracts model data from TensorFlow Lite C/C++ files and creates
the appropriate model_data.h file for ESP32 deployment.

Usage:
    python model_converter.py [path_to_cc_file] [output_path]

Arguments:
    path_to_cc_file: Path to the .cc file containing model data (optional)
    output_path: Path for the output model_data.h file (optional)

If no arguments are provided, it looks for hebrew_wake_word_model_cnn_int8.cc
in the training/models directory and outputs to ESP32/model_data.h
"""

import sys
import os
import re
from pathlib import Path

def extract_model_data(cc_file_path):
    """
    Extract model data array from a C/C++ file.

    Args:
        cc_file_path (str): Path to the .cc file

    Returns:
        tuple: (model_data_lines, model_length) where model_data_lines is a list
               of hex byte strings and model_length is the array size
    """
    print(f"Reading model data from: {cc_file_path}")

    with open(cc_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Find the model data array declaration
    # Look for: const unsigned char g_hebrew_wake_word_model_cnn_int8[] = {
    array_pattern = r'const\s+unsigned\s+char\s+g_hebrew_wake_word_model_cnn_int8\[\]\s*=\s*\{([^}]+)\};'
    match = re.search(array_pattern, content, re.DOTALL)

    if not match:
        raise ValueError("Could not find model data array in the file")

    array_content = match.group(1)

    # Extract hex bytes (format: 0xXX, )
    hex_pattern = r'0x[0-9a-fA-F]{2}'
    hex_bytes = re.findall(hex_pattern, array_content)

    print(f"Extracted {len(hex_bytes)} bytes of model data")

    # Group into lines of 12 bytes each for readability
    lines = []
    for i in range(0, len(hex_bytes), 12):
        line_bytes = hex_bytes[i:i+12]
        line_str = '  ' + ', '.join(line_bytes) + ','
        lines.append(line_str)

    return lines, len(hex_bytes)

def create_model_header(model_lines, model_length, output_path):
    """
    Create the model_data.h file for ESP32.

    Args:
        model_lines (list): Lines of hex bytes
        model_length (int): Total number of bytes
        output_path (str): Output file path
    """
    print(f"Creating model_data.h at: {output_path}")

    header_content = f"""unsigned char wake_word_model_tflite[] = {{
{model_lines[0]}
"""

    # Add all lines except the first and last
    for line in model_lines[1:-1]:
        header_content += line + '\n'

    # Add the last line without comma
    if model_lines:
        last_line = model_lines[-1]
        if last_line.endswith(','):
            last_line = last_line[:-1]
        header_content += last_line + '\n'

    header_content += f"""}};
unsigned int wake_word_model_tflite_len = {model_length};
"""

    with open(output_path, 'w') as f:
        f.write(header_content)

    print(f"Model data written to {output_path}")
    print(f"Model size: {model_length} bytes ({model_length/1024:.1f} KB)")

def main():
    # Default paths
    script_dir = Path(__file__).parent
    default_cc_file = script_dir.parent / "tensorflow_wake_word_detection" / "training" / "models" / "hebrew_wake_word_model_cnn_int8.cc"
    default_output = script_dir / "model_data.h"

    # Check command line arguments
    if len(sys.argv) >= 2:
        cc_file_path = Path(sys.argv[1])
    else:
        cc_file_path = default_cc_file

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = default_output

    # Validate input file
    if not cc_file_path.exists():
        print(f"Error: Model file not found at {cc_file_path}")
        print(f"Looking for: {default_cc_file}")
        sys.exit(1)

    try:
        # Extract model data
        model_lines, model_length = extract_model_data(cc_file_path)

        # Create header file
        create_model_header(model_lines, model_length, output_path)

        print("\n✅ Model conversion completed successfully!")
        print(f"📁 Model data saved to: {output_path}")
        print(f"📊 Model size: {model_length} bytes")
        print("\nNext steps:")
        print("1. Open your ESP32 project in Arduino IDE or PlatformIO")
        print("2. Replace the existing model_data.h with the new one")
        print("3. Upload the code to your ESP32")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
