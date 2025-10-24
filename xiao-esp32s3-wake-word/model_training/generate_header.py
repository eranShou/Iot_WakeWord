"""
Generate C header files for ESP32-S3 deployment
Creates wake_word_model.h (model data) and model_config.h (configuration)
All parameters from config.json - ensures training/inference alignment
"""

import json
import os
from datetime import datetime

def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r') as f:
        return json.load(f)

def generate_model_header(config):
    """
    Generate wake_word_model.h with TFLite model as C array
    """
    tflite_path = config['output']['tflite_model']
    header_path = config['output']['model_header']
    
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")
    
    print(f"Generating model header: {header_path}")
    
    # Read TFLite model as binary
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    model_size = len(model_data)
    print(f"Model size: {model_size} bytes ({model_size/1024:.2f} KB)")
    
    # Generate header content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    header_content = f"""/*
 * Wake Word Model Header
 * Auto-generated from wake_word_model.tflite
 * Generated: {timestamp}
 * Model size: {model_size} bytes ({model_size/1024:.2f} KB)
 * 
 * This file contains the TFLite model as a C array for ESP32-S3 deployment.
 * Include this file in your Arduino sketch along with model_config.h
 */

#ifndef WAKE_WORD_MODEL_H
#define WAKE_WORD_MODEL_H

// Model data as unsigned char array
const unsigned char wake_word_model[] = {{
"""
    
    # Convert binary data to hex array
    hex_lines = []
    for i in range(0, model_size, 16):  # 16 bytes per line
        chunk = model_data[i:i+16]
        hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
        hex_lines.append(f"  {hex_values}")
    
    header_content += ',\n'.join(hex_lines)
    header_content += f"""
}};

// Model size in bytes
const unsigned int wake_word_model_len = {model_size};

#endif // WAKE_WORD_MODEL_H
"""
    
    # Write header file
    with open(header_path, 'w') as f:
        f.write(header_content)
    
    print(f"Model header saved to: {header_path}")
    return header_path

def generate_config_header(config):
    """
    Generate model_config.h with all configuration parameters
    """
    header_path = config['output']['config_header']
    
    print(f"Generating config header: {header_path}")
    
    # Extract configuration
    audio_config = config['audio']
    spectrogram_config = config['spectrogram']
    model_config = config['model']
    classes = config['classes']
    esp32_config = config['esp32']
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    header_content = f"""/*
 * Model Configuration Header
 * Auto-generated from config.json
 * Generated: {timestamp}
 * 
 * This file contains all configuration parameters for ESP32-S3 deployment.
 * These values are guaranteed to match the training configuration.
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

// ============================================================================
// AUDIO CONFIGURATION
// ============================================================================

#define SAMPLE_RATE {audio_config['sample_rate']}
#define DURATION_SECONDS {int(audio_config['duration_seconds'])}
#define NUM_SAMPLES {audio_config['num_samples']}
#define NUM_CHANNELS {audio_config['channels']}

// ============================================================================
// SPECTROGRAM CONFIGURATION
// ============================================================================

#define FRAME_LENGTH {spectrogram_config['frame_length']}
#define FRAME_STEP {spectrogram_config['frame_step']}
#define FFT_LENGTH {spectrogram_config['fft_length']}
#define SPECTROGRAM_HEIGHT {spectrogram_config['target_height']}
#define SPECTROGRAM_WIDTH {spectrogram_config['target_width']}

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

#define NUM_CLASSES {model_config['num_classes']}
#define CONFIDENCE_THRESHOLD {esp32_config['confidence_threshold']}
#define TENSOR_ARENA_SIZE {esp32_config['tensor_arena_size']}

// Model input shape
#define MODEL_INPUT_HEIGHT {model_config['input_shape'][0]}
#define MODEL_INPUT_WIDTH {model_config['input_shape'][1]}
#define MODEL_INPUT_CHANNELS {model_config['input_shape'][2]}

// ============================================================================
// CLASS LABELS
// ============================================================================

// Class labels in training order (matching label_map from config.json)
const char* CLASS_LABELS[NUM_CLASSES] = {{
"""
    
    # Add class labels
    for i, label in enumerate(classes['labels']):
        header_content += f'  "{label}"'
        if i < len(classes['labels']) - 1:
            header_content += ","
        header_content += f"  // index {i}\n"
    
    header_content += """};

// ============================================================================
// CLASS INDEX MAPPING
// ============================================================================

// Class indices (matching config.json label_map)
#define CLASS_LEHITRAOOT 0
#define CLASS_SHALOOM 1
#define CLASS_NOISE 2
#define CLASS_UNKNOWN 3

// ============================================================================
// VALIDATION MACROS
// ============================================================================

// Validate configuration consistency
#if NUM_SAMPLES != (SAMPLE_RATE * DURATION_SECONDS)
#error "NUM_SAMPLES must equal SAMPLE_RATE * DURATION_SECONDS"
#endif

#if NUM_CLASSES != 4
#error "NUM_CLASSES must be 4 for this model"
#endif

#if SPECTROGRAM_HEIGHT != 32 || SPECTROGRAM_WIDTH != 32
#error "Spectrogram dimensions must be 32x32 for this model"
#endif

#endif // MODEL_CONFIG_H
"""
    
    # Write header file
    with open(header_path, 'w') as f:
        f.write(header_content)
    
    print(f"Config header saved to: {header_path}")
    return header_path

def verify_headers(config):
    """
    Verify generated headers are valid
    """
    print("\nVerifying generated headers...")
    
    # Check model header
    model_header = config['output']['model_header']
    if os.path.exists(model_header):
        print(f"✓ Model header generated: {model_header}")
    else:
        print(f"✗ Model header not found: {model_header}")
    
    # Check config header
    config_header = config['output']['config_header']
    if os.path.exists(config_header):
        print(f"✓ Config header generated: {config_header}")
    else:
        print(f"✗ Config header not found: {config_header}")
    
    # Check TFLite model exists
    tflite_model = config['output']['tflite_model']
    if os.path.exists(tflite_model):
        print(f"✓ TFLite model exists: {tflite_model}")
    else:
        print(f"✗ TFLite model not found: {tflite_model}")

def generate_headers():
    """
    Main function to generate both header files
    """
    config = load_config()
    
    print("Generating C header files for ESP32-S3...")
    print("=" * 50)
    
    try:
        # Generate model header
        model_header = generate_model_header(config)
        
        # Generate config header
        config_header = generate_config_header(config)
        
        # Verify headers
        verify_headers(config)
        
        print(f"\nHeader generation completed successfully!")
        print(f"Model header: {model_header}")
        print(f"Config header: {config_header}")
        print(f"\nNext steps:")
        print(f"1. Run deploy_esp32.py to copy headers to ESP32 deployment folder")
        print(f"2. Include both headers in your Arduino sketch")
        print(f"3. Use CLASS_LABELS array for output formatting")
        
        return model_header, config_header
        
    except Exception as e:
        print(f"Error generating headers: {e}")
        raise

if __name__ == "__main__":
    generate_headers()
    print("Header generation completed successfully!")
