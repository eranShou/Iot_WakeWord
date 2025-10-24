/*
 * Model Configuration Header
 * Auto-generated from config.json
 * Generated: 2025-10-24 21:37:01
 * 
 * This file contains all configuration parameters for ESP32-S3 deployment.
 * These values are guaranteed to match the training configuration.
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

// ============================================================================
// AUDIO CONFIGURATION
// ============================================================================

#define SAMPLE_RATE 16000
#define DURATION_SECONDS 1
#define NUM_SAMPLES 16000
#define NUM_CHANNELS 1

// ============================================================================
// SPECTROGRAM CONFIGURATION
// ============================================================================

#define FRAME_LENGTH 255
#define FRAME_STEP 128
#define FFT_LENGTH 256
#define SPECTROGRAM_HEIGHT 32
#define SPECTROGRAM_WIDTH 32

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

#define NUM_CLASSES 4
#define CONFIDENCE_THRESHOLD 0.5
#define TENSOR_ARENA_SIZE 8000

// Model input shape
#define MODEL_INPUT_HEIGHT 32
#define MODEL_INPUT_WIDTH 32
#define MODEL_INPUT_CHANNELS 1

// ============================================================================
// CLASS LABELS
// ============================================================================

// Class labels in training order (matching label_map from config.json)
const char* CLASS_LABELS[NUM_CLASSES] = {
  "lehitraoot",  // index 0
  "shalom",  // index 1
  "background",  // index 2
  "unknown"  // index 3
};

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
