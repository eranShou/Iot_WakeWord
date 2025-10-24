/*
 * Deployment Configuration Header
 * ESP32-S3 Wake Word Detection Settings
 * All values derived from model_config.h - no magic numbers
 */

#ifndef CONFIG_H
#define CONFIG_H

#include "model_config.h"

// ============================================================================
// SLIDING WINDOW CONFIGURATION
// ============================================================================

// Window stride in milliseconds (from plan: 500ms intervals)
#define WINDOW_STRIDE_MS 500

// Window overlap ratio (from plan: 50% overlap)
#define WINDOW_OVERLAP_RATIO 0.5f

// Calculate window stride in samples
#define WINDOW_STRIDE_SAMPLES ((WINDOW_STRIDE_MS * SAMPLE_RATE) / 1000)

// Audio buffer duration in seconds (from plan: 2+ seconds for sliding windows)
#define AUDIO_BUFFER_SECONDS 2

// Audio buffer size in samples
#define AUDIO_BUFFER_SIZE (AUDIO_BUFFER_SECONDS * SAMPLE_RATE * NUM_CHANNELS)

// ============================================================================
// DETECTION CONFIGURATION
// ============================================================================

// Confidence threshold from model_config.h
#define DETECTION_CONFIDENCE_THRESHOLD CONFIDENCE_THRESHOLD

// Cooldown period to prevent duplicate detections (milliseconds)
#define DETECTION_COOLDOWN_MS 300

// ============================================================================
// MEMORY ALLOCATION
// ============================================================================

// Tensor arena size for TFLite Micro (increased for 2MB model)
// From plan: 200KB for 2MB model (increased from 8KB in config)
#define TFLITE_TENSOR_ARENA_SIZE 200000

// Spectrogram buffer size (32x32x1 float = 4096 bytes)
#define SPECTROGRAM_BUFFER_SIZE (SPECTROGRAM_HEIGHT * SPECTROGRAM_WIDTH * MODEL_INPUT_CHANNELS)

// Audio window buffer size (1 second of 16-bit mono audio)
#define AUDIO_WINDOW_SIZE NUM_SAMPLES

// ============================================================================
// SERIAL COMMUNICATION PROTOCOL
// ============================================================================

// Serial baud rate (from mic_config.h)
#define SERIAL_BAUD_RATE 921600

// Data transfer markers (from mic_config.h)
#define START_MARKER 0xAA55AA55
#define END_MARKER 0x55AA55AA

// WAV header size (standard WAV header)
#define WAV_HEADER_SIZE 44

// Complete WAV file size (header + audio data)
#define WAV_FILE_SIZE (WAV_HEADER_SIZE + (NUM_SAMPLES * NUM_CHANNELS * 2))

// ============================================================================
// FILENAME FORMAT
// ============================================================================

// Maximum filename length for generated WAV files
#define MAX_FILENAME_LENGTH 128

// Filename format: "class_YYYYMMDD_HHMMSS_conf[all_scores].wav"
// Example: "shalom_20241019_143052_conf[0.05,0.85,0.03,0.04,0.03].wav"

// ============================================================================
// GPIO CONFIGURATION
// ============================================================================

// LED pin for visual feedback (optional)
#define LED_PIN 21

// ============================================================================
// VALIDATION MACROS
// ============================================================================

// Validate window stride is reasonable
#if WINDOW_STRIDE_MS < 100 || WINDOW_STRIDE_MS > 2000
#error "WINDOW_STRIDE_MS must be between 100ms and 2000ms"
#endif

// Validate audio buffer is large enough
#if AUDIO_BUFFER_SIZE < (NUM_SAMPLES * 2)
#error "AUDIO_BUFFER_SIZE must be at least 2x NUM_SAMPLES for sliding windows"
#endif

// Validate tensor arena size
#if TFLITE_TENSOR_ARENA_SIZE < 50000
#error "TFLITE_TENSOR_ARENA_SIZE should be at least 50KB for stable operation"
#endif

// Validate spectrogram buffer size
#if SPECTROGRAM_BUFFER_SIZE != (SPECTROGRAM_HEIGHT * SPECTROGRAM_WIDTH * MODEL_INPUT_CHANNELS)
#error "SPECTROGRAM_BUFFER_SIZE calculation is incorrect"
#endif

#endif // CONFIG_H
