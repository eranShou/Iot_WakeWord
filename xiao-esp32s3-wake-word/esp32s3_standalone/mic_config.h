/*
 * Microphone Configuration Header
 * Centralized configuration for ESP32 audio recording
 * 
 * Hardware: Seeed Studio XIAO ESP32S3
 * Microphone: Built-in PDM microphone (MSM261D3526H1CPM)
 */

#ifndef MIC_CONFIG_H
#define MIC_CONFIG_H

// ============================================================================
// AUDIO RECORDING CONFIGURATION
// ============================================================================

// Audio Format Settings
#define SAMPLE_RATE 16000           // Sample rate in Hz (16kHz for speech)
#define BITS_PER_SAMPLE 16          // Bit depth (16-bit audio)
#define NUM_CHANNELS 1              // Mono recording
#define BYTES_PER_SAMPLE 2          // 16-bit = 2 bytes per sample

// Recording Duration Settings
#define DEFAULT_RECORD_TIME 15       // Default recording duration (seconds)
#define MIN_RECORD_TIME 1           // Minimum recording duration (seconds)
#define MAX_RECORD_TIME 60          // Maximum recording duration (seconds)

// ============================================================================
// AUDIO GAIN CONFIGURATION
// ============================================================================

// Gain Settings
#define DEFAULT_GAIN 8.0f           // Default gain multiplier (8x = +18dB boost for wake word training)
#define MIN_GAIN 0.1f               // Minimum gain (0.1x)
#define MAX_GAIN 20.0f              // Maximum gain (20x for high-gain scenarios)
#define GAIN_ENABLED_BY_DEFAULT true // Enable gain by default

// ============================================================================
// AUDIO COMPRESSION CONFIGURATION
// ============================================================================

// Compression Settings for Wake Word Training
#define COMPRESSION_ENABLED_BY_DEFAULT true  // Enable compression by default
#define COMPRESSION_THRESHOLD 0.5f           // Threshold for compression (0.0-1.0) - Lower for more headroom
#define COMPRESSION_RATIO 3.0f               // Compression ratio (3:1) - Less aggressive
#define COMPRESSION_ATTACK_MS 3.0f           // Attack time in milliseconds - Faster attack
#define COMPRESSION_RELEASE_MS 30.0f         // Release time in milliseconds - Faster release
#define COMPRESSION_MAKEUP_GAIN 1.5f         // Makeup gain after compression (1.5x = +3.5dB) - Reduced

// ============================================================================
// HARDWARE CONFIGURATION
// ============================================================================

// PDM Microphone Pin Configuration (XIAO ESP32S3 built-in)
#define PDM_CLK_PIN 42              // PDM Clock pin
#define PDM_DATA_PIN 41             // PDM Data pin

// Serial Communication
#define SERIAL_BAUD_RATE 921600     // High-speed serial for audio data
#define SERIAL_TIMEOUT_MS 2000      // Serial timeout for commands

// ============================================================================
// PROTOCOL CONFIGURATION
// ============================================================================

// Data Transfer Markers
#define START_MARKER 0xAA55AA55     // Start of audio data marker
#define END_MARKER 0x55AA55AA       // End of audio data marker

// Data Transfer Settings
#define CHUNK_SIZE 1024             // Size of data chunks for transfer
#define SETUP_DELAY_MS 200          // Delay before sending audio data

// ============================================================================
// CALCULATION MACROS
// ============================================================================

// Calculate audio data size for given duration
#define CALCULATE_AUDIO_SIZE(duration) \
    ((SAMPLE_RATE) * (duration) * (NUM_CHANNELS) * (BYTES_PER_SAMPLE))

// Calculate sample count for given duration
#define CALCULATE_SAMPLE_COUNT(duration) \
    ((SAMPLE_RATE) * (duration) * (NUM_CHANNELS))

// ============================================================================
// AUDIO PROCESSING LIMITS
// ============================================================================

// 16-bit audio limits
#define AUDIO_MAX_POSITIVE 32767    // Maximum positive 16-bit value
#define AUDIO_MAX_NEGATIVE -32768   // Maximum negative 16-bit value

// ============================================================================
// COMMAND STRINGS
// ============================================================================

// Recording Commands
#define CMD_RECORD "RECORD"
#define CMD_RECORD_PREFIX "RECORD:"
#define CMD_STATUS "STATUS"
#define CMD_SET_DURATION "SET_DURATION"

// Gain Control Commands
#define CMD_GAIN_PREFIX "GAIN:"
#define CMD_GAIN_ON "GAIN_ON"
#define CMD_GAIN_OFF "GAIN_OFF"
#define CMD_GAIN_STATUS "GAIN_STATUS"

// Compression Control Commands
#define CMD_COMPRESSION_PREFIX "COMPRESSION:"
#define CMD_COMPRESSION_ON "COMPRESSION_ON"
#define CMD_COMPRESSION_OFF "COMPRESSION_OFF"
#define CMD_COMPRESSION_STATUS "COMPRESSION_STATUS"

// ============================================================================
// STATUS MESSAGES
// ============================================================================

#define MSG_READY "READY"
#define MSG_WAITING "WAITING"
#define MSG_I2S_OK "I2S_OK"
#define MSG_RECORDING_START "RECORDING_START"
#define MSG_RECORDING_COMPLETE "RECORDING_COMPLETE"
#define MSG_ERROR_MICROPHONE "ERROR: Can't find microphone!"
#define MSG_ERROR_BOARD "Make sure you selected XIAO_ESP32S3 board."
#define MSG_ERROR_RECORD "ERROR: Failed to record audio!"

// Configuration Messages
#define MSG_SAMPLE_RATE "SAMPLE_RATE:"
#define MSG_BITS "BITS:"
#define MSG_CHANNELS "CHANNELS:"
#define MSG_DATA_SIZE "DATA_SIZE:"

// ============================================================================
// VALIDATION FUNCTIONS
// ============================================================================

// Validate recording duration
#define IS_VALID_DURATION(duration) \
    ((duration) >= (MIN_RECORD_TIME) && (duration) <= (MAX_RECORD_TIME))

// Validate gain value
#define IS_VALID_GAIN(gain) \
    ((gain) >= (MIN_GAIN) && (gain) <= (MAX_GAIN))

// ============================================================================
// DEBUG CONFIGURATION
// ============================================================================

#define DEBUG_ENABLED true         // Enable debug output
#define DEBUG_DELAY_MS 1000         // Debug delay for error loops

#endif // MIC_CONFIG_H
