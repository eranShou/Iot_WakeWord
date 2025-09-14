/*
 * Configuration Header for Hebrew Wake Word Detection
 *
 * This file contains all configuration constants and settings for the
 * ESP32-S3 Hebrew wake word detection system.
 */

#ifndef CONFIG_H
#define CONFIG_H

// ===== AUDIO CONFIGURATION =====
#define SAMPLE_RATE 16000
#define AUDIO_LENGTH_MS 1000
#define BUFFER_SIZE (SAMPLE_RATE * AUDIO_LENGTH_MS / 1000)

// ===== MFCC CONFIGURATION =====
#define WINDOW_SIZE_MS 30
#define WINDOW_STRIDE_MS 10
#define FEATURE_BINS 40
#define NUM_MFCC_COEFFS 10

// Derived audio constants
#define WINDOW_SIZE_SAMPLES (WINDOW_SIZE_MS * SAMPLE_RATE / 1000)
#define WINDOW_STRIDE_SAMPLES (WINDOW_STRIDE_MS * SAMPLE_RATE / 1000)
#define FFT_SIZE WINDOW_SIZE_SAMPLES

// ===== WAKE WORD DETECTION =====
#define DETECTION_THRESHOLD 0.8f
#define COOLDOWN_MS 2000
#define NUM_CLASSES 4

// Wake word labels
#define WAKE_WORD_SILENCE 0
#define WAKE_WORD_UNKNOWN 1
#define WAKE_WORD_SHALOM 2
#define WAKE_WORD_LEHITRAOT 3

const char* const WAKE_WORD_LABELS[] = {
    "silence",
    "unknown",
    "shalom",
    "lehitraot"
};

// ===== HARDWARE CONFIGURATION =====
// XIAO ESP32-S3 pins
#define LED_PIN 21
#define MIC_DATA_PIN 23
#define MIC_CLOCK_PIN 22

// ===== TENSORFLOW LITE MICRO =====
#define TENSOR_ARENA_SIZE (80 * 1024)  // 80KB for model
#define MAX_OP_RESOLVER_SIZE 10

// ===== POWER MANAGEMENT =====
#define CPU_FREQUENCY_MHZ 80
#define LIGHT_SLEEP_DELAY_US 100000  // 100ms

// ===== DEBUGGING =====
#define ENABLE_SERIAL_DEBUG true
#define ENABLE_PERFORMANCE_MONITORING false

// Performance monitoring intervals (in milliseconds)
#define PERFORMANCE_REPORT_INTERVAL 5000

// ===== CUSTOM WAKE WORD SUPPORT =====
#define MAX_CUSTOM_WORDS 5
#define CUSTOM_WORD_AUDIO_SAMPLES 100

// ===== ERROR HANDLING =====
#define MAX_RETRIES 3
#define ERROR_BACKOFF_MS 1000

// ===== MEMORY MANAGEMENT =====
#define AUDIO_BUFFER_POOL_SIZE 2
#define FEATURE_BUFFER_POOL_SIZE 2

#endif // CONFIG_H
