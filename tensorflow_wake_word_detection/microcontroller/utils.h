/*
 * Utility Functions for Hebrew Wake Word Detection
 *
 * This file contains utility functions for debugging, performance monitoring,
 * and common operations used in the wake word detection system.
 */

#ifndef UTILS_H
#define UTILS_H

#include <Arduino.h>
#include "config.h"

// ===== DEBUGGING UTILITIES =====

#ifdef ENABLE_SERIAL_DEBUG
    #define DEBUG_PRINT(x) Serial.print(x)
    #define DEBUG_PRINTLN(x) Serial.println(x)
    #define DEBUG_PRINTF(...) Serial.printf(__VA_ARGS__)
#else
    #define DEBUG_PRINT(x)
    #define DEBUG_PRINTLN(x)
    #define DEBUG_PRINTF(...)
#endif

// ===== PERFORMANCE MONITORING =====

class PerformanceMonitor {
private:
    unsigned long start_time;
    unsigned long total_time;
    unsigned int sample_count;
    const char* operation_name;

public:
    PerformanceMonitor(const char* name) : operation_name(name), total_time(0), sample_count(0) {}

    void start() {
        start_time = micros();
    }

    void stop() {
        unsigned long duration = micros() - start_time;
        total_time += duration;
        sample_count++;

        #if ENABLE_PERFORMANCE_MONITORING
        DEBUG_PRINTF("%s took %lu us\n", operation_name, duration);
        #endif
    }

    void report() {
        if (sample_count > 0) {
            unsigned long avg_time = total_time / sample_count;
            DEBUG_PRINTF("%s - Average: %lu us, Samples: %u\n",
                        operation_name, avg_time, sample_count);
        }
    }

    void reset() {
        total_time = 0;
        sample_count = 0;
    }
};

// ===== MEMORY UTILITIES =====

class MemoryMonitor {
public:
    static void printMemoryInfo() {
        DEBUG_PRINTLN("\n=== Memory Information ===");
        DEBUG_PRINTF("Total heap: %d bytes\n", ESP.getHeapSize());
        DEBUG_PRINTF("Free heap: %d bytes\n", ESP.getFreeHeap());
        DEBUG_PRINTF("Min free heap: %d bytes\n", ESP.getMinFreeHeap());
        DEBUG_PRINTF("Max alloc heap: %d bytes\n", ESP.getMaxAllocHeap());
    }

    static bool checkMemoryThreshold(size_t threshold = 10000) {
        size_t free_heap = ESP.getFreeHeap();
        if (free_heap < threshold) {
            DEBUG_PRINTF("WARNING: Low memory! Free heap: %d bytes\n", free_heap);
            return false;
        }
        return true;
    }
};

// ===== AUDIO UTILITIES =====

class AudioUtils {
public:
    static void normalizeAudio(int16_t* buffer, size_t size, float* normalized) {
        const float scale = 1.0f / 32768.0f;
        for (size_t i = 0; i < size; i++) {
            normalized[i] = buffer[i] * scale;
        }
    }

    static float calculateRMS(float* buffer, size_t size) {
        float sum = 0.0f;
        for (size_t i = 0; i < size; i++) {
            sum += buffer[i] * buffer[i];
        }
        return sqrtf(sum / size);
    }

    static float calculateSNR(float* buffer, size_t size) {
        // Simple SNR calculation
        float signal_power = 0.0f;
        float noise_power = 0.0f;

        // Assume first half is signal, second half is noise (simplified)
        size_t half_size = size / 2;
        float signal_rms = calculateRMS(buffer, half_size);
        float noise_rms = calculateRMS(&buffer[half_size], half_size);

        signal_power = signal_rms * signal_rms;
        noise_power = noise_rms * noise_rms;

        if (noise_power > 0) {
            return 10.0f * log10f(signal_power / noise_power);
        }
        return 0.0f;
    }
};

// ===== LED UTILITIES =====

class LEDController {
private:
    uint8_t pin;
    bool state;

public:
    LEDController(uint8_t led_pin) : pin(led_pin), state(false) {
        pinMode(pin, OUTPUT);
        off();
    }

    void on() {
        digitalWrite(pin, HIGH);
        state = true;
    }

    void off() {
        digitalWrite(pin, LOW);
        state = false;
    }

    void toggle() {
        state ? off() : on();
    }

    void blink(unsigned long duration_ms, int times = 1) {
        for (int i = 0; i < times; i++) {
            on();
            delay(duration_ms);
            off();
            if (i < times - 1) delay(duration_ms);
        }
    }

    void blinkPattern(const uint8_t* pattern, size_t length, unsigned long interval_ms = 200) {
        for (size_t i = 0; i < length; i++) {
            if (pattern[i]) on(); else off();
            delay(interval_ms);
        }
        off();
    }

    bool getState() { return state; }
};

// ===== ERROR HANDLING =====

enum ErrorCode {
    ERROR_NONE = 0,
    ERROR_MICROPHONE_INIT = 1,
    ERROR_TFLM_INIT = 2,
    ERROR_MEMORY_ALLOCATION = 3,
    ERROR_AUDIO_PROCESSING = 4,
    ERROR_MODEL_INFERENCE = 5,
    ERROR_LOW_MEMORY = 6
};

class ErrorHandler {
private:
    static ErrorCode last_error;
    static unsigned long last_error_time;
    static int retry_count;

public:
    static void setError(ErrorCode error) {
        last_error = error;
        last_error_time = millis();
        retry_count = 0;

        DEBUG_PRINTF("ERROR: Code %d occurred at %lu ms\n", error, last_error_time);

        // Handle error based on type
        switch (error) {
            case ERROR_LOW_MEMORY:
                MemoryMonitor::printMemoryInfo();
                break;
            case ERROR_MICROPHONE_INIT:
                DEBUG_PRINTLN("Microphone initialization failed");
                break;
            case ERROR_TFLM_INIT:
                DEBUG_PRINTLN("TensorFlow Lite Micro initialization failed");
                break;
            default:
                break;
        }
    }

    static ErrorCode getLastError() { return last_error; }

    static bool shouldRetry() {
        return retry_count < MAX_RETRIES && (millis() - last_error_time) > ERROR_BACKOFF_MS;
    }

    static void incrementRetry() { retry_count++; }

    static void clearError() {
        last_error = ERROR_NONE;
        retry_count = 0;
    }

    static const char* getErrorMessage(ErrorCode error) {
        switch (error) {
            case ERROR_NONE: return "No error";
            case ERROR_MICROPHONE_INIT: return "Microphone initialization failed";
            case ERROR_TFLM_INIT: return "TFLM initialization failed";
            case ERROR_MEMORY_ALLOCATION: return "Memory allocation failed";
            case ERROR_AUDIO_PROCESSING: return "Audio processing failed";
            case ERROR_MODEL_INFERENCE: return "Model inference failed";
            case ERROR_LOW_MEMORY: return "Low memory condition";
            default: return "Unknown error";
        }
    }
};

// Initialize static members
ErrorCode ErrorHandler::last_error = ERROR_NONE;
unsigned long ErrorHandler::last_error_time = 0;
int ErrorHandler::retry_count = 0;

// ===== SYSTEM UTILITIES =====

class SystemUtils {
public:
    static void printSystemInfo() {
        DEBUG_PRINTLN("\n=== System Information ===");
        DEBUG_PRINTF("ESP32 Chip model: %s\n", ESP.getChipModel());
        DEBUG_PRINTF("ESP32 Chip revision: %d\n", ESP.getChipRevision());
        DEBUG_PRINTF("SDK version: %s\n", ESP.getSdkVersion());
        DEBUG_PRINTF("CPU frequency: %d MHz\n", getCpuFrequencyMhz());
        DEBUG_PRINTF("Flash size: %d MB\n", ESP.getFlashChipSize() / (1024 * 1024));
        DEBUG_PRINTF("PSRAM size: %d MB\n", ESP.getPsramSize() / (1024 * 1024));
    }

    static void setupPowerManagement() {
        // Reduce CPU frequency for power savings
        setCpuFrequencyMhz(CPU_FREQUENCY_MHZ);

        // Disable unused peripherals
        WiFi.mode(WIFI_OFF);
        btStop();

        DEBUG_PRINTLN("Power management configured");
    }

    static void enterLightSleep(unsigned long sleep_time_us = LIGHT_SLEEP_DELAY_US) {
        esp_sleep_enable_timer_wakeup(sleep_time_us);
        esp_light_sleep_start();
    }
};

// ===== TIMER UTILITIES =====

class Timer {
private:
    unsigned long start_time;
    bool running;

public:
    Timer() : start_time(0), running(false) {}

    void start() {
        start_time = millis();
        running = true;
    }

    void stop() {
        running = false;
    }

    unsigned long elapsed() {
        if (running) {
            return millis() - start_time;
        }
        return 0;
    }

    void reset() {
        start_time = millis();
    }

    bool isRunning() { return running; }
};

#endif // UTILS_H
