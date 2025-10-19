/*
 * Audio Provider Header
 * PDM Microphone and Circular Buffer Management
 * Based on ESP_I2S library (same as audio_recorder_esp32_pc.ino) - no magic numbers
 */

#ifndef AUDIO_PROVIDER_H
#define AUDIO_PROVIDER_H

#include "ESP_I2S.h"
#include <Arduino.h>
#include <math.h>
#include <stdint.h>
#include "config.h"
#include "model_config.h"
#include "mic_config.h"

class AudioProvider {
private:
    // Circular buffer for continuous audio capture
    int16_t* audioBuffer;
    size_t bufferWriteIndex;
    size_t bufferReadIndex;
    size_t bufferSize;
    
    // ESP_I2S instance
    I2SClass i2s;
    
    // Window extraction state
    unsigned long lastWindowTime;
    bool isInitialized;
    
    // Audio level monitoring
    float currentRMS;
    float maxLevel;
    
    // Audio processing settings (same as audio_recorder_esp32_pc.ino)
    float current_gain;
    bool gain_enabled;
    bool compression_enabled;
    float compression_threshold;
    float compression_ratio;
    float compression_makeup_gain;
    float compression_attack_coeff;
    float compression_release_coeff;
    float compression_envelope;
    
public:
    AudioProvider();
    ~AudioProvider();
    
    // Initialize PDM microphone and circular buffer
    bool init();
    
    // Continuous audio capture (call in loop)
    void update();
    
    // Check if new window is ready for inference
    bool hasNewWindow();
    
    // Extract 1-second audio window for inference
    bool getNextWindow(int16_t* windowBuffer, size_t samples);
    
    // Get current audio level for monitoring
    float getAudioLevel() const { return currentRMS; }
    float getMaxLevel() const { return maxLevel; }
    
    // Check if audio provider is initialized
    bool isReady() const { return isInitialized; }
    
    // Reset audio levels
    void resetLevels() { maxLevel = 0.0f; }
    
    // Audio processing methods (same as audio_recorder_esp32_pc.ino)
    void applyAudioCompressionAndGain(int16_t* audio_data, uint32_t audio_size);
    
private:
    // Calculate RMS audio level
    float calculateRMS(const int16_t* audio, size_t length);
    
    // Allocate circular buffer
    bool allocateBuffer();
    
    // Free circular buffer
    void freeBuffer();
};

// ============================================================================
// IMPLEMENTATION
// ============================================================================

AudioProvider::AudioProvider() 
    : audioBuffer(nullptr)
    , bufferWriteIndex(0)
    , bufferReadIndex(0)
    , bufferSize(AUDIO_BUFFER_SIZE)
    , lastWindowTime(0)
    , isInitialized(false)
    , currentRMS(0.0f)
    , maxLevel(0.0f)
    , current_gain(DEFAULT_GAIN)
    , gain_enabled(GAIN_ENABLED_BY_DEFAULT)
    , compression_enabled(COMPRESSION_ENABLED_BY_DEFAULT)
    , compression_threshold(COMPRESSION_THRESHOLD)
    , compression_ratio(COMPRESSION_RATIO)
    , compression_makeup_gain(COMPRESSION_MAKEUP_GAIN)
    , compression_attack_coeff(1.0f - exp(-1.0f / (COMPRESSION_ATTACK_MS * SAMPLE_RATE / 1000.0f)))
    , compression_release_coeff(1.0f - exp(-1.0f / (COMPRESSION_RELEASE_MS * SAMPLE_RATE / 1000.0f)))
    , compression_envelope(0.0f) {
}

AudioProvider::~AudioProvider() {
    freeBuffer();
}

bool AudioProvider::init() {
    Serial.println("Initializing AudioProvider...");
    
    // Allocate circular buffer
    if (!allocateBuffer()) {
        Serial.println("ERROR: Failed to allocate audio buffer");
        return false;
    }
    
    // Set PDM pins using configuration (same as audio_recorder)
    i2s.setPinsPdmRx(PDM_CLK_PIN, PDM_DATA_PIN);
    
    // Initialize PDM RX mode with 16-bit mono (same as audio_recorder)
    if (!i2s.begin(I2S_MODE_PDM_RX, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
        Serial.println("ERROR: Failed to initialize ESP_I2S PDM microphone");
        Serial.println("Make sure you selected XIAO_ESP32S3 board.");
        freeBuffer();
        return false;
    }
    
    isInitialized = true;
    lastWindowTime = millis();
    
    Serial.printf("AudioProvider initialized: %d samples buffer\n", bufferSize);
    Serial.printf("Sample rate: %d Hz, Channels: %d\n", SAMPLE_RATE, NUM_CHANNELS);
    Serial.println("PDM microphone ready");
    
    return true;
}

void AudioProvider::update() {
    if (!isInitialized) return;
    
    // Read available I2S data using ESP_I2S library
    int16_t tempBuffer[1024]; // Increased buffer size for better throughput
    int samplesRead = 0;
    
    // Read data from I2S - ESP_I2S library read() returns single sample
    // We need to read multiple samples in a loop
    for (int i = 0; i < 1024; i++) {
        if (i2s.available()) {
            int16_t sample = i2s.read();
            tempBuffer[i] = sample;
            samplesRead++;
            
            // Write sample to circular buffer
            audioBuffer[bufferWriteIndex] = sample;
            bufferWriteIndex = (bufferWriteIndex + 1) % bufferSize;
        } else {
            break; // No more data available
        }
    }
    
    // Debug: Print samples read occasionally
    static unsigned long lastDebugTime = 0;
    static int totalSamplesRead = 0;
    totalSamplesRead += samplesRead;
    
    if (millis() - lastDebugTime > 1000) { // Every second
        Serial.printf("DEBUG: Samples read this update: %d, Total samples/sec: %d\n", samplesRead, totalSamplesRead);
        totalSamplesRead = 0;
        lastDebugTime = millis();
    }
    
    // Update audio level if we got some samples
    if (samplesRead > 0) {
        currentRMS = calculateRMS(tempBuffer, samplesRead);
        if (currentRMS > maxLevel) {
            maxLevel = currentRMS;
        }
    }
}

bool AudioProvider::hasNewWindow() {
    if (!isInitialized) return false;
    
    unsigned long currentTime = millis();
    bool ready = (currentTime - lastWindowTime) >= WINDOW_STRIDE_MS;
    
    if (ready) {
        Serial.printf("DEBUG: Window ready! Time since last: %lu ms (target: %d ms)\n", 
                     currentTime - lastWindowTime, WINDOW_STRIDE_MS);
    }
    
    return ready;
}

bool AudioProvider::getNextWindow(int16_t* windowBuffer, size_t samples) {
    if (!isInitialized || !windowBuffer || samples != NUM_SAMPLES) {
        Serial.println("ERROR: getNextWindow failed - not initialized or invalid parameters");
        return false;
    }
    
    // Check if we have enough data in buffer
    size_t availableSamples = (bufferWriteIndex >= bufferReadIndex) 
        ? (bufferWriteIndex - bufferReadIndex)
        : (bufferSize - bufferReadIndex + bufferWriteIndex);
    
    Serial.printf("DEBUG: Available samples: %d, Required: %d\n", availableSamples, samples);
    
    if (availableSamples < samples) {
        Serial.printf("WARNING: Not enough samples in buffer (%d < %d)\n", availableSamples, samples);
        return false;
    }
    
    // Extract window from circular buffer
    for (size_t i = 0; i < samples; i++) {
        windowBuffer[i] = audioBuffer[bufferReadIndex];
        bufferReadIndex = (bufferReadIndex + 1) % bufferSize;
    }
    
    // Apply audio processing (gain and compression) - same as audio_recorder_esp32_pc.ino
    if ((gain_enabled && current_gain > 1.0f) || compression_enabled) {
        applyAudioCompressionAndGain(windowBuffer, samples * sizeof(int16_t));
    }
    
    // Advance read index by stride amount for sliding window (not full window size)
    // This creates the intended 50% overlap between windows
    size_t oldReadIndex = bufferReadIndex;
    bufferReadIndex = (bufferReadIndex + WINDOW_STRIDE_SAMPLES) % bufferSize;
    
    Serial.printf("DEBUG: Sliding window - Old read index: %d, New read index: %d, Stride: %d\n", 
                  oldReadIndex, bufferReadIndex, WINDOW_STRIDE_SAMPLES);
    
    lastWindowTime = millis();
    return true;
}

float AudioProvider::calculateRMS(const int16_t* audio, size_t length) {
    if (length == 0) return 0.0f;
    
    float sum = 0.0f;
    for (size_t i = 0; i < length; i++) {
        float sample = audio[i] / 32768.0f; // Normalize to [-1, 1]
        sum += sample * sample;
    }
    
    return sqrt(sum / length);
}

bool AudioProvider::allocateBuffer() {
    // Allocate buffer in PSRAM if available, otherwise in heap
    audioBuffer = (int16_t*)ps_malloc(bufferSize * sizeof(int16_t));
    if (!audioBuffer) {
        audioBuffer = (int16_t*)malloc(bufferSize * sizeof(int16_t));
    }
    
    if (!audioBuffer) {
        Serial.println("ERROR: Failed to allocate audio buffer memory");
        return false;
    }
    
    // Initialize buffer to zero
    memset(audioBuffer, 0, bufferSize * sizeof(int16_t));
    
    return true;
}

void AudioProvider::freeBuffer() {
    if (audioBuffer) {
        free(audioBuffer);
        audioBuffer = nullptr;
    }
}

// Audio processing function (copied from audio_recorder_esp32_pc.ino)
void AudioProvider::applyAudioCompressionAndGain(int16_t* audio_data, uint32_t audio_size) {
    // Process 16-bit samples (2 bytes per sample)
    int16_t* samples = audio_data;
    uint32_t sample_count = audio_size / BYTES_PER_SAMPLE;
    
    // Reset compression envelope for new recording
    compression_envelope = 0.0f;
    
    for (uint32_t i = 0; i < sample_count; i++) {
        float sample = (float)samples[i];
        
        // Step 1: Apply initial gain
        if (gain_enabled && current_gain > 1.0f) {
            sample *= current_gain;
        }
        
        // Step 2: Apply compression if enabled
        if (compression_enabled) {
            // Calculate input level (absolute value normalized to 0-1)
            float input_level = abs(sample) / 32767.0f;
            
            // Update compression envelope with attack/release
            if (input_level > compression_envelope) {
                // Attack phase
                compression_envelope += compression_attack_coeff * (input_level - compression_envelope);
            } else {
                // Release phase
                compression_envelope += compression_release_coeff * (input_level - compression_envelope);
            }
            
            // Apply compression if above threshold
            if (compression_envelope > compression_threshold) {
                float over_threshold = compression_envelope - compression_threshold;
                float compression_factor = 1.0f - (over_threshold * (1.0f - 1.0f/compression_ratio));
                sample *= compression_factor;
            }
            
            // Apply makeup gain after compression
            sample *= compression_makeup_gain;
        }
        
        // Step 3: Final limiting to prevent clipping
        if (sample > AUDIO_MAX_POSITIVE) {
            sample = AUDIO_MAX_POSITIVE;
        } else if (sample < AUDIO_MAX_NEGATIVE) {
            sample = AUDIO_MAX_NEGATIVE;
        }
        
        // Convert back to 16-bit integer
        samples[i] = (int16_t)sample;
    }
}

#endif // AUDIO_PROVIDER_H
