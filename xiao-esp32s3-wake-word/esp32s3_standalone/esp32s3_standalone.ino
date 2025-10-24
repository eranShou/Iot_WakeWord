/*
 * ESP32-S3 Standalone Wake Word Detection System
 * Real-time wake word detection with continuous PDM microphone listening
 * Based on ESP_I2S library and TFLite Micro - no magic numbers
 * 
 * Hardware: Seeed Studio XIAO ESP32S3
 * Microphone: Built-in PDM microphone (MSM261D3526H1CPM)
 * Model: 4-class Hebrew wake word detection (lehitraoot, shalom, background, unknown)
 * 
 * Standalone operation: No PC transmission, local inference only
 * Output: Serial logging + LED visual feedback
 */

#include "config.h"
#include "model_config.h"
#include "mic_config.h"
#include "audio_provider.h"
#include "spectrogram_extractor.h"
#include "inference_engine.h"

// ============================================================================
// GLOBAL OBJECTS
// ============================================================================

AudioProvider audioProvider;
SpectrogramExtractor spectrogramExtractor;
InferenceEngine inferenceEngine;

// ============================================================================
// WORKSPACE BUFFERS
// ============================================================================

// Audio window buffer (1 second of 16-bit mono audio)
int16_t* audioWindow;

// Spectrogram buffer (32x32x1 float)
float* spectrogramBuffer;

// Inference probabilities (4 classes)
float* probabilities;

// ============================================================================
// STATE VARIABLES
// ============================================================================

unsigned long lastDetectionTime = 0;
unsigned long inferenceCount = 0;
unsigned long detectionCount = 0;
bool systemReady = false;

// ============================================================================
// LED CONFIGURATION
// ============================================================================

#ifdef LED_PIN
bool ledState = false;
unsigned long ledBlinkTime = 0;
#endif

// ============================================================================
// SETUP FUNCTION
// ============================================================================

void setup() {
    Serial.begin(SERIAL_BAUD_RATE);
    delay(2000);
    
    Serial.println();
    Serial.println("========================================");
    Serial.println("ESP32-S3 Wake Word Detection System");
    Serial.println("========================================");
    
    // Initialize LED pin
    #ifdef LED_PIN
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
    #endif
    
    // Allocate workspace buffers
    if (!allocateWorkspace()) {
        Serial.println("ERROR: Failed to allocate workspace buffers");
        while (1) {
            delay(1000);
        }
    }
    
    Serial.println("Allocating workspace buffers...");
    Serial.printf("Audio window: %d bytes\n", AUDIO_WINDOW_SIZE * sizeof(int16_t));
    Serial.printf("Spectrogram: %d bytes\n", SPECTROGRAM_BUFFER_SIZE * sizeof(float));
    Serial.printf("Probabilities: %d bytes\n", NUM_CLASSES * sizeof(float));
    
    // Initialize audio provider
    Serial.println("Initializing audio provider...");
    if (!audioProvider.init()) {
        Serial.println("ERROR: Failed to initialize audio provider");
        while (1) {
            delay(1000);
        }
    }
    
    // Initialize spectrogram extractor
    Serial.println("Initializing spectrogram extractor...");
    if (!spectrogramExtractor.init()) {
        Serial.println("ERROR: Failed to initialize spectrogram extractor");
        while (1) {
            delay(1000);
        }
    }
    
    // Initialize inference engine
    Serial.println("Initializing inference engine...");
    if (!inferenceEngine.init()) {
        Serial.println("ERROR: Failed to initialize inference engine");
        while (1) {
            delay(1000);
        }
    }
    
    systemReady = true;
    
    Serial.println();
    Serial.println("System initialized successfully!");
    Serial.printf("Model size: %d bytes (%.2f KB)\n", 
                  inferenceEngine.getModelSize(), 
                  inferenceEngine.getModelSize() / 1024.0f);
    Serial.printf("Tensor arena: %d bytes (%.2f KB)\n", 
                  inferenceEngine.getTensorArenaSize(),
                  inferenceEngine.getTensorArenaSize() / 1024.0f);
    Serial.printf("Confidence threshold: %.2f\n", DETECTION_CONFIDENCE_THRESHOLD);
    Serial.printf("Window stride: %d ms\n", WINDOW_STRIDE_MS);
    Serial.println();
    Serial.println("Audio Processing Settings (same as audio_recorder_esp32_pc.ino):");
    Serial.printf("  Sample rate: %d Hz\n", SAMPLE_RATE);
    Serial.printf("  Audio gain: %.1fx (enabled: %s)\n", DEFAULT_GAIN, GAIN_ENABLED_BY_DEFAULT ? "YES" : "NO");
    Serial.printf("  Compression: %s (threshold: %.2f, ratio: %.1f:1)\n", 
                  COMPRESSION_ENABLED_BY_DEFAULT ? "ENABLED" : "DISABLED",
                  COMPRESSION_THRESHOLD, COMPRESSION_RATIO);
    Serial.printf("  PDM pins: CLK=%d, DATA=%d\n", PDM_CLK_PIN, PDM_DATA_PIN);
    Serial.println();
    Serial.println("Filling audio buffer...");
    
    // Wait for audio buffer to fill up before starting detection
    Serial.println("Waiting for audio buffer to fill...");
    unsigned long bufferFillStart = millis();
    int samplesReceived = 0;
    while (millis() - bufferFillStart < 5000) { // Wait up to 5 seconds
        audioProvider.update();
        samplesReceived++;
        if (samplesReceived % 100 == 0) {
            Serial.print(".");
        }
        delay(10);
    }
    
    Serial.println();
    Serial.println("Audio buffer ready!");
    Serial.println("Starting continuous wake word detection...");
    Serial.println("Listening for Hebrew wake words:");
    for (int i = 0; i < NUM_CLASSES; i++) {
        Serial.printf("  %d: %s\n", i, CLASS_LABELS[i]);
    }
    Serial.printf("Detection threshold: %.2f\n", DETECTION_CONFIDENCE_THRESHOLD);
    Serial.println("Standalone mode: Local inference only, no PC transmission");
    Serial.println();
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
    if (!systemReady) {
        delay(100);
        return;
    }
    
    // Update audio capture (continuous)
    audioProvider.update();
    
    // Check if new window is ready for inference
    if (audioProvider.hasNewWindow()) {
        // Extract 1-second audio window
        if (audioProvider.getNextWindow(audioWindow, AUDIO_WINDOW_SIZE)) {
            // Process audio window
            processAudioWindow();
        }
        // If window extraction failed, just continue - buffer will fill up eventually
    }
    
    // Handle LED blinking for detections
    #ifdef LED_PIN
    handleLEDBlinking();
    #endif
    
    // Small delay to prevent overwhelming the system
    delay(10);
}

// ============================================================================
// AUDIO PROCESSING
// ============================================================================

void processAudioWindow() {
    inferenceCount++;
    
    // Compute spectrogram from audio window
    if (!spectrogramExtractor.computeSTFT(audioWindow, spectrogramBuffer)) {
        Serial.println("ERROR: Failed to compute spectrogram");
        return;
    }
    
    // Run inference
    if (!inferenceEngine.runInference(spectrogramBuffer, probabilities)) {
        Serial.println("ERROR: Failed to run inference");
        return;
    }
    
    // Get best class and confidence
    float confidence;
    int predictedClass = inferenceEngine.getBestClass(probabilities, confidence);
    
    // Check if detection meets confidence threshold and is not background
    if (confidence >= DETECTION_CONFIDENCE_THRESHOLD && predictedClass != CLASS_NOISE) {
        detectionCount++;
        unsigned long currentTime = millis();
        lastDetectionTime = currentTime;
        
        Serial.printf("*** WAKE WORD DETECTED #%lu ***\n", detectionCount);
        Serial.printf("Class: %s, Confidence: %.3f (threshold: %.3f)\n", 
                      CLASS_LABELS[predictedClass], confidence, DETECTION_CONFIDENCE_THRESHOLD);
        
        // Print all confidence scores for detected wake word
        Serial.print("All confidences: [");
        for (int i = 0; i < NUM_CLASSES; i++) {
            Serial.print(probabilities[i], 3);
            if (i < NUM_CLASSES - 1) Serial.print(", ");
        }
        Serial.println("]");
        
        // Blink LED for visual feedback
        #ifdef LED_PIN
        ledState = true;
        digitalWrite(LED_PIN, ledState);
        ledBlinkTime = currentTime;
        #endif
    }
}

// ============================================================================
// STANDALONE DETECTION (No PC transmission)
// ============================================================================

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

bool allocateWorkspace() {
    // Allocate audio window buffer
    audioWindow = (int16_t*)ps_malloc(AUDIO_WINDOW_SIZE * sizeof(int16_t));
    if (!audioWindow) audioWindow = (int16_t*)malloc(AUDIO_WINDOW_SIZE * sizeof(int16_t));
    
    // Allocate spectrogram buffer
    spectrogramBuffer = (float*)ps_malloc(SPECTROGRAM_BUFFER_SIZE * sizeof(float));
    if (!spectrogramBuffer) spectrogramBuffer = (float*)malloc(SPECTROGRAM_BUFFER_SIZE * sizeof(float));
    
    // Allocate probabilities buffer
    probabilities = (float*)ps_malloc(NUM_CLASSES * sizeof(float));
    if (!probabilities) probabilities = (float*)malloc(NUM_CLASSES * sizeof(float));
    
    // Check if all allocations succeeded
    if (!audioWindow || !spectrogramBuffer || !probabilities) {
        Serial.println("ERROR: Failed to allocate workspace buffers");
        freeWorkspace();
        return false;
    }
    
    return true;
}

void freeWorkspace() {
    if (audioWindow) { free(audioWindow); audioWindow = nullptr; }
    if (spectrogramBuffer) { free(spectrogramBuffer); spectrogramBuffer = nullptr; }
    if (probabilities) { free(probabilities); probabilities = nullptr; }
}

// WAV file functions removed for standalone operation

#ifdef LED_PIN
void handleLEDBlinking() {
    if (ledState && (millis() - ledBlinkTime > 200)) {
        ledState = false;
        digitalWrite(LED_PIN, ledState);
    }
}
#endif
