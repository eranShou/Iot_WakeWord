/*
 * ESP32-S3 Wake Word Detection System
 * Real-time wake word detection with continuous PDM microphone listening
 * Based on ESP_I2S library and TFLite Micro - no magic numbers
 * 
 * Hardware: Seeed Studio XIAO ESP32S3
 * Microphone: Built-in PDM microphone (MSM261D3526H1CPM)
 * Model: 5-class Hebrew wake word detection (lehitraoot, shalom, bait, background, unknown)
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

// Inference probabilities (5 classes)
float* probabilities;

// WAV file buffer for transmission
uint8_t* wavBuffer;

// ============================================================================
// STATE VARIABLES
// ============================================================================

unsigned long lastDetectionTime = 0;
unsigned long inferenceCount = 0;
unsigned long detectionCount = 0;
unsigned long recordingCount = 0;  // Counter for recording numbers
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
    Serial.printf("WAV buffer: %d bytes\n", WAV_FILE_SIZE);
    
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
    Serial.println("Starting continuous wake word detection...");
    Serial.println("Listening for Hebrew wake words:");
    for (int i = 0; i < NUM_CLASSES; i++) {
        Serial.printf("  %d: %s\n", i, CLASS_LABELS[i]);
    }
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
        Serial.printf("*** NEW WINDOW READY #%lu ***\n", inferenceCount + 1);
        Serial.println("Attempting to extract audio window...");
        // Extract 1-second audio window
        if (audioProvider.getNextWindow(audioWindow, AUDIO_WINDOW_SIZE)) {
            Serial.println("Window extracted successfully, processing...");
            // Process audio window
            processAudioWindow();
        } else {
            Serial.println("WARNING: Failed to extract audio window - not enough samples in buffer");
        }
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
    
    // Print inference results
    Serial.printf("Inference #%lu: %s (%.3f)\n", 
                  inferenceCount, CLASS_LABELS[predictedClass], confidence);
    
    // Print all confidence scores
    Serial.print("Confidences: [");
    for (int i = 0; i < NUM_CLASSES; i++) {
        Serial.print(probabilities[i], 3);
        if (i < NUM_CLASSES - 1) Serial.print(", ");
    }
    Serial.println("]");
    
    // Send audio file for every inference
    detectionCount++;
    recordingCount++;  // Increment recording counter
    unsigned long currentTime = millis();
    lastDetectionTime = currentTime;
    
    Serial.printf("*** PREDICTION #%lu ***\n", detectionCount);
    Serial.printf("Class: %s, Confidence: %.3f\n", 
                  CLASS_LABELS[predictedClass], confidence);
    
    // Send audio file with prediction
    sendAudioFile(audioWindow, predictedClass, confidence);
    
    // Blink LED for visual feedback
    #ifdef LED_PIN
    ledState = true;
    digitalWrite(LED_PIN, ledState);
    ledBlinkTime = currentTime;
    #endif
    
    // Print audio level for monitoring
    float audioLevel = audioProvider.getAudioLevel();
    if (inferenceCount % 10 == 0) { // Print every 10 inferences
        Serial.printf("Audio level: %.3f (max: %.3f)\n", 
                      audioLevel, audioProvider.getMaxLevel());
        Serial.println("Audio processing: Gain + Compression applied to window");
    }
}

// ============================================================================
// AUDIO FILE TRANSMISSION
// ============================================================================

void sendAudioFile(const int16_t* audioData, int predictedClass, float confidence) {
    Serial.println("Preparing to send audio file...");
    
    // Create WAV header
    createWAVHeader(wavBuffer, AUDIO_WINDOW_SIZE, SAMPLE_RATE, NUM_CHANNELS);
    
    // Copy audio data to WAV buffer (after header)
    int16_t* wavAudioData = (int16_t*)(wavBuffer + WAV_HEADER_SIZE);
    memcpy(wavAudioData, audioData, AUDIO_WINDOW_SIZE * sizeof(int16_t));
    
    // Generate filename with all confidence scores
    char filename[MAX_FILENAME_LENGTH];
    generateFilename(filename, CLASS_LABELS[predictedClass], probabilities, confidence);
    
    Serial.printf("Sending: %s\n", filename);
    
    // Send start marker
    uint32_t marker = START_MARKER;
    Serial.write((uint8_t*)&marker, sizeof(marker));
    
    // Send filename
    Serial.print("FILENAME:");
    Serial.println(filename);
    
    // Send data size
    Serial.print("DATA_SIZE:");
    Serial.println(WAV_FILE_SIZE);
    
    // Flush to ensure text is sent before binary data
    Serial.flush();
    delay(100);
    
    // Send WAV file data in chunks
    size_t chunkSize = 1024;
    for (size_t i = 0; i < WAV_FILE_SIZE; i += chunkSize) {
        size_t currentChunk = min(chunkSize, WAV_FILE_SIZE - i);
        Serial.write(wavBuffer + i, currentChunk);
        Serial.flush();
    }
    
    // Send end marker
    marker = END_MARKER;
    Serial.write((uint8_t*)&marker, sizeof(marker));
    Serial.flush();
    
    Serial.println("Audio file sent successfully!");
}

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
    
    // Allocate WAV buffer
    wavBuffer = (uint8_t*)ps_malloc(WAV_FILE_SIZE);
    if (!wavBuffer) wavBuffer = (uint8_t*)malloc(WAV_FILE_SIZE);
    
    // Check if all allocations succeeded
    if (!audioWindow || !spectrogramBuffer || !probabilities || !wavBuffer) {
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
    if (wavBuffer) { free(wavBuffer); wavBuffer = nullptr; }
}

void createWAVHeader(uint8_t* header, uint32_t sampleCount, uint32_t sampleRate, uint16_t channels) {
    uint32_t byteRate = sampleRate * channels * sizeof(int16_t);
    uint32_t dataSize = sampleCount * channels * sizeof(int16_t);
    uint32_t fileSize = WAV_HEADER_SIZE + dataSize;
    
    // RIFF header
    header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
    *(uint32_t*)(header + 4) = fileSize - 8;
    header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';
    
    // fmt chunk
    header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
    *(uint32_t*)(header + 16) = 16; // fmt chunk size
    *(uint16_t*)(header + 20) = 1;  // PCM format
    *(uint16_t*)(header + 22) = channels;
    *(uint32_t*)(header + 24) = sampleRate;
    *(uint32_t*)(header + 28) = byteRate;
    *(uint16_t*)(header + 32) = channels * sizeof(int16_t); // block align
    *(uint16_t*)(header + 34) = 16; // bits per sample
    
    // data chunk
    header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
    *(uint32_t*)(header + 40) = dataSize;
}

void generateFilename(char* filename, const char* className, const float* probs, float maxConf) {
    // Create filename: recordingNumber_class_conf[all_scores].wav
    snprintf(filename, MAX_FILENAME_LENGTH, "%03lu_%s_conf[", 
             recordingCount, className);
    
    // Append all confidence scores
    char* pos = filename + strlen(filename);
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (i > 0) {
            *pos++ = ',';
            *pos++ = ' ';
        }
        pos += sprintf(pos, "%.3f", probs[i]);
    }
    
    // Close bracket and add .wav
    *pos++ = ']';
    *pos++ = '.';
    *pos++ = 'w';
    *pos++ = 'a';
    *pos++ = 'v';
    *pos = '\0';
}

#ifdef LED_PIN
void handleLEDBlinking() {
    if (ledState && (millis() - ledBlinkTime > 200)) {
        ledState = false;
        digitalWrite(LED_PIN, ledState);
    }
}
#endif
