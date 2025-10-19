/*
 * Inference Engine Header
 * Simplified inference engine for wake word detection
 * TODO: Integrate with TensorFlow Lite Micro when library compatibility is resolved
 * Based on model_config.h settings - no magic numbers
 */

#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <Arduino.h>
#include "config.h"
#include "model_config.h"

class InferenceEngine {
private:
    // Simplified inference components (placeholder for TFLite integration)
    bool isInitialized;
    
    // Mock inference results for testing
    float mock_probabilities[NUM_CLASSES];
    int mock_inference_count;
    
public:
    InferenceEngine();
    ~InferenceEngine();
    
    // Initialize inference engine (simplified version)
    bool init();
    
    // Run inference on spectrogram (mock implementation for testing)
    bool runInference(const float* spectrogram, float* probabilities);
    
    // Get best class prediction and confidence
    int getBestClass(const float* probabilities, float& confidence);
    
    // Check if inference engine is ready
    bool isReady() const { return isInitialized; }
    
    // Get model info (placeholder)
    size_t getModelSize() const { return 2121588; } // 2.07 MB
    
    // Get tensor arena size (placeholder)
    size_t getTensorArenaSize() const { return TFLITE_TENSOR_ARENA_SIZE; }
    
private:
    // Generate mock probabilities for testing (simulates different wake words)
    void generateMockProbabilities(float* probabilities);
};

// ============================================================================
// IMPLEMENTATION
// ============================================================================

InferenceEngine::InferenceEngine() 
    : isInitialized(false)
    , mock_inference_count(0) {
}

InferenceEngine::~InferenceEngine() {
    // Simplified destructor - no dynamic memory to free
}

bool InferenceEngine::init() {
    Serial.println("Initializing InferenceEngine (Simplified Version)...");
    
    // Initialize mock probabilities
    for (int i = 0; i < NUM_CLASSES; i++) {
        mock_probabilities[i] = 0.0f;
    }
    
    isInitialized = true;
    
    Serial.println("InferenceEngine initialized successfully (Mock Mode)");
    Serial.printf("Input shape: [1, %d, %d, %d]\n", MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_CHANNELS);
    Serial.printf("Output shape: [1, %d]\n", NUM_CLASSES);
    Serial.printf("Model size: %d bytes\n", getModelSize());
    Serial.println("NOTE: Using mock inference for testing. TFLite integration pending library compatibility fix.");
    
    return true;
}

bool InferenceEngine::runInference(const float* spectrogram, float* probabilities) {
    if (!isInitialized || !spectrogram || !probabilities) {
        return false;
    }
    
    // Generate mock probabilities for testing
    generateMockProbabilities(probabilities);
    
    mock_inference_count++;
    
    return true;
}

int InferenceEngine::getBestClass(const float* probabilities, float& confidence) {
    if (!probabilities) {
        return -1;
    }
    
    int best_class = 0;
    float max_prob = probabilities[0];
    
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (probabilities[i] > max_prob) {
            max_prob = probabilities[i];
            best_class = i;
        }
    }
    
    confidence = max_prob;
    return best_class;
}

void InferenceEngine::generateMockProbabilities(float* probabilities) {
    // Generate realistic-looking probabilities that cycle through different wake words
    // This simulates the behavior of the actual model for testing
    
    // Cycle through different classes every few inferences
    int cycle = (mock_inference_count / 3) % NUM_CLASSES;
    
    // Set base probabilities
    for (int i = 0; i < NUM_CLASSES; i++) {
        probabilities[i] = 0.1f + (random(100) / 1000.0f); // 0.1-0.2 baseline
    }
    
    // Make one class more prominent
    probabilities[cycle] = 0.6f + (random(300) / 1000.0f); // 0.6-0.9 for dominant class
    
    // Normalize probabilities
    float sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        sum += probabilities[i];
    }
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        probabilities[i] /= sum;
    }
}

#endif // INFERENCE_ENGINE_H
