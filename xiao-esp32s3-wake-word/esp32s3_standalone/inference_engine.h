/*
 * Inference Engine Header
 * TensorFlow Lite Micro inference for wake word detection
 * Uses model settings from model_config.h and runtime limits from config.h
 */

#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <Arduino.h>
#include "config.h"
#include "model_config.h"
#include "wake_word_model.h"

// Workaround for FlatBuffers span assignment issue in some Arduino builds
#ifndef FLATBUFFERS_SPAN_MINIMAL
#define FLATBUFFERS_SPAN_MINIMAL
#endif

// TFLite Micro includes
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

class InferenceEngine {
private:
    bool isInitialized;

    // TFLite Micro state
    const tflite::Model* model;
    tflite::AllOpsResolver resolver;
    tflite::MicroAllocator* allocator;
    tflite::MicroErrorReporter errorReporter;
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input;
    TfLiteTensor* output;

    // Aligned tensor arena
    alignas(16) static uint8_t tensorArena[TFLITE_TENSOR_ARENA_SIZE];

public:
    InferenceEngine();
    ~InferenceEngine();

    // Initialize TFLite Micro
    bool init();

    // Run inference on a 32x32x1 float spectrogram
    bool runInference(const float* spectrogram, float* probabilities);

    // Get best class prediction and confidence
    int getBestClass(const float* probabilities, float& confidence);

    // Status and info
    bool isReady() const { return isInitialized; }
    size_t getModelSize() const { return sizeof(wake_word_model); }
    size_t getTensorArenaSize() const { return TFLITE_TENSOR_ARENA_SIZE; }
};

// ============================================================================
// IMPLEMENTATION
// ============================================================================

uint8_t InferenceEngine::tensorArena[TFLITE_TENSOR_ARENA_SIZE];

InferenceEngine::InferenceEngine() 
    : isInitialized(false)
    , model(nullptr)
    , resolver()
    , allocator(nullptr)
    , interpreter(nullptr)
    , input(nullptr)
    , output(nullptr) {
}

InferenceEngine::~InferenceEngine() {
    if (interpreter) { delete interpreter; interpreter = nullptr; }
    // allocator is created in tensor arena; no delete
    model = nullptr;
    input = nullptr;
    output = nullptr;
}

bool InferenceEngine::init() {
    Serial.println("Initializing InferenceEngine (TFLite Micro)...");

    // Map the model
    model = tflite::GetModel(wake_word_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("ERROR: Model schema %d != supported %d\n", model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // Create allocator on the tensor arena
    allocator = tflite::MicroAllocator::Create(tensorArena, sizeof(tensorArena), &errorReporter);

    // Create interpreter (allocator-based API)
    interpreter = new tflite::MicroInterpreter(model, resolver, allocator, &errorReporter, /*resource_vars=*/nullptr, /*profiler=*/nullptr);

    // Allocate tensors
    TfLiteStatus allocStatus = interpreter->AllocateTensors();
    if (allocStatus != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors failed");
        return false;
    }

    // Get input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Validate expected shapes
    if (input->dims->size != 4 ||
        input->dims->data[1] != MODEL_INPUT_HEIGHT ||
        input->dims->data[2] != MODEL_INPUT_WIDTH ||
        input->dims->data[3] != MODEL_INPUT_CHANNELS) {
        Serial.println("WARNING: Input tensor shape does not match expected [1,32,32,1]");
    }
    if (output->dims->size != 2 || output->dims->data[1] != NUM_CLASSES) {
        Serial.println("WARNING: Output tensor shape does not match expected [1,NUM_CLASSES]");
    }

    isInitialized = true;

    Serial.println("InferenceEngine initialized successfully (TFLite Micro)");
    Serial.printf("Input shape: [1, %d, %d, %d]\n", MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_CHANNELS);
    Serial.printf("Output shape: [1, %d]\n", NUM_CLASSES);
    Serial.printf("Model size: %d bytes\n", getModelSize());
    return true;
}

bool InferenceEngine::runInference(const float* spectrogram, float* probabilities) {
    if (!isInitialized || !spectrogram || !probabilities) {
        return false;
    }

    // Copy spectrogram into model input
    if (input->type == kTfLiteFloat32) {
        float* in = input->data.f;
        const size_t count = (size_t)(MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH * MODEL_INPUT_CHANNELS);
        memcpy(in, spectrogram, count * sizeof(float));
    } else if (input->type == kTfLiteInt8) {
        int8_t* in = input->data.int8;
        const size_t count = (size_t)(MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH * MODEL_INPUT_CHANNELS);
        const float scale = input->params.scale;
        const int zero_point = input->params.zero_point;
        for (size_t i = 0; i < count; i++) {
            int q = (int)lroundf(spectrogram[i] / scale) + zero_point;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            in[i] = (int8_t)q;
        }
    } else {
        Serial.println("ERROR: Unsupported input tensor type");
        return false;
    }

    // Invoke
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("ERROR: Inference invocation failed");
        return false;
    }

    // Read output and apply softmax to convert logits to probabilities
    if (output->type == kTfLiteFloat32) {
        const float* out = output->data.f;
        // Apply softmax to convert logits to probabilities
        float max_logit = out[0];
        for (int i = 1; i < NUM_CLASSES; i++) {
            if (out[i] > max_logit) max_logit = out[i];
        }
        
        float sum_exp = 0.0f;
        for (int i = 0; i < NUM_CLASSES; i++) {
            probabilities[i] = expf(out[i] - max_logit); // Subtract max for numerical stability
            sum_exp += probabilities[i];
        }
        
        // Normalize to get probabilities
        for (int i = 0; i < NUM_CLASSES; i++) {
            probabilities[i] /= sum_exp;
        }
    } else if (output->type == kTfLiteInt8) {
        const int8_t* out = output->data.int8;
        const float scale = output->params.scale;
        const int zero_point = output->params.zero_point;
        
        // Convert quantized logits to float
        float logits[NUM_CLASSES];
        for (int i = 0; i < NUM_CLASSES; i++) {
            logits[i] = (out[i] - zero_point) * scale;
        }
        
        // Apply softmax
        float max_logit = logits[0];
        for (int i = 1; i < NUM_CLASSES; i++) {
            if (logits[i] > max_logit) max_logit = logits[i];
        }
        
        float sum_exp = 0.0f;
        for (int i = 0; i < NUM_CLASSES; i++) {
            probabilities[i] = expf(logits[i] - max_logit);
            sum_exp += probabilities[i];
        }
        
        // Normalize to get probabilities
        for (int i = 0; i < NUM_CLASSES; i++) {
            probabilities[i] /= sum_exp;
        }
    } else {
        Serial.println("ERROR: Unsupported output tensor type");
        return false;
    }

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

// mock generator removed

#endif // INFERENCE_ENGINE_H
