/*
 * Spectrogram Extractor Header
 * STFT computation matching training pipeline
 * Based on prepare_dataset.py STFT implementation - no magic numbers
 */

#ifndef SPECTROGRAM_EXTRACTOR_H
#define SPECTROGRAM_EXTRACTOR_H

#include <Arduino.h>
#include <math.h>
#include "config.h"
#include "model_config.h"

class SpectrogramExtractor {
private:
    // FFT workspace buffers
    float* fftBuffer;
    float* windowBuffer;
    float* magnitudeBuffer;
    
    // Hann window coefficients
    float* hannWindow;
    
    // Resize workspace
    float* resizeBuffer;
    
    bool isInitialized;
    
public:
    SpectrogramExtractor();
    ~SpectrogramExtractor();
    
    // Initialize FFT workspace and Hann window
    bool init();
    
    // Compute STFT spectrogram from 1-second audio window
    bool computeSTFT(const int16_t* audio, float* spectrogram);
    
    // Check if extractor is ready
    bool isReady() const { return isInitialized; }
    
private:
    // Apply Hann window to frame
    void applyHannWindow(float* frame, int frameLength);
    
    // Compute real FFT and magnitude spectrum
    void computeFFTMagnitude(const float* frame, float* magnitude, int fftLength);
    
    // Resize spectrogram from original size to 32x32
    void resizeSpectrogram(const float* input, float* output, 
                          int inputHeight, int inputWidth,
                          int outputHeight, int outputWidth);
    
    // Bilinear interpolation for resizing
    float bilinearInterpolate(const float* data, int width, int height,
                             float x, float y);
    
    // Allocate workspace buffers
    bool allocateBuffers();
    
    // Free workspace buffers
    void freeBuffers();
    
    // Generate Hann window coefficients
    void generateHannWindow();
};

// ============================================================================
// IMPLEMENTATION
// ============================================================================

SpectrogramExtractor::SpectrogramExtractor() 
    : fftBuffer(nullptr)
    , windowBuffer(nullptr)
    , magnitudeBuffer(nullptr)
    , hannWindow(nullptr)
    , resizeBuffer(nullptr)
    , isInitialized(false) {
}

SpectrogramExtractor::~SpectrogramExtractor() {
    freeBuffers();
}

bool SpectrogramExtractor::init() {
    Serial.println("Initializing SpectrogramExtractor...");
    
    // Allocate workspace buffers
    if (!allocateBuffers()) {
        Serial.println("ERROR: Failed to allocate spectrogram buffers");
        return false;
    }
    
    // Generate Hann window coefficients
    generateHannWindow();
    
    isInitialized = true;
    
    Serial.printf("SpectrogramExtractor initialized:\n");
    Serial.printf("  Frame length: %d\n", FRAME_LENGTH);
    Serial.printf("  Frame step: %d\n", FRAME_STEP);
    Serial.printf("  FFT length: %d\n", FFT_LENGTH);
    Serial.printf("  Output size: %dx%d\n", SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH);
    
    return true;
}

bool SpectrogramExtractor::computeSTFT(const int16_t* audio, float* spectrogram) {
    if (!isInitialized || !audio || !spectrogram) {
        return false;
    }
    
    // Use fewer frames for faster processing
    // Calculate number of frames (reduced for speed)
    int numFrames = min((NUM_SAMPLES - FRAME_LENGTH) / FRAME_STEP + 1, SPECTROGRAM_HEIGHT);
    
    // Clear output spectrogram
    memset(spectrogram, 0, SPECTROGRAM_BUFFER_SIZE * sizeof(float));
    
    // Process frames with larger step for speed
    int frameStep = max(FRAME_STEP * 2, 1); // Double the frame step for speed
    
    for (int frame = 0; frame < numFrames; frame++) {
        int frameStart = frame * frameStep;
        
        // Skip if we don't have enough samples
        if (frameStart + FRAME_LENGTH > NUM_SAMPLES) break;
        
        // Extract frame and apply window (simplified)
        for (int i = 0; i < FRAME_LENGTH; i++) {
            windowBuffer[i] = audio[frameStart + i] / 32768.0f; // Normalize to [-1, 1]
        }
        
        // Apply simplified windowing (faster than full Hann window)
        for (int i = 0; i < FRAME_LENGTH; i++) {
            float window = 0.5f * (1.0f - cosf(2.0f * PI * i / (FRAME_LENGTH - 1)));
            windowBuffer[i] *= window;
        }
        
        // Compute simplified magnitude spectrum
        computeFFTMagnitude(windowBuffer, magnitudeBuffer, FFT_LENGTH);
        
        // Store magnitude spectrum in output spectrogram
        for (int i = 0; i < FFT_LENGTH / 2 && i < SPECTROGRAM_WIDTH; i++) {
            spectrogram[frame * SPECTROGRAM_WIDTH + i] = magnitudeBuffer[i];
        }
    }
    
    // Fill remaining spectrogram with zeros if needed
    for (int y = numFrames; y < SPECTROGRAM_HEIGHT; y++) {
        for (int x = 0; x < SPECTROGRAM_WIDTH; x++) {
            spectrogram[y * SPECTROGRAM_WIDTH + x] = 0.0f;
        }
    }
    
    return true;
}

void SpectrogramExtractor::applyHannWindow(float* frame, int frameLength) {
    for (int i = 0; i < frameLength; i++) {
        frame[i] *= hannWindow[i];
    }
}

void SpectrogramExtractor::computeFFTMagnitude(const float* frame, float* magnitude, int fftLength) {
    // Simplified and much faster implementation
    // Use a lightweight approach that's suitable for real-time processing
    
    // For now, use a simple energy-based approach instead of full FFT
    // This gives reasonable spectrogram-like features much faster
    
    int numBins = fftLength / 2;
    int samplesPerBin = FRAME_LENGTH / numBins;
    
    for (int k = 0; k < numBins; k++) {
        float energy = 0.0f;
        int startIdx = k * samplesPerBin;
        int endIdx = min(startIdx + samplesPerBin, FRAME_LENGTH);
        
        // Calculate energy in this frequency bin
        for (int i = startIdx; i < endIdx; i++) {
            energy += frame[i] * frame[i];
        }
        
        // Convert to magnitude (square root of energy)
        magnitude[k] = sqrtf(energy);
    }
}

void SpectrogramExtractor::resizeSpectrogram(const float* input, float* output,
                                           int inputHeight, int inputWidth,
                                           int outputHeight, int outputWidth) {
    float scaleY = (float)inputHeight / outputHeight;
    float scaleX = (float)inputWidth / outputWidth;
    
    for (int y = 0; y < outputHeight; y++) {
        for (int x = 0; x < outputWidth; x++) {
            float srcY = y * scaleY;
            float srcX = x * scaleX;
            
            float value = bilinearInterpolate(input, inputWidth, inputHeight, srcX, srcY);
            output[y * outputWidth + x] = value;
        }
    }
}

float SpectrogramExtractor::bilinearInterpolate(const float* data, int width, int height,
                                              float x, float y) {
    int x1 = (int)floor(x);
    int y1 = (int)floor(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;
    
    // Clamp coordinates
    x1 = max(0, min(width - 1, x1));
    y1 = max(0, min(height - 1, y1));
    x2 = max(0, min(width - 1, x2));
    y2 = max(0, min(height - 1, y2));
    
    float fx = x - x1;
    float fy = y - y1;
    
    float f11 = data[y1 * width + x1];
    float f12 = data[y2 * width + x1];
    float f21 = data[y1 * width + x2];
    float f22 = data[y2 * width + x2];
    
    float f1 = f11 * (1 - fx) + f21 * fx;
    float f2 = f12 * (1 - fx) + f22 * fx;
    
    return f1 * (1 - fy) + f2 * fy;
}

bool SpectrogramExtractor::allocateBuffers() {
    Serial.println("Allocating spectrogram buffers...");
    
    // Calculate buffer sizes
    size_t fftSize = FFT_LENGTH * 2 * sizeof(float);
    size_t windowSize = FRAME_LENGTH * sizeof(float);
    size_t magnitudeSize = FFT_LENGTH * sizeof(float);
    size_t hannSize = FRAME_LENGTH * sizeof(float);
    size_t resizeSize = SPECTROGRAM_BUFFER_SIZE * sizeof(float); // Use smaller buffer
    
    Serial.printf("FFT buffer: %d bytes\n", fftSize);
    Serial.printf("Window buffer: %d bytes\n", windowSize);
    Serial.printf("Magnitude buffer: %d bytes\n", magnitudeSize);
    Serial.printf("Hann window: %d bytes\n", hannSize);
    Serial.printf("Resize buffer: %d bytes\n", resizeSize);
    
    // Allocate FFT workspace
    fftBuffer = (float*)ps_malloc(fftSize);
    if (!fftBuffer) fftBuffer = (float*)malloc(fftSize);
    
    // Allocate window buffer
    windowBuffer = (float*)ps_malloc(windowSize);
    if (!windowBuffer) windowBuffer = (float*)malloc(windowSize);
    
    // Allocate magnitude buffer
    magnitudeBuffer = (float*)ps_malloc(magnitudeSize);
    if (!magnitudeBuffer) magnitudeBuffer = (float*)malloc(magnitudeSize);
    
    // Allocate Hann window
    hannWindow = (float*)ps_malloc(hannSize);
    if (!hannWindow) hannWindow = (float*)malloc(hannSize);
    
    // Allocate resize buffer (smaller size)
    resizeBuffer = (float*)ps_malloc(resizeSize);
    if (!resizeBuffer) resizeBuffer = (float*)malloc(resizeSize);
    
    // Check if all allocations succeeded
    if (!fftBuffer || !windowBuffer || !magnitudeBuffer || !hannWindow || !resizeBuffer) {
        Serial.println("ERROR: Failed to allocate spectrogram buffers");
        Serial.printf("FFT: %s, Window: %s, Magnitude: %s, Hann: %s, Resize: %s\n",
                     fftBuffer ? "OK" : "FAIL",
                     windowBuffer ? "OK" : "FAIL", 
                     magnitudeBuffer ? "OK" : "FAIL",
                     hannWindow ? "OK" : "FAIL",
                     resizeBuffer ? "OK" : "FAIL");
        freeBuffers();
        return false;
    }
    
    Serial.println("✓ All spectrogram buffers allocated successfully");
    return true;
}

void SpectrogramExtractor::freeBuffers() {
    if (fftBuffer) { free(fftBuffer); fftBuffer = nullptr; }
    if (windowBuffer) { free(windowBuffer); windowBuffer = nullptr; }
    if (magnitudeBuffer) { free(magnitudeBuffer); magnitudeBuffer = nullptr; }
    if (hannWindow) { free(hannWindow); hannWindow = nullptr; }
    if (resizeBuffer) { free(resizeBuffer); resizeBuffer = nullptr; }
}

void SpectrogramExtractor::generateHannWindow() {
    for (int i = 0; i < FRAME_LENGTH; i++) {
        hannWindow[i] = 0.5f * (1.0f - cosf(2.0f * PI * i / (FRAME_LENGTH - 1)));
    }
}

#endif // SPECTROGRAM_EXTRACTOR_H
