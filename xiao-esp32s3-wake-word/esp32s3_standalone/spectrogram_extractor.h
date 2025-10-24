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
    
    // Custom FFT implementation matching Python exactly
    void performFFT(float* real, float* imag, int n);
    
    // Bit-reversal permutation for FFT
    void bitReversePermutation(float* real, float* imag, int n);
    
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
    
    // Calculate number of frames exactly as in training config
    const int numFrames = min((NUM_SAMPLES - FRAME_LENGTH) / FRAME_STEP + 1, SPECTROGRAM_HEIGHT);
    
    // Clear output spectrogram
    memset(spectrogram, 0, SPECTROGRAM_BUFFER_SIZE * sizeof(float));
    
    // Use configured frame step (match training)
    const int frameStep = FRAME_STEP;
    
    for (int frame = 0; frame < numFrames; frame++) {
        int frameStart = frame * frameStep;
        
        // Skip if we don't have enough samples
        if (frameStart + FRAME_LENGTH > NUM_SAMPLES) break;
        
        // Extract frame and apply Hann window
        for (int i = 0; i < FRAME_LENGTH; i++) {
            windowBuffer[i] = audio[frameStart + i] / 32768.0f; // Normalize to [-1, 1]
        }
        applyHannWindow(windowBuffer, FRAME_LENGTH);
        
        // Pad frame to FFT length if needed
        float* paddedFrame = windowBuffer;
        if (FRAME_LENGTH < FFT_LENGTH) {
            // Pad with zeros
            for (int i = FRAME_LENGTH; i < FFT_LENGTH; i++) {
                windowBuffer[i] = 0.0f;
            }
        } else if (FRAME_LENGTH > FFT_LENGTH) {
            // Truncate to FFT length
            paddedFrame = windowBuffer;
        }
        
        // Compute real FFT magnitude spectrum
        computeFFTMagnitude(paddedFrame, magnitudeBuffer, FFT_LENGTH);
        
        // Store magnitude spectrum in output spectrogram
        // Only use first half of FFT result (positive frequencies)
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
    // Real FFT implementation matching Python custom_fft exactly
    // Use complex FFT workspace
    float* real = fftBuffer;
    float* imag = fftBuffer + fftLength;
    
    // Initialize with frame data
    for (int i = 0; i < fftLength; i++) {
        real[i] = frame[i];
        imag[i] = 0.0f;
    }
    
    // Perform FFT
    performFFT(real, imag, fftLength);
    
    // Compute magnitude spectrum
    int numBins = fftLength / 2;
    for (int k = 0; k < numBins; k++) {
        magnitude[k] = sqrtf(real[k] * real[k] + imag[k] * imag[k]);
    }
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

void SpectrogramExtractor::performFFT(float* real, float* imag, int n) {
    // Radix-2 FFT implementation matching Python custom_fft exactly
    
    // Bit-reversal permutation
    bitReversePermutation(real, imag, n);
    
    // FFT computation (Cooley-Tukey)
    for (int length = 2; length <= n; length <<= 1) {
        float angle = -2.0f * PI / length;
        float wlen_real = cosf(angle);
        float wlen_imag = sinf(angle);
        
        for (int i = 0; i < n; i += length) {
            float w_real = 1.0f;
            float w_imag = 0.0f;
            
            for (int j = 0; j < length / 2; j++) {
                int u = i + j;
                int v = i + j + length / 2;
                
                float t_real = w_real * real[v] - w_imag * imag[v];
                float t_imag = w_real * imag[v] + w_imag * real[v];
                
                real[v] = real[u] - t_real;
                imag[v] = imag[u] - t_imag;
                real[u] += t_real;
                imag[u] += t_imag;
                
                float next_w_real = w_real * wlen_real - w_imag * wlen_imag;
                float next_w_imag = w_real * wlen_imag + w_imag * wlen_real;
                w_real = next_w_real;
                w_imag = next_w_imag;
            }
        }
    }
}

void SpectrogramExtractor::bitReversePermutation(float* real, float* imag, int n) {
    // Bit-reversal permutation matching Python implementation
    for (int i = 0; i < n; i++) {
        int j = 0;
        int temp = i;
        int log2n = 0;
        int temp_n = n;
        while (temp_n >>= 1) log2n++;
        
        for (int k = 0; k < log2n; k++) {
            j = (j << 1) | (temp & 1);
            temp >>= 1;
        }
        
        if (i < j) {
            // Swap real parts
            float temp_real = real[i];
            real[i] = real[j];
            real[j] = temp_real;
            
            // Swap imaginary parts
            float temp_imag = imag[i];
            imag[i] = imag[j];
            imag[j] = temp_imag;
        }
    }
}

#endif // SPECTROGRAM_EXTRACTOR_H
