/*
 * MFCC Processor for ESP32-S3 Wake Word Detection
 *
 * This header file implements MFCC (Mel-Frequency Cepstral Coefficients)
 * feature extraction optimized for microcontrollers.
 *
 * Based on the TensorFlow Lite Micro signal processing library.
 */

#ifndef MFCC_PROCESSOR_H
#define MFCC_PROCESSOR_H

#include <Arduino.h>
#include <cmath>

// MFCC processing constants
#define SAMPLE_RATE 16000
#define WINDOW_SIZE_MS 30
#define WINDOW_STRIDE_MS 10
#define FEATURE_BINS 40
#define NUM_MFCC_COEFFS 10  // Number of MFCC coefficients to compute

// Derived constants
#define WINDOW_SIZE_SAMPLES (WINDOW_SIZE_MS * SAMPLE_RATE / 1000)
#define WINDOW_STRIDE_SAMPLES (WINDOW_STRIDE_MS * SAMPLE_RATE / 1000)
#define FFT_SIZE WINDOW_SIZE_SAMPLES
#define MEL_BINS 40

// Hann window for FFT
#define HANNING_WINDOW(size) \
    for (int i = 0; i < size; i++) { \
        window[i] = 0.5f * (1.0f - cosf(2.0f * PI * i / (size - 1))); \
    }

class MFCCProcessor {
private:
    // FFT related
    float* fft_input;
    float* fft_output;
    float* window;

    // Mel filterbank
    float* mel_filterbank;
    float* mel_freqs;

    // DCT matrix for MFCC
    float* dct_matrix;

    // Working buffers
    float* power_spectrum;
    float* mel_energies;
    float* mfcc_coeffs;

    // Pre-computed constants
    float pre_emphasis_alpha;
    int fft_size;
    int mel_bins;
    int mfcc_coeffs_num;

public:
    MFCCProcessor();
    ~MFCCProcessor();

    // Initialization
    bool initialize();

    // Feature extraction
    bool extractFeatures(int16_t* audio_data, int data_size, float* features, int max_frames);

    // Individual processing steps
    void preEmphasis(int16_t* input, float* output, int size);
    void applyWindow(float* input, float* output, int size);
    void computeFFT(float* input, float* output, int size);
    void computePowerSpectrum(float* fft_output, float* power_spectrum, int size);
    void applyMelFilterbank(float* power_spectrum, float* mel_energies, int fft_size);
    void applyDCT(float* mel_energies, float* mfcc_coeffs, int mel_bins, int mfcc_num);

private:
    // Helper functions
    void createMelFilterbank();
    void createDCTMatrix();
    float hzToMel(float hz);
    float melToHz(float mel);
    void normalizeFeatures(float* features, int num_features);

    // FFT implementation (simplified for microcontroller)
    void fft(float* input, float* output, int size);
    void ifft(float* input, float* output, int size);
};

// Simplified FFT implementation for microcontrollers
class SimpleFFT {
private:
    int n;
    int* bit_reverse_table;
    float* cos_table;
    float* sin_table;

public:
    SimpleFFT(int size);
    ~SimpleFFT();

    void forward(float* input, float* output);
    void inverse(float* input, float* output);

private:
    void bitReverse(float* data);
    void fftIteration(float* data, int size);
    int reverseBits(int x, int bits);
};

// Feature extraction result
struct AudioFeatures {
    float* features;      // MFCC features
    int num_frames;       // Number of feature frames
    int feature_dim;      // Feature dimension (MFCC bins)
    int total_features;   // Total number of features

    AudioFeatures() : features(nullptr), num_frames(0), feature_dim(0), total_features(0) {}
    ~AudioFeatures() {
        if (features) {
            delete[] features;
        }
    }

    bool allocate(int frames, int dim) {
        num_frames = frames;
        feature_dim = dim;
        total_features = frames * dim;

        features = new float[total_features];
        return features != nullptr;
    }
};

#endif // MFCC_PROCESSOR_H
