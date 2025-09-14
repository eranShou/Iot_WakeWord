/*
 * MFCC Processor Implementation for ESP32-S3
 *
 * This file implements the MFCC feature extraction for wake word detection
 * on resource-constrained microcontrollers.
 */

#include "mfcc_processor.h"

// Constructor
MFCCProcessor::MFCCProcessor()
    : fft_input(nullptr), fft_output(nullptr), window(nullptr),
      mel_filterbank(nullptr), mel_freqs(nullptr), dct_matrix(nullptr),
      power_spectrum(nullptr), mel_energies(nullptr), mfcc_coeffs(nullptr),
      pre_emphasis_alpha(0.97f), fft_size(FFT_SIZE), mel_bins(MEL_BINS),
      mfcc_coeffs_num(NUM_MFCC_COEFFS) {
}

// Destructor
MFCCProcessor::~MFCCProcessor() {
    if (fft_input) delete[] fft_input;
    if (fft_output) delete[] fft_output;
    if (window) delete[] window;
    if (mel_filterbank) delete[] mel_filterbank;
    if (mel_freqs) delete[] mel_freqs;
    if (dct_matrix) delete[] dct_matrix;
    if (power_spectrum) delete[] power_spectrum;
    if (mel_energies) delete[] mel_energies;
    if (mfcc_coeffs) delete[] mfcc_coeffs;
}

// Initialize the MFCC processor
bool MFCCProcessor::initialize() {
    // Allocate memory for FFT
    fft_input = new float[fft_size];
    fft_output = new float[2 * fft_size];  // Complex output (real, imag)
    window = new float[fft_size];

    if (!fft_input || !fft_output || !window) {
        Serial.println("ERROR: Failed to allocate FFT buffers");
        return false;
    }

    // Create Hann window
    HANNING_WINDOW(fft_size);

    // Allocate memory for mel filterbank
    mel_filterbank = new float[mel_bins * (fft_size / 2 + 1)];
    mel_freqs = new float[mel_bins + 2];

    if (!mel_filterbank || !mel_freqs) {
        Serial.println("ERROR: Failed to allocate mel filterbank");
        return false;
    }

    // Create mel filterbank
    createMelFilterbank();

    // Allocate memory for DCT
    dct_matrix = new float[mfcc_coeffs_num * mel_bins];

    if (!dct_matrix) {
        Serial.println("ERROR: Failed to allocate DCT matrix");
        return false;
    }

    // Create DCT matrix
    createDCTMatrix();

    // Allocate working buffers
    power_spectrum = new float[fft_size / 2 + 1];
    mel_energies = new float[mel_bins];
    mfcc_coeffs = new float[mfcc_coeffs_num];

    if (!power_spectrum || !mel_energies || !mfcc_coeffs) {
        Serial.println("ERROR: Failed to allocate working buffers");
        return false;
    }

    Serial.println("MFCC processor initialized successfully");
    return true;
}

// Main feature extraction function
bool MFCCProcessor::extractFeatures(int16_t* audio_data, int data_size, float* features, int max_frames) {
    int feature_index = 0;
    int frame_count = 0;

    // Process audio in overlapping windows
    for (int start = 0; start + WINDOW_SIZE_SAMPLES <= data_size && frame_count < max_frames;
         start += WINDOW_STRIDE_SAMPLES) {

        // Extract window of audio
        float windowed_audio[WINDOW_SIZE_SAMPLES];
        preEmphasis(&audio_data[start], windowed_audio, WINDOW_SIZE_SAMPLES);
        applyWindow(windowed_audio, windowed_audio, WINDOW_SIZE_SAMPLES);

        // Copy to FFT input (zero-pad if necessary)
        memset(fft_input, 0, fft_size * sizeof(float));
        memcpy(fft_input, windowed_audio, WINDOW_SIZE_SAMPLES * sizeof(float));

        // Compute FFT
        computeFFT(fft_input, fft_output, fft_size);

        // Compute power spectrum
        computePowerSpectrum(fft_output, power_spectrum, fft_size);

        // Apply mel filterbank
        applyMelFilterbank(power_spectrum, mel_energies, fft_size);

        // Apply DCT to get MFCCs
        applyDCT(mel_energies, mfcc_coeffs, mel_bins, mfcc_coeffs_num);

        // Copy MFCCs to feature buffer
        memcpy(&features[feature_index], mfcc_coeffs, mfcc_coeffs_num * sizeof(float));
        feature_index += mfcc_coeffs_num;

        frame_count++;
    }

    return true;
}

// Pre-emphasis filter
void MFCCProcessor::preEmphasis(int16_t* input, float* output, int size) {
    // Convert to float and apply pre-emphasis
    output[0] = input[0] / 32768.0f;  // Normalize to [-1, 1]

    for (int i = 1; i < size; i++) {
        float current = input[i] / 32768.0f;
        output[i] = current - pre_emphasis_alpha * (input[i-1] / 32768.0f);
    }
}

// Apply Hann window
void MFCCProcessor::applyWindow(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * window[i];
    }
}

// Compute FFT (simplified implementation)
void MFCCProcessor::computeFFT(float* input, float* output, int size) {
    // Copy input to output as complex (imaginary part = 0)
    for (int i = 0; i < size; i++) {
        output[2 * i] = input[i];     // Real
        output[2 * i + 1] = 0.0f;    // Imaginary
    }

    // Simple FFT implementation (Cooley-Tukey)
    // This is a basic implementation - for production use a more optimized FFT
    fft(input, output, size);
}

// Compute power spectrum
void MFCCProcessor::computePowerSpectrum(float* fft_output, float* power_spectrum, int fft_size) {
    int spectrum_size = fft_size / 2 + 1;

    for (int i = 0; i < spectrum_size; i++) {
        float real = fft_output[2 * i];
        float imag = fft_output[2 * i + 1];
        power_spectrum[i] = real * real + imag * imag;
    }
}

// Apply mel filterbank
void MFCCProcessor::applyMelFilterbank(float* power_spectrum, float* mel_energies, int fft_size) {
    int spectrum_size = fft_size / 2 + 1;

    for (int mel = 0; mel < mel_bins; mel++) {
        float energy = 0.0f;

        for (int freq = 0; freq < spectrum_size; freq++) {
            energy += power_spectrum[freq] * mel_filterbank[mel * spectrum_size + freq];
        }

        mel_energies[mel] = energy;
    }

    // Apply log compression
    for (int i = 0; i < mel_bins; i++) {
        if (mel_energies[i] > 0) {
            mel_energies[i] = logf(mel_energies[i]);
        } else {
            mel_energies[i] = -10.0f;  // Small negative value for zero energy
        }
    }
}

// Apply Discrete Cosine Transform
void MFCCProcessor::applyDCT(float* mel_energies, float* mfcc_coeffs, int mel_bins, int mfcc_num) {
    for (int k = 0; k < mfcc_num; k++) {
        float sum = 0.0f;

        for (int n = 0; n < mel_bins; n++) {
            sum += mel_energies[n] * dct_matrix[k * mel_bins + n];
        }

        mfcc_coeffs[k] = sum;
    }
}

// Create mel filterbank
void MFCCProcessor::createMelFilterbank() {
    int spectrum_size = fft_size / 2 + 1;
    float f_min = 0.0f;
    float f_max = SAMPLE_RATE / 2.0f;

    // Create mel frequency points
    float mel_min = hzToMel(f_min);
    float mel_max = hzToMel(f_max);

    for (int i = 0; i < mel_bins + 2; i++) {
        float mel = mel_min + (mel_max - mel_min) * i / (mel_bins + 1);
        mel_freqs[i] = melToHz(mel);
    }

    // Create filterbank
    for (int mel = 0; mel < mel_bins; mel++) {
        float f_left = mel_freqs[mel];
        float f_center = mel_freqs[mel + 1];
        float f_right = mel_freqs[mel + 2];

        for (int freq = 0; freq < spectrum_size; freq++) {
            float f = (float)freq * SAMPLE_RATE / fft_size;
            float weight = 0.0f;

            if (f >= f_left && f <= f_center) {
                weight = (f - f_left) / (f_center - f_left);
            } else if (f >= f_center && f <= f_right) {
                weight = (f_right - f) / (f_right - f_center);
            }

            mel_filterbank[mel * spectrum_size + freq] = weight;
        }
    }
}

// Create DCT matrix
void MFCCProcessor::createDCTMatrix() {
    float sqrt_2_over_n = sqrtf(2.0f / mel_bins);

    for (int k = 0; k < mfcc_coeffs_num; k++) {
        for (int n = 0; n < mel_bins; n++) {
            dct_matrix[k * mel_bins + n] = sqrt_2_over_n *
                cosf(PI * k * (2.0f * n + 1.0f) / (2.0f * mel_bins));
        }
    }
}

// Convert Hz to Mel
float MFCCProcessor::hzToMel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

// Convert Mel to Hz
float MFCCProcessor::melToHz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Simplified FFT implementation (basic Cooley-Tukey)
void MFCCProcessor::fft(float* input, float* output, int size) {
    // This is a very basic FFT implementation
    // For production use, consider using a more optimized library

    // Copy input to complex output
    for (int i = 0; i < size; i++) {
        output[2 * i] = input[i];
        output[2 * i + 1] = 0.0f;
    }

    // Simple DFT (very slow, but works for small sizes)
    for (int k = 0; k < size; k++) {
        float real = 0.0f;
        float imag = 0.0f;

        for (int n = 0; n < size; n++) {
            float angle = -2.0f * PI * k * n / size;
            real += input[n] * cosf(angle);
            imag += input[n] * sinf(angle);
        }

        output[2 * k] = real;
        output[2 * k + 1] = imag;
    }
}

// ===== SimpleFFT Implementation =====

SimpleFFT::SimpleFFT(int size) : n(size) {
    bit_reverse_table = new int[n];
    cos_table = new float[n/2];
    sin_table = new float[n/2];

    // Pre-compute bit reversal table
    for (int i = 0; i < n; i++) {
        bit_reverse_table[i] = reverseBits(i, (int)log2f(n));
    }

    // Pre-compute trigonometric tables
    for (int i = 0; i < n/2; i++) {
        float angle = -2.0f * PI * i / n;
        cos_table[i] = cosf(angle);
        sin_table[i] = sinf(angle);
    }
}

SimpleFFT::~SimpleFFT() {
    delete[] bit_reverse_table;
    delete[] cos_table;
    delete[] sin_table;
}

void SimpleFFT::forward(float* input, float* output) {
    // Copy input to output
    memcpy(output, input, 2 * n * sizeof(float));

    // Apply bit reversal
    bitReverse(output);

    // FFT iterations
    for (int size = 2; size <= n; size *= 2) {
        int half_size = size / 2;
        float angle_step = -2.0f * PI / size;

        for (int i = 0; i < n; i += size) {
            for (int j = i, k = 0; j < i + half_size; j++, k++) {
                float cos_val = cos_table[k * n / size];
                float sin_val = sin_table[k * n / size];

                float t_real = output[2 * (j + half_size)] * cos_val -
                              output[2 * (j + half_size) + 1] * sin_val;
                float t_imag = output[2 * (j + half_size)] * sin_val +
                              output[2 * (j + half_size) + 1] * cos_val;

                float u_real = output[2 * j];
                float u_imag = output[2 * j + 1];

                output[2 * j] = u_real + t_real;
                output[2 * j + 1] = u_imag + t_imag;
                output[2 * (j + half_size)] = u_real - t_real;
                output[2 * (j + half_size) + 1] = u_imag - t_imag;
            }
        }
    }
}

void SimpleFFT::bitReverse(float* data) {
    for (int i = 0; i < n; i++) {
        int j = bit_reverse_table[i];
        if (j > i) {
            // Swap
            float temp_real = data[2 * i];
            float temp_imag = data[2 * i + 1];
            data[2 * i] = data[2 * j];
            data[2 * j] = temp_real;
            data[2 * i + 1] = data[2 * j + 1];
            data[2 * j + 1] = temp_imag;
        }
    }
}

int SimpleFFT::reverseBits(int x, int bits) {
    int result = 0;
    for (int i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}
