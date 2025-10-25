#include <Arduino.h>
#include <I2S.h>
#include <arduinoFFT.h>
#include <math.h>
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// Include your float TFLite model converted as a header
#include "model_data.h"

// ================= CONFIGURATION =================
#define SAMPLE_RATE       16000
#define AUDIO_BUFFER_SIZE 512         // Must be power of 2 for FFT
#define NUM_MFCC          13
#define NUM_MEL_FILTERS   26
#define NUM_LABELS        4
#define TF_NUM_OPS        2
#define ARENA_SIZE        12000

// Labels for classification
const char* labels[NUM_LABELS] = {"shalom", "unknown", "noise", "lehit"};

// ================ GLOBAL VARIABLES ================
int16_t audioBuffer[AUDIO_BUFFER_SIZE];
float vReal[AUDIO_BUFFER_SIZE];
float vImag[AUDIO_BUFFER_SIZE];
float magnitudeSpectrum[AUDIO_BUFFER_SIZE / 2 + 1];

ArduinoFFT<float> FFT = ArduinoFFT<float>();

float hammingWindow[AUDIO_BUFFER_SIZE];

// Mel filterbank storage
float melFilterBank[NUM_MEL_FILTERS][AUDIO_BUFFER_SIZE / 2 + 1];

// Buffer for mel energies and MFCCs
float melEnergies[NUM_MEL_FILTERS];
float mfccFeatures[NUM_MFCC];

// TensorFlow Lite Micro model object
Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

// ================ AUDIO / MFCC UTILS ================

// Convert frequency in Hz to Mel scale
float hzToMel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

// Convert Mel scale back to Hz
float melToHz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Create a Hamming window
void createHammingWindow() {
    for (int i = 0; i < AUDIO_BUFFER_SIZE; i++) {
        hammingWindow[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (AUDIO_BUFFER_SIZE - 1));
    }
}

// Create Mel filterbank weights
void createMelFilterbank() {
    float lowMel = hzToMel(0);
    float highMel = hzToMel(SAMPLE_RATE / 2);
    float melPoints[NUM_MEL_FILTERS + 2];

    for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
        melPoints[i] = lowMel + (highMel - lowMel) * i / (NUM_MEL_FILTERS + 1);
    }

    float freqPoints[NUM_MEL_FILTERS + 2];
    for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
        freqPoints[i] = melToHz(melPoints[i]);
    }

    int fftSize = AUDIO_BUFFER_SIZE;
    for (int m = 1; m <= NUM_MEL_FILTERS; m++) {
        float f_m_minus = freqPoints[m - 1];
        float f_m = freqPoints[m];
        float f_m_plus = freqPoints[m + 1];

        for (int k = 0; k <= fftSize / 2; k++) {
            float freq = (SAMPLE_RATE / 2.0f) * k / (fftSize / 2);
            float weight = 0.0f;
            if (freq >= f_m_minus && freq <= f_m) {
                weight = (freq - f_m_minus) / (f_m - f_m_minus);
            } else if (freq > f_m && freq <= f_m_plus) {
                weight = (f_m_plus - freq) / (f_m_plus - f_m);
            }
            melFilterBank[m - 1][k] = weight;
        }
    }
}

// Compute Discrete Cosine Transform (DCT) on Mel energies to get MFCC
void computeDCT(float* input, float* output, int inputSize, int outputSize) {
    for (int n = 0; n < outputSize; n++) {
        float sum = 0.0f;
        for (int m = 0; m < inputSize; m++) {
            sum += input[m] * cosf(M_PI * n * (m + 0.5f) / inputSize);
        }
        output[n] = sum;
    }
}

// Capture audio, compute MFCC features
bool captureAndComputeMFCC(float* mfccFeaturesOut) {
    int bytesRead = I2S.read(audioBuffer, AUDIO_BUFFER_SIZE * sizeof(int16_t));
    if (bytesRead <= 0) return false;

    // Convert audio samples to float and apply Hamming window
    for (int i = 0; i < AUDIO_BUFFER_SIZE; i++) {
        vReal[i] = (float)audioBuffer[i] / 32768.0f * hammingWindow[i];
        vImag[i] = 0.0f;
    }

    // Compute FFT
    FFT.compute(vReal, vImag, AUDIO_BUFFER_SIZE, FFT_FORWARD);
    FFT.complexToMagnitude(vReal, vImag, AUDIO_BUFFER_SIZE);

    // Copy magnitude spectrum
    for (int i = 0; i <= AUDIO_BUFFER_SIZE / 2; i++) {
        magnitudeSpectrum[i] = vReal[i];
    }

    // Calculate Mel energies (apply Mel filterbank)
    for (int m = 0; m < NUM_MEL_FILTERS; m++) {
        float melSum = 0.0f;
        for (int k = 0; k <= AUDIO_BUFFER_SIZE / 2; k++) {
            melSum += magnitudeSpectrum[k] * melFilterBank[m][k];
        }
        melEnergies[m] = logf(melSum + 1e-6f); // log energy to match MFCC norms
    }

    // Compute DCT to obtain MFCCs
    computeDCT(melEnergies, mfccFeaturesOut, NUM_MEL_FILTERS, NUM_MFCC);

    return true;
}

// ================= SETUP & LOOP =================

void setup() {
    Serial.begin(115200);

    // Initialize microphone via I2S: adjust pins as needed for your hardware
    I2S.setAllPins(-1, 42, 41, -1, -1);
    if (!I2S.begin(PDM_MONO_MODE, SAMPLE_RATE, 16)) {
        Serial.println("Failed to initialize I2S");
        while (true) delay(1);
    }
    Serial.println("I2S microphone initialized.");

    // Prepare Hamming window & Mel filterbank
    createHammingWindow();
    createMelFilterbank();

    // Initialize TensorFlow Lite Micro model
    tf.setNumInputs(NUM_MFCC);
    tf.setNumOutputs(NUM_LABELS);
    tf.resolver.AddFullyConnected();
    tf.resolver.AddSoftmax();
    while(!tf.begin(TFLM_wakeword_model_esp32mfcc_tflite).isOk()) {
        Serial.println(tf.exception.toString());
    }
}

void loop() {
    bool success = captureAndComputeMFCC(mfccFeatures);
    if (success) {
        if (tf.predict(mfccFeatures).isOk()) {
            // Print results
            for (int i = 0; i < NUM_LABELS; i++) {
                Serial.print(labels[i]);
                Serial.print(": ");
                Serial.print(tf.output(i), 5);
                Serial.print(" | ");
            }
            Serial.println();
        } else {
            Serial.println("Prediction failed");
        }
    } else {
        Serial.println("Audio capture failed");
    }

    delay(1000);
}
