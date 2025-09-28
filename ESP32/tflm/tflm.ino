#include "model_data.h"        // Model file, converted from .tflite with xxd -i
#include <tflm_esp32.h>        // ESP32-specific TensorFlow Lite Micro runtime
#include <eloquent_tinyml.h>   // TinyML wrapper
#include <I2S.h>              // Microphone input
#include <arduinoFFT.h>       // FFT for MFCC computation
#include <math.h>

// =========================
//     MODEL PARAMETERS
// =========================
#define NUM_MFCC        13
#define NUM_LABELS      4
#define TF_NUM_OPS      2
#define ARENA_SIZE      12000

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;
const char* labels[] = {"shalom", "unknown", "noise", "lehit"};
float mfcc_features[NUM_MFCC];

// =========================
//     AUDIO PARAMETERS
// =========================
#define SAMPLE_RATE        16000
#define SAMPLE_BITS        16
#define AUDIO_BUFFER_SIZE  512    // Must be power of 2 for FFT

int16_t audioBuffer[AUDIO_BUFFER_SIZE];
float vReal[AUDIO_BUFFER_SIZE];
float vImag[AUDIO_BUFFER_SIZE];
ArduinoFFT<float> FFT = ArduinoFFT<float>();

// =========================
//   MICROPHONE SETUP
// =========================
void setupMicrophone() {
    I2S.setAllPins(-1, 42, 41, -1, -1);

    if (!I2S.begin(PDM_MONO_MODE, SAMPLE_RATE, SAMPLE_BITS)) {
        Serial.println("Failed to initialize I2S!");
        while (true) delay(1);
    }
    Serial.println("I2S microphone initialized.");
}

// =========================
//     MFCC EXTRACTION
// =========================
#define NUM_MEL_FILTERS 26
#define PRE_EMPHASIS    0.97f

float preEmphasized[AUDIO_BUFFER_SIZE];
float hammingWindow[AUDIO_BUFFER_SIZE];
float magnitudeSpectrum[AUDIO_BUFFER_SIZE / 2 + 1];
float melEnergies[NUM_MEL_FILTERS];
float melFilterBank[NUM_MEL_FILTERS][AUDIO_BUFFER_SIZE / 2 + 1];
bool melInitialized = false;

float hzToMel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float melToHz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

void createHammingWindow() {
    for (int i = 0; i < AUDIO_BUFFER_SIZE; i++) {
        hammingWindow[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (AUDIO_BUFFER_SIZE - 1));
    }
}

void createMelFilterbank() {
    float lowMel = hzToMel(0);
    float highMel = hzToMel(SAMPLE_RATE / 2);
    float melPoints[NUM_MEL_FILTERS + 2];

    for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
        melPoints[i] = lowMel + (highMel - lowMel) * i / (NUM_MEL_FILTERS + 1);
    }

    float hzPoints[NUM_MEL_FILTERS + 2];
    for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
        hzPoints[i] = melToHz(melPoints[i]);
    }

    int fftSize = AUDIO_BUFFER_SIZE;
    for (int m = 1; m <= NUM_MEL_FILTERS; m++) {
        float f_m_minus = hzPoints[m - 1];
        float f_m = hzPoints[m];
        float f_m_plus = hzPoints[m + 1];

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

    melInitialized = true;
}

void captureAndComputeMFCC(float* mfcc_features) {
    if (!melInitialized) {
        createHammingWindow();
        createMelFilterbank();
    }

    int bytesRead = I2S.read(audioBuffer, AUDIO_BUFFER_SIZE * sizeof(int16_t));
    if (bytesRead <= 0) {
        Serial.println("I2S read failed");
        for (int i = 0; i < NUM_MFCC; i++) mfcc_features[i] = 0.0f;
        return;
    }

    // Pre-emphasis
    preEmphasized[0] = (float)audioBuffer[0] / 32768.0f;
    for (int i = 1; i < AUDIO_BUFFER_SIZE; i++) {
        float x = (float)audioBuffer[i] / 32768.0f;
        preEmphasized[i] = x - PRE_EMPHASIS * ((float)audioBuffer[i - 1] / 32768.0f);
    }

    // Window + FFT
    for (int i = 0; i < AUDIO_BUFFER_SIZE; i++) {
        vReal[i] = preEmphasized[i] * hammingWindow[i];
        vImag[i] = 0.0f;
    }

    FFT.compute(vReal, vImag, AUDIO_BUFFER_SIZE, FFT_FORWARD);
    FFT.complexToMagnitude(vReal, vImag, AUDIO_BUFFER_SIZE);

    for (int i = 0; i <= AUDIO_BUFFER_SIZE / 2; i++) {
        magnitudeSpectrum[i] = vReal[i];
    }

    // Mel filtering
    for (int m = 0; m < NUM_MEL_FILTERS; m++) {
        float sum = 0.0f;
        for (int k = 0; k <= AUDIO_BUFFER_SIZE / 2; k++) {
            sum += magnitudeSpectrum[k] * melFilterBank[m][k];
        }
        melEnergies[m] = logf(sum + 1e-6f);
    }

    // DCT to MFCC
    for (int n = 0; n < NUM_MFCC; n++) {
        float sum = 0.0f;
        for (int m = 0; m < NUM_MEL_FILTERS; m++) {
            sum += melEnergies[m] * cosf(M_PI * n * (m + 0.5f) / NUM_MEL_FILTERS);
        }
        mfcc_features[n] = sum;
    }
}

// =========================
//         SETUP
// =========================
void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    Serial.begin(115200);
    setupMicrophone();

    tf.setNumInputs(NUM_MFCC);
    tf.setNumOutputs(NUM_LABELS);
    tf.resolver.AddFullyConnected();
    tf.resolver.AddSoftmax();

    while (!tf.begin(wake_word_model_tflite).isOk()) {
        Serial.println(tf.exception.toString());
    }
}

// =========================
//         LOOP
// =========================
void loop() {
    captureAndComputeMFCC(mfcc_features);

    if (!tf.predict(mfcc_features).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }

    for (int i = 0; i < NUM_LABELS; i++) {
        Serial.print(labels[i]);
        Serial.print(": ");
        Serial.print(tf.output(i));
        Serial.print(" | ");
    }
    Serial.println();

    delay(1000);
}
