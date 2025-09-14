/*
 * Hebrew Wake Word Detection for ESP32-S3
 *
 * This Arduino sketch implements real-time wake word detection for Hebrew words
 * "shalom" (hello) and "lehitraot" (goodbye) using TensorFlow Lite Micro.
 *
 * Features:
 * - Real-time audio capture from built-in microphone
 * - MFCC feature extraction
 * - TensorFlow Lite Micro inference
 * - LED and serial output for wake word detection
 * - Low-power optimizations
 * - Support for custom wake words
 *
 * Hardware: Seeed Studio XIAO ESP32-S3
 * Microphone: Built-in PDM microphone
 */

#include <I2S.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include model data
#include "hebrew_wake_word_model_data.h"

// Audio processing constants
#define SAMPLE_RATE 16000
#define AUDIO_LENGTH 1000  // 1 second of audio
#define BUFFER_SIZE (SAMPLE_RATE * AUDIO_LENGTH / 1000)
#define WINDOW_SIZE 480    // 30ms window at 16kHz
#define WINDOW_STRIDE 160  // 10ms stride at 16kHz
#define FEATURE_BINS 40    // MFCC bins

// Wake word detection constants
#define DETECTION_THRESHOLD 0.8f
#define COOLDOWN_MS 2000  // Minimum time between detections

// Hardware pins (XIAO ESP32-S3)
#define LED_PIN 21
#define MIC_DATA_PIN 23
#define MIC_CLOCK_PIN 22

// TensorFlow Lite globals
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Memory arena for TFLM
  constexpr int kTensorArenaSize = 80 * 1024;  // 80KB for model
  uint8_t tensor_arena[kTensorArenaSize];

  // Wake word labels
  const char* wake_words[] = {"silence", "unknown", "shalom", "lehitraot"};
  const int num_classes = sizeof(wake_words) / sizeof(wake_words[0]);
}

// Audio processing globals
int16_t audio_buffer[BUFFER_SIZE];
volatile int audio_index = 0;
volatile bool buffer_ready = false;

// Detection state
unsigned long last_detection_time = 0;
bool detection_active = false;

// Function declarations
bool setup_tensorflow();
void process_audio();
float* extract_mfcc_features(int16_t* audio_data, int data_size);
int detect_wake_word(float* features);
void trigger_action(int wake_word_index);
void setup_microphone();
bool read_audio_sample();

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=== Hebrew Wake Word Detection ===");
  Serial.println("Initializing ESP32-S3...");

  // Initialize LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Setup microphone
  setup_microphone();

  // Setup TensorFlow Lite Micro
  if (!setup_tensorflow()) {
    Serial.println("ERROR: Failed to setup TensorFlow Lite Micro!");
    while (1) {
      digitalWrite(LED_PIN, HIGH);
      delay(100);
      digitalWrite(LED_PIN, LOW);
      delay(100);
    }
  }

  Serial.println("System ready! Listening for Hebrew wake words...");
  Serial.println("Say 'shalom' (hello) or 'lehitraot' (goodbye)");
}

void loop() {
  // Check if audio buffer is ready
  if (buffer_ready) {
    buffer_ready = false;

    // Process audio and extract features
    process_audio();

    // Reset audio index for next capture
    audio_index = 0;
  }

  // Small delay to prevent overwhelming the system
  delay(10);
}

bool setup_tensorflow() {
  // Setup error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model
  model = tflite::GetModel(g_hebrew_wake_word_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
        "Model provided is schema version %d not equal to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  // Setup op resolver with required operations
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return false;
  }

  // Get input/output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TensorFlow Lite Micro initialized successfully");
  Serial.print("Input tensor shape: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print("x");
  }
  Serial.println();

  return true;
}

void setup_microphone() {
  // Configure I2S for PDM microphone input
  I2S.setDataInPin(MIC_DATA_PIN);
  I2S.setClkPin(MIC_CLOCK_PIN);
  I2S.setSampleRate(SAMPLE_RATE);
  I2S.setChannel(I2S_CHANNEL_MONO);
  I2S.setBitsPerSample(I2S_BITS_PER_SAMPLE_16BIT);

  if (!I2S.begin(I2S_MODE_PDM)) {
    Serial.println("ERROR: Failed to initialize I2S microphone!");
    while (1);
  }

  Serial.println("PDM microphone initialized");
}

void process_audio() {
  // Extract MFCC features from audio buffer
  float* features = extract_mfcc_features(audio_buffer, BUFFER_SIZE);

  if (features == nullptr) {
    Serial.println("ERROR: Failed to extract MFCC features");
    return;
  }

  // Run inference
  int wake_word_index = detect_wake_word(features);

  // Trigger action if wake word detected
  if (wake_word_index > 1) {  // Index 0= silence, 1=unknown, 2+=wake words
    trigger_action(wake_word_index);
  }

  // Free features memory
  delete[] features;
}

float* extract_mfcc_features(int16_t* audio_data, int data_size) {
  // Simplified MFCC extraction for microcontroller
  // In a full implementation, you'd use the TFLM signal processing library

  const int num_frames = 49;  // 49 frames of 40 MFCC bins each
  float* features = new float[num_frames * FEATURE_BINS];

  if (features == nullptr) {
    return nullptr;
  }

  // Simple feature extraction (placeholder)
  // In practice, this would implement proper MFCC computation
  for (int frame = 0; frame < num_frames; frame++) {
    int start_sample = frame * WINDOW_STRIDE;
    int end_sample = start_sample + WINDOW_SIZE;

    if (end_sample > data_size) break;

    // Calculate simple spectral features as placeholder
    float energy = 0.0f;
    for (int i = start_sample; i < end_sample; i++) {
      energy += abs(audio_data[i]);
    }
    energy /= WINDOW_SIZE;

    // Fill MFCC bins with simplified features
    for (int bin = 0; bin < FEATURE_BINS; bin++) {
      // Simple spectral representation
      float freq = (float)bin / FEATURE_BINS;
      features[frame * FEATURE_BINS + bin] = energy * (0.5f + 0.5f * sin(2 * PI * freq));
    }
  }

  return features;
}

int detect_wake_word(float* features) {
  // Prepare input tensor
  float* input_data = input->data.f;

  // Copy features to input tensor
  memcpy(input_data, features, input->bytes);

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return -1;
  }

  // Get output
  float* output_data = output->data.f;

  // Find the class with highest probability
  int max_index = 0;
  float max_prob = output_data[0];

  for (int i = 1; i < num_classes; i++) {
    if (output_data[i] > max_prob) {
      max_prob = output_data[i];
      max_index = i;
    }
  }

  // Check detection threshold and cooldown
  unsigned long current_time = millis();
  if (max_prob >= DETECTION_THRESHOLD &&
      (current_time - last_detection_time) > COOLDOWN_MS) {

    Serial.printf("Wake word detected: %s (confidence: %.2f)\n",
                  wake_words[max_index], max_prob);

    last_detection_time = current_time;
    return max_index;
  }

  return -1;  // No detection
}

void trigger_action(int wake_word_index) {
  // Visual feedback - blink LED
  digitalWrite(LED_PIN, HIGH);

  // Serial output with wake word info
  Serial.printf("🎤 WAKE WORD: %s detected!\n", wake_words[wake_word_index]);

  // Different blink patterns for different words
  if (wake_word_index == 2) {  // Shalom
    delay(500);
    digitalWrite(LED_PIN, LOW);
    delay(100);
    digitalWrite(LED_PIN, HIGH);
    delay(500);
  } else if (wake_word_index == 3) {  // Lehitraot
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(100);
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(100);
    digitalWrite(LED_PIN, HIGH);
    delay(200);
  }

  digitalWrite(LED_PIN, LOW);

  // Here you could add more actions:
  // - Send signal to other devices
  // - Trigger voice assistant
  // - Control smart home devices
  // - Send notification
}

// I2S interrupt handler for audio capture
void onI2SData() {
  if (audio_index < BUFFER_SIZE) {
    int16_t sample;
    if (I2S.read(&sample, sizeof(sample)) > 0) {
      audio_buffer[audio_index++] = sample;
    }
  } else if (!buffer_ready) {
    buffer_ready = true;
  }
}

// Optional: Power management functions
void enter_light_sleep() {
  // Light sleep to save power between detections
  esp_sleep_enable_timer_wakeup(100000);  // 100ms
  esp_light_sleep_start();
}

void setup_power_management() {
  // Configure power management for low power consumption
  setCpuFrequencyMhz(80);  // Reduce CPU frequency

  // Configure ADC and WiFi power down
  WiFi.mode(WIFI_OFF);
  btStop();

  Serial.println("Power management configured");
}

// Debug functions
void print_tensor_info() {
  Serial.println("\n=== Tensor Information ===");
  Serial.printf("Input tensor shape: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.printf("%d ", input->dims->data[i]);
  }
  Serial.println();

  Serial.printf("Output tensor shape: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.printf("%d ", output->dims->data[i]);
  }
  Serial.println();
}

void print_system_info() {
  Serial.println("\n=== System Information ===");
  Serial.printf("ESP32 Chip model: %s\n", ESP.getChipModel());
  Serial.printf("ESP32 Chip revision: %d\n", ESP.getChipRevision());
  Serial.printf("Total heap: %d bytes\n", ESP.getHeapSize());
  Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("CPU frequency: %d MHz\n", getCpuFrequencyMhz());
}
