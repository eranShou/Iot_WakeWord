/*
 * ESP32 Audio Recorder - Stream to PC
 * For XIAO ESP32S3 with BUILT-IN MICROPHONE
 * 
 * Uses legacy I2S driver (IDF 4.x API) compatible with Board Manager 2.0.16
 * Based on working code that records to SD card
 * 
 * Hardware: Seeed Studio XIAO ESP32S3
 * Microphone: Built-in PDM microphone (MSM261D3526H1CPM)
 * 
 * NO WIRING NEEDED - Uses built-in microphone!
 */

#include <driver/i2s.h>
#include <Arduino.h>
#include <math.h>
#include <stdint.h>
#include "mic_config.h"

// Global variables
uint8_t record_time = DEFAULT_RECORD_TIME;
float current_gain = DEFAULT_GAIN;
bool gain_enabled = GAIN_ENABLED_BY_DEFAULT;

// Compression variables
bool compression_enabled = COMPRESSION_ENABLED_BY_DEFAULT;
float compression_threshold = COMPRESSION_THRESHOLD;
float compression_ratio = COMPRESSION_RATIO;
float compression_makeup_gain = COMPRESSION_MAKEUP_GAIN;
float compression_attack_coeff = 1.0f - exp(-1.0f / (COMPRESSION_ATTACK_MS * SAMPLE_RATE / 1000.0f));
float compression_release_coeff = 1.0f - exp(-1.0f / (COMPRESSION_RELEASE_MS * SAMPLE_RATE / 1000.0f));
float compression_envelope = 0.0f;

// I2S configuration
bool i2sInitialized = false;

// Function to initialize I2S with PDM microphone
bool initI2S() {
  // Configure legacy I2S in PDM RX mode (IDF 4.x API compatible with Arduino core 2.0.x)
  i2s_config_t i2s_config;
  memset(&i2s_config, 0, sizeof(i2s_config));
  i2s_config.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM);
  i2s_config.sample_rate = SAMPLE_RATE;
  i2s_config.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  i2s_config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  i2s_config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  i2s_config.intr_alloc_flags = 0;
  i2s_config.dma_buf_count = 8;
  i2s_config.dma_buf_len = 256;
  i2s_config.use_apll = false;
  i2s_config.tx_desc_auto_clear = false;
  i2s_config.fixed_mclk = 0;

  if (i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL) != ESP_OK) {
    Serial.println("ERROR: i2s_driver_install failed");
    return false;
  }

  i2s_pin_config_t pin_config;
  pin_config.bck_io_num = I2S_PIN_NO_CHANGE; // Not used in PDM RX
  pin_config.ws_io_num = PDM_CLK_PIN;        // PDM clock
  pin_config.data_out_num = I2S_PIN_NO_CHANGE;
  pin_config.data_in_num = PDM_DATA_PIN;     // PDM data
  pin_config.mck_io_num = I2S_PIN_NO_CHANGE;

  if (i2s_set_pin(I2S_NUM_0, &pin_config) != ESP_OK) {
    Serial.println("ERROR: i2s_set_pin failed");
    i2s_driver_uninstall(I2S_NUM_0);
    return false;
  }

  // Set PDM RX clock and sample rate
  i2s_set_clk(I2S_NUM_0, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_MONO);
  i2s_set_pdm_rx_down_sample(I2S_NUM_0, I2S_PDM_DSR_8S);

  i2sInitialized = true;
  return true;
}

// Function to record audio using legacy I2S driver
bool recordAudioI2S(uint8_t* audio_buffer, uint32_t buffer_size) {
  if (!i2sInitialized) {
    Serial.println("ERROR: I2S not initialized");
    return false;
  }

  size_t bytes_read = 0;
  uint32_t total_bytes_read = 0;
  
  // Read audio data in chunks
  while (total_bytes_read < buffer_size) {
    size_t bytes_to_read = min((size_t)(buffer_size - total_bytes_read), (size_t)1024);
    
    if (i2s_read(I2S_NUM_0, audio_buffer + total_bytes_read, bytes_to_read, &bytes_read, portMAX_DELAY) != ESP_OK) {
      Serial.println("ERROR: i2s_read failed");
      return false;
    }
    
    total_bytes_read += bytes_read;
    
    // Small delay to prevent overwhelming the system
    delay(1);
  }
  
  return true;
}

// Function to apply compression and gain to 16-bit audio samples
void applyAudioCompressionAndGain(uint8_t* audio_data, uint32_t audio_size) {
  // Process 16-bit samples (2 bytes per sample)
  int16_t* samples = (int16_t*)audio_data;
  uint32_t sample_count = audio_size / BYTES_PER_SAMPLE;
  
  // Reset compression envelope for new recording
  compression_envelope = 0.0f;
  
  for (uint32_t i = 0; i < sample_count; i++) {
    float sample = (float)samples[i];
    
    // Step 1: Apply initial gain
    if (gain_enabled && current_gain > 1.0f) {
      sample *= current_gain;
    }
    
    // Step 2: Apply compression if enabled
    if (compression_enabled) {
      // Calculate input level (absolute value normalized to 0-1)
      float input_level = abs(sample) / 32767.0f;
      
      // Update compression envelope with attack/release
      if (input_level > compression_envelope) {
        // Attack phase
        compression_envelope += compression_attack_coeff * (input_level - compression_envelope);
      } else {
        // Release phase
        compression_envelope += compression_release_coeff * (input_level - compression_envelope);
      }
      
      // Apply compression if above threshold
      if (compression_envelope > compression_threshold) {
        float over_threshold = compression_envelope - compression_threshold;
        float compression_factor = 1.0f - (over_threshold * (1.0f - 1.0f/compression_ratio));
        sample *= compression_factor;
      }
      
      // Apply makeup gain after compression
      sample *= compression_makeup_gain;
    }
    
    // Step 3: Final limiting to prevent clipping
    if (sample > AUDIO_MAX_POSITIVE) {
      sample = AUDIO_MAX_POSITIVE;
    } else if (sample < AUDIO_MAX_NEGATIVE) {
      sample = AUDIO_MAX_NEGATIVE;
    }
    
    // Convert back to 16-bit integer
    samples[i] = (int16_t)sample;
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  delay(2000);
  
  Serial.println();
  Serial.println("========================================");
  Serial.println("ESP32 Audio Recorder - PC Streaming");
  Serial.println("========================================");
  Serial.println(MSG_READY);
  
  // Initialize built-in PDM microphone using legacy I2S driver
  Serial.println("Initializing built-in PDM microphone...");
  
  // Initialize I2S with PDM configuration
  if (!initI2S()) {
    Serial.println(MSG_ERROR_MICROPHONE);
    Serial.println(MSG_ERROR_BOARD);
    while (1) {
      delay(DEBUG_DELAY_MS);
    }
  }
  
  Serial.println(MSG_I2S_OK);
  Serial.println(MSG_WAITING);
  Serial.println();
  Serial.print("Audio gain: ");
  Serial.print(gain_enabled ? "ON" : "OFF");
  Serial.print(" (");
  Serial.print(current_gain);
  Serial.println("x)");
  Serial.print("Audio compression: ");
  Serial.print(compression_enabled ? "ON" : "OFF");
  Serial.print(" (threshold: ");
  Serial.print(compression_threshold);
  Serial.print(", ratio: ");
  Serial.print(compression_ratio);
  Serial.print(":1)");
  Serial.println();
  Serial.println("Ready to record! Waiting for command from PC...");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == CMD_RECORD) {
      recordAndSendAudio();
    } else if (command.startsWith(CMD_RECORD_PREFIX)) {
      // Parse custom duration: RECORD:10 (for 10 seconds)
      int colonIndex = command.indexOf(':');
      if (colonIndex != -1) {
        String durationStr = command.substring(colonIndex + 1);
        int custom_duration = durationStr.toInt();
        
        // Validate duration
        if (IS_VALID_DURATION(custom_duration)) {
          record_time = custom_duration;
          Serial.print("Duration set to: ");
          Serial.print(record_time);
          Serial.println(" seconds");
          recordAndSendAudio();
        } else {
          Serial.print("ERROR: Duration must be between ");
          Serial.print(MIN_RECORD_TIME);
          Serial.print(" and ");
          Serial.print(MAX_RECORD_TIME);
          Serial.println(" seconds");
        }
      }
    } else if (command == CMD_STATUS) {
      Serial.println(MSG_READY);
    } else if (command == CMD_SET_DURATION) {
      Serial.print("Current duration: ");
      Serial.print(record_time);
      Serial.println(" seconds");
      Serial.print("Range: ");
      Serial.print(MIN_RECORD_TIME);
      Serial.print("-");
      Serial.print(MAX_RECORD_TIME);
      Serial.println(" seconds");
    } else if (command.startsWith(CMD_GAIN_PREFIX)) {
      // Parse gain setting: GAIN:2.5 (for 2.5x gain)
      int colonIndex = command.indexOf(':');
      if (colonIndex != -1) {
        String gainStr = command.substring(colonIndex + 1);
        float new_gain = gainStr.toFloat();
        
        // Validate gain
        if (IS_VALID_GAIN(new_gain)) {
          current_gain = new_gain;
          Serial.print("Gain set to: ");
          Serial.print(current_gain);
          Serial.println("x");
        } else {
          Serial.print("ERROR: Gain must be between ");
          Serial.print(MIN_GAIN);
          Serial.print(" and ");
          Serial.print(MAX_GAIN);
        }
      }
    } else if (command == CMD_GAIN_ON) {
      gain_enabled = true;
      Serial.println("Audio gain enabled");
    } else if (command == CMD_GAIN_OFF) {
      gain_enabled = false;
      Serial.println("Audio gain disabled");
    } else if (command == CMD_GAIN_STATUS) {
      Serial.print("Gain: ");
      Serial.print(gain_enabled ? "ON" : "OFF");
      Serial.print(" (");
      Serial.print(current_gain);
      Serial.println("x)");
    } else if (command.startsWith(CMD_COMPRESSION_PREFIX)) {
      // Parse compression setting: COMPRESSION:0.8 (for 0.8 threshold)
      int colonIndex = command.indexOf(':');
      if (colonIndex != -1) {
        String compressionStr = command.substring(colonIndex + 1);
        float new_threshold = compressionStr.toFloat();
        
        // Validate threshold (0.0-1.0)
        if (new_threshold >= 0.0f && new_threshold <= 1.0f) {
          compression_threshold = new_threshold;
          Serial.print("Compression threshold set to: ");
          Serial.println(compression_threshold);
        } else {
          Serial.println("ERROR: Compression threshold must be between 0.0 and 1.0");
        }
      }
    } else if (command == CMD_COMPRESSION_ON) {
      compression_enabled = true;
      Serial.println("Audio compression enabled");
    } else if (command == CMD_COMPRESSION_OFF) {
      compression_enabled = false;
      Serial.println("Audio compression disabled");
    } else if (command == CMD_COMPRESSION_STATUS) {
      Serial.print("Compression: ");
      Serial.print(compression_enabled ? "ON" : "OFF");
      Serial.print(" (threshold: ");
      Serial.print(compression_threshold);
      Serial.print(", ratio: ");
      Serial.print(compression_ratio);
      Serial.print(":1, makeup: ");
      Serial.print(compression_makeup_gain);
      Serial.println("x)");
    }
  }
  
  delay(10);
}

void recordAndSendAudio() {
  Serial.println(MSG_RECORDING_START);
  
  // Send audio configuration
  Serial.print(MSG_SAMPLE_RATE);
  Serial.println(SAMPLE_RATE);
  Serial.print(MSG_BITS);
  Serial.println(BITS_PER_SAMPLE);
  Serial.print(MSG_CHANNELS);
  Serial.println(NUM_CHANNELS);
  
  // Calculate data size using macro
  uint32_t data_size = CALCULATE_AUDIO_SIZE(record_time);
  Serial.print(MSG_DATA_SIZE);
  Serial.println(data_size);
  
  // Flush serial to ensure all text is sent before binary data
  Serial.flush();
  delay(SETUP_DELAY_MS);
  
  // Allocate buffer for raw audio data
  uint32_t audio_size = CALCULATE_AUDIO_SIZE(record_time);
  uint8_t *audio_buffer = (uint8_t*)malloc(audio_size);
  
  if (audio_buffer == NULL) {
    Serial.println("ERROR: Failed to allocate audio buffer");
    Serial.println(MSG_ERROR_RECORD);
    return;
  }
  
  // Record audio using legacy I2S driver
  Serial.println("Recording audio...");
  if (!recordAudioI2S(audio_buffer, audio_size)) {
    Serial.println(MSG_ERROR_RECORD);
    free(audio_buffer);
    return;
  }
  
  uint8_t *audio_data = audio_buffer;
  
  // Apply compression and gain to boost audio volume if enabled
  if ((gain_enabled && current_gain > 1.0f) || compression_enabled) {
    Serial.print("Applying audio processing: ");
    if (compression_enabled) {
      Serial.print("Compression + ");
    }
    if (gain_enabled && current_gain > 1.0f) {
      Serial.print(current_gain);
      Serial.print("x gain");
    }
    Serial.println("...");
    applyAudioCompressionAndGain(audio_data, audio_size);
  }
  
  // Send start marker (PC should be waiting for this now)
  uint32_t marker = START_MARKER;
  Serial.write((uint8_t*)&marker, sizeof(marker));
  
  // Send audio data in chunks
  for (uint32_t i = 0; i < audio_size; i += CHUNK_SIZE) {
    uint32_t chunk = (audio_size - i) < CHUNK_SIZE ? (audio_size - i) : CHUNK_SIZE;
    Serial.write(audio_data + i, chunk);
    Serial.flush();  // Make sure data is sent
  }
  
  // Send end marker
  marker = END_MARKER;
  Serial.write((uint8_t*)&marker, sizeof(marker));
  Serial.flush();
  
  // Clean up
  free(audio_buffer);
  
  Serial.println();
  Serial.println(MSG_RECORDING_COMPLETE);
  Serial.println(MSG_WAITING);
}
