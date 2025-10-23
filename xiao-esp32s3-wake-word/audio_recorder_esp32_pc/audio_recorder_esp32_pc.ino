/*
 * ESP32 Audio Recorder - Stream to PC
 * For XIAO ESP32S3 with BUILT-IN MICROPHONE
 * 
 * Uses NEW ESP_I2S library with proper PDM support
 * Based on working code that records to SD card
 * 
 * Hardware: Seeed Studio XIAO ESP32S3
 * Microphone: Built-in PDM microphone (MSM261D3526H1CPM)
 * 
 * NO WIRING NEEDED - Uses built-in microphone!
 */

#include "ESP_I2S.h"
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

I2SClass i2s;

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
  
  // Initialize built-in PDM microphone using NEW I2S library
  Serial.println("Initializing built-in PDM microphone...");
  
  // Set PDM pins using configuration
  i2s.setPinsPdmRx(PDM_CLK_PIN, PDM_DATA_PIN);
  
  // Initialize PDM RX mode with 16-bit mono
  if (!i2s.begin(I2S_MODE_PDM_RX, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
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
  
  // Use the new I2S library's recordWAV function
  // This handles all the PDM->PCM conversion properly!
  size_t wav_size;
  uint8_t *wav_buffer = i2s.recordWAV(record_time, &wav_size);
  
  if (wav_buffer == NULL) {
    Serial.println(MSG_ERROR_RECORD);
    return;
  }
  
  // The recordWAV function returns a complete WAV file with header
  // We need to extract just the audio data (skip 44-byte WAV header)
  uint8_t *audio_data = wav_buffer + 44;
  uint32_t audio_size = wav_size - 44;
  
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
  free(wav_buffer);
  
  Serial.println();
  Serial.println(MSG_RECORDING_COMPLETE);
  Serial.println(MSG_WAITING);
}
