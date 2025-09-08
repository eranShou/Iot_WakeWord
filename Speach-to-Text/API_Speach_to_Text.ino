// Arduino Core and C++ Libraries
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <vector>
#include "base64.h" // Library for encoding audio data


// --- CONFIGURATION ---

// Wokwi Simulator Wi-Fi Credentials
const char *WIFI_SSID = "Wokwi-GUEST";
const char *WIFI_PASSWORD = ""; // No password for Wokwi's guest network

// IMPORTANT: You must get an API Key from the Google Cloud Console.
// 1. Go to https://console.cloud.google.com/
// 2. Create a new project.
// 3. Enable the "Cloud Speech-to-Text API".
// 4. Go to "Credentials" and create a new API Key.
const char *GOOGLE_CLOUD_API_KEY = "GOOGLE_CLOUD_API_KEY"; 



// A structure to hold a single transcription alternative
struct TranscriptionAlternative {
  String text;
  float confidence;
};

// A structure to hold the complete result from the API
struct TranscriptionResult {
  bool success;
  std::vector<TranscriptionAlternative> alternatives;
};

//
// --- 3. DUMMY AUDIO DATA ---
//
// This is a placeholder audio buffer. For a real transcription,
// you need to replace this with a real recording of a Hebrew word.
// Format: 16000Hz, 16-bit, mono, signed-integer, little-endian PCM.
const uint8_t shalom_audio_raw[] = {
    0x00, 0x00, 0x02, 0x00, 0x04, 0x00, 0x06, 0x00, 0x08, 0x00, 0x0A, 0x00, 0x0C, 0x00, 0x0E, 0x00,
    0x10, 0x00, 0x12, 0x00, 0x14, 0x00, 0x12, 0x00, 0x10, 0x00, 0x0E, 0x00, 0x0C, 0x00, 0x0A, 0x00
};


// --- 4. API COMMUNICATION FUNCTION (Google Cloud) ---

TranscriptionResult transcribeAudioBuffer(const uint8_t *audioBuffer, size_t bufferSize) {
  TranscriptionResult result;
  result.success = false;

  if (bufferSize == 0) {
    Serial.println("Error: Audio buffer provided is empty.");
    return result;
  }

  String encodedAudio = base64::encode(audioBuffer, bufferSize);
  
  DynamicJsonDocument jsonRequest(2048);
  JsonObject config = jsonRequest.createNestedObject("config");
  config["encoding"] = "LINEAR16";
  config["sampleRateHertz"] = 16000;
  config["languageCode"] = "he-IL"; // Language Hebrew
  config["profanityFilter"] = false;
  config["enableAutomaticPunctuation"] = false;
  config["model"] = "default";

  JsonObject audio = jsonRequest.createNestedObject("audio");
  audio["content"] = encodedAudio;

  String requestBody;
  serializeJson(jsonRequest, requestBody);
  
  // --- Send the HTTP POST Request ---
  HTTPClient http;
  String apiUrl = "https://speech.googleapis.com/v1/speech:recognize?key=" + String(GOOGLE_CLOUD_API_KEY);
  http.begin(apiUrl);
  http.addHeader("Content-Type", "application/json");

  Serial.println("Sending audio data to Google Cloud API...");
  int httpResponseCode = http.POST(requestBody);

  if (httpResponseCode > 0) {
    String payload = http.getString();
    Serial.printf("HTTP Response code: %d\n", httpResponseCode);
    Serial.println("Received payload: " + payload);

    if(httpResponseCode == 200) {
        DynamicJsonDocument jsonResponse(2048);
        deserializeJson(jsonResponse, payload);
        
        if (jsonResponse.containsKey("results")) {
            JsonArray results = jsonResponse["results"];
            if (results.size() > 0) {
                JsonArray alternatives = results[0]["alternatives"];
                for(JsonObject alt : alternatives) {
                    TranscriptionAlternative newAlt;
                    newAlt.text = alt["transcript"].as<String>();
                    newAlt.confidence = alt["confidence"].as<float>();
                    result.alternatives.push_back(newAlt);
                }
                result.success = true;
            }
        }
    }
  } else {
    Serial.printf("HTTP Error sending request. Code: %d\n", httpResponseCode);
  }
  
  http.end();
  return result;
}


// --- Test ---


void setup() {
  Serial.begin(115200);
  
  // --- Connect to Wi-Fi ---
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");
}

void loop() {
  Serial.println("\n--- Sending hardcoded audio buffer for 'שלום' ---");

  size_t audioSize = sizeof(shalom_audio_raw);
  TranscriptionResult result = transcribeAudioBuffer(shalom_audio_raw, audioSize);

  if (result.success && !result.alternatives.empty()) {
    Serial.println("\n--- Transcription Alternatives (Hebrew) ---");
    Serial.println("!!! IMPORTANT: Use a UTF-8 compatible Serial Monitor to see Hebrew characters correctly. !!!");
    
    for (const auto& alt : result.alternatives) {
      Serial.printf("Text: %s, Confidence: %.2f%%\n", alt.text.c_str(), alt.confidence * 100);
    }
  } else {
    Serial.println("\n--- Transcription Failed ---");
  }
  
  delay(10000); 
}

