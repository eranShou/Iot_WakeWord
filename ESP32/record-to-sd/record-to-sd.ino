#include "ESP_I2S.h"
#include "FS.h"
#include "SD.h"

const uint32_t SAMPLERATE = 16000;
const int LEN = 10;  // seconds
const byte ledPin = BUILTIN_LED;

I2SClass i2s;

void recordAudio(String userName) {
  static char filename[64];
  uint8_t *wav_buffer;
  size_t wav_size;

  Serial.print("RECORDING ... ");
  wav_buffer = i2s.recordWAV(LEN, &wav_size);

  sprintf(filename, "/%s.wav", userName.c_str());
  File file = SD.open(filename, FILE_WRITE);
  file.write(wav_buffer, wav_size);
  file.close();
  free(wav_buffer);
  Serial.printf("COMPLETE => %s\n", filename);
}

void setup() {
  Serial.begin(115200);
  Serial.println("start \n");
  Serial.setTimeout(0);

  pinMode(ledPin, OUTPUT);

  i2s.setPinsPdmRx(42, 41);
  if (!i2s.begin(I2S_MODE_PDM_RX, SAMPLERATE,
                 I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("Can't find microphone!");
  }

  if (!SD.begin(21)) {
    Serial.println("Failed to mount SD Card!");
  }
}

void loop() {

  Serial.println("enter file name");
  
  // Wait for filename input
  while (!Serial.available()) {
    delay(10);
  }
  String fileName = Serial.readStringUntil('\n');
  fileName.trim(); // Remove any whitespace/newlines

  Serial.println("type rec to record");
  
  // Wait for "rec" input
  while (!Serial.available()) {
    delay(10);
  }
  String input = Serial.readStringUntil('\n');
  input.trim(); // Remove any whitespace/newlines

  if (input.equalsIgnoreCase("rec")) {
    delay(500);
    digitalWrite(ledPin, LOW);
    recordAudio(fileName);
    digitalWrite(ledPin, HIGH);
  }
  else {
    Serial.println("Cancel");
  }
  
}