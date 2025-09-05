#include <ESP_I2S.h>
I2SClass I2S;

void setup() {
  // A baud rate of 115200 is used for a faster and stable data rate
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  // setup 42 PDM clock and 41 PDM data pins
  I2S.setPinsPdmRx(42, 41);

  // start I2S at 16 kHz with 16-bits per sample
  if (!I2S.begin(I2S_MODE_PDM_RX, 16000, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("Failed to initialize I2S!");
    while (1); // do nothing
  }
}

void loop() {
  // read a sample
  int32_t sample = I2S.read();
  
  // Check for valid data before sending
  if (sample) {
    // Write the raw 4 bytes of the 32-bit integer directly to the serial port
    Serial.write((byte *)&sample, sizeof(sample));
  }
}