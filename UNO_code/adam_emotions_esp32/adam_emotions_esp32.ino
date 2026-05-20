/*
 * adam_emotions_esp32.ino
 * ADAM Emotion Renderer for ESP32 Dev Module + ILI9341
 * Version: 2.1
 * Board: ESP32 Dev Module
 * Display: ILI9341 SPI 320x240 (landscape)
 * Pins: CS=5, RST=4, DC=2, MOSI=23, SCLK=18, MISO=19
 *
 * Demo mode:
 *   Cycles through all emotions every 5 seconds.
 *
 * Notes:
 * - Uses the local adam_emotions.h renderer again.
 * - Does not call psramInit() on plain ESP32 boards.
 * - Uses adam_emotions.h on top of Adafruit_ILI9341.
 */

#include <Arduino.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>
#include "adam_emotions.h"

#define TFT_CS    5
#define TFT_RST   4
#define TFT_DC    2
#define TFT_MOSI 23
#define TFT_SCLK 18
#define TFT_MISO 19

Adafruit_ILI9341 tft(TFT_CS, TFT_DC, TFT_RST);

static const uint32_t EMOTION_CYCLE_MS = 5000;
static uint32_t lastEmotionSwitchMs = 0;
static int emotionIndex = 0;

static void printEmotionList() {
  Serial.print("emotions: ");
  for (int i = 0; i < EMO_COUNT; i++) {
    Serial.print(_EMO_NAMES[i]);
    if (i < EMO_COUNT - 1) Serial.print(", ");
  }
  Serial.println();
}

static void cycleEmotionEvery5s() {
  uint32_t now = millis();
  if (now - lastEmotionSwitchMs < EMOTION_CYCLE_MS) return;

  lastEmotionSwitchMs = now;
  emotionIndex = (emotionIndex + 1) % EMO_COUNT;
  setEmotion((AdamEmotion)emotionIndex);

  Serial.print("Emotion: ");
  Serial.println(_EMO_NAMES[emotionIndex]);
}

void setup() {
  Serial.begin(115200);
  delay(200);

  SPI.begin(TFT_SCLK, TFT_MISO, TFT_MOSI);

  tft.begin();
  tft.invertDisplay(false);
  initEmotions(tft);

  Serial.print("info: sprite color depth = ");
  Serial.println((int)emotionSpriteDepth());
  Serial.println("info: running adam_emotions.h via Adafruit_ILI9341");

  emotionIndex = 0;
  setEmotion((AdamEmotion)emotionIndex);
  lastEmotionSwitchMs = millis();

  Serial.println("ADAM ESP32 Emotion Renderer Ready (adam_emotions.h mode)");
  printEmotionList();
}

void loop() {
  cycleEmotionEvery5s();
  updateEmotion();
}
