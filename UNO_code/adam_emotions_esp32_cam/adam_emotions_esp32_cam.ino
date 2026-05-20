/*
 * adam_emotions_esp32_cam.ino
 * ADAM Emotion Face Renderer Runner (ESP32-CAM + ILI9341 + TFT_eSPI)
 *
 * Serial protocol from Pi:
 *   {"emotion":"happy"}
 *   {"emotion":"thinking"}
 *   ping
 *
 * Board: ESP32-CAM (AI Thinker) or ESP32 Dev Module with PSRAM
 */

#include <Arduino.h>
#include <TFT_eSPI.h>
#include "adam_emotions.h"

// Display objects required by adam_emotions.h
TFT_eSPI tft;
TFT_eSprite spr(&tft);

static String rxLine;

static void printHelp() {
  Serial.println();
  Serial.println("ADAM Emotion Renderer Ready");
  Serial.println("Send JSON over Serial:");
  Serial.println("  {\"emotion\":\"happy\"}");
  Serial.println("  {\"emotion\":\"sad\"}");
  Serial.println("Commands:");
  Serial.println("  ping  -> pong");
  Serial.println("  list  -> supported emotions");
  Serial.println("  get   -> current emotion");
  Serial.println();
}

static void printEmotionList() {
  Serial.print("emotions: ");
  for (int i = 0; i < EMO_COUNT; i++) {
    Serial.print(_EMO_NAMES[i]);
    if (i < EMO_COUNT - 1) Serial.print(", ");
  }
  Serial.println();
}

// Minimal parser for payloads like {"emotion":"happy"}
static bool parseEmotionFromJson(const String &line, String &emotionOut) {
  int keyPos = line.indexOf("\"emotion\"");
  if (keyPos < 0) return false;

  int colonPos = line.indexOf(':', keyPos);
  if (colonPos < 0) return false;

  int q1 = line.indexOf('"', colonPos + 1);
  if (q1 < 0) return false;

  int q2 = line.indexOf('"', q1 + 1);
  if (q2 < 0) return false;

  emotionOut = line.substring(q1 + 1, q2);
  emotionOut.trim();
  return emotionOut.length() > 0;
}

static void handleSerialLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  if (line.equalsIgnoreCase("ping")) {
    Serial.println("pong");
    return;
  }

  if (line.equalsIgnoreCase("list")) {
    printEmotionList();
    return;
  }

  if (line.equalsIgnoreCase("get")) {
    Serial.print("current: ");
    Serial.println(_EMO_NAMES[(int)getEmotion()]);
    return;
  }

  String emo;
  if (parseEmotionFromJson(line, emo)) {
    AdamEmotion next = nameToEmotion(emo.c_str());
    setEmotion(next);
    Serial.print("ok: ");
    Serial.println(_EMO_NAMES[(int)next]);
    return;
  }

  // Optional direct command support: happy, sad, thinking, etc.
  AdamEmotion next = nameToEmotion(line.c_str());
  if (next != EMO_IDLE || line.equalsIgnoreCase("idle")) {
    setEmotion(next);
    Serial.print("ok: ");
    Serial.println(_EMO_NAMES[(int)next]);
    return;
  }

  Serial.println("err: unknown input");
}

static void pollSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n' || c == '\r') {
      if (rxLine.length() > 0) {
        handleSerialLine(rxLine);
        rxLine = "";
      }
      continue;
    }

    if (rxLine.length() < 384) {
      rxLine += c;
    } else {
      // Guard against unbounded input.
      rxLine = "";
      Serial.println("err: input too long");
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);

  bool psramOk = psramInit();
  if (!psramOk) {
    Serial.println("warn: PSRAM init failed; sprite allocation may fail");
  }

  tft.init();
  initEmotions(tft, spr);
  setEmotion(EMO_IDLE);

  printHelp();
  printEmotionList();
}

void loop() {
  pollSerial();
  updateEmotion();
}
