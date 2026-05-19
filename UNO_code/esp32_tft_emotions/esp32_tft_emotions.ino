/*
 * esp32_tft_emotions.ino
 * ADAM ESP32 ILI9341 Emotion Cycle Demo
 *
 * Board: ESP32
 * Display: ILI9341 (320x240)
 * Libraries:
 *   - Adafruit GFX Library
 *   - Adafruit ILI9341
 *
 * Pins (as requested):
 *   TFT_CS   = 5
 *   TFT_RST  = 4
 *   TFT_DC   = 2
 *   TFT_MOSI = 23
 *   TFT_SCLK = 18
 *   TFT_MISO = 19
 */

#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>

// TFT Pins
#define TFT_CS    5
#define TFT_RST   4
#define TFT_DC    2

#define TFT_MOSI 23
#define TFT_SCLK 18
#define TFT_MISO 19

// Display object
Adafruit_ILI9341 tft = Adafruit_ILI9341(TFT_CS, TFT_DC, TFT_RST);

static const uint32_t EMOTION_INTERVAL_MS = 5000;

// Emotions from your folder
static const char* EMOTIONS[] = {
  "angry",
  "confused",
  "happy",
  "ideal",
  "love",
  "panic",
  "reconnecting",
  "rizz",
  "sad",
  "search-thinking",
  "shy",
  "sleep",
  "speeking",
  "surprised"
};

static const uint8_t EMOTION_COUNT = sizeof(EMOTIONS) / sizeof(EMOTIONS[0]);

uint8_t currentEmotion = 0;
uint32_t lastSwitchMs = 0;

// Face anchor points for landscape 320x240
static const int FACE_CX = 160;
static const int FACE_CY = 135;
static const int LEFT_EYE_X = 105;
static const int RIGHT_EYE_X = 215;
static const int EYE_Y = 105;
static const int MOUTH_Y = 170;

void drawHeader(const char* emotionName, uint16_t fg) {
  tft.fillRect(0, 0, 320, 38, ILI9341_BLACK);
  tft.drawFastHLine(0, 38, 320, fg);

  tft.setTextColor(fg);
  tft.setTextSize(2);
  tft.setCursor(8, 10);
  tft.print("ADAM: ");
  tft.print(emotionName);
}

void drawFaceOutline(uint16_t color) {
  tft.drawCircle(FACE_CX, FACE_CY, 88, color);
}

void drawEyeCircle(int x, int y, int rOuter, int rInner, uint16_t outerColor, uint16_t innerColor) {
  tft.fillCircle(x, y, rOuter, outerColor);
  tft.fillCircle(x, y, rInner, innerColor);
}

void drawBlinkEyes(uint16_t color) {
  tft.drawFastHLine(LEFT_EYE_X - 20, EYE_Y, 40, color);
  tft.drawFastHLine(RIGHT_EYE_X - 20, EYE_Y, 40, color);
}

void drawNeutralEyes(uint16_t eyeColor, uint16_t pupilColor) {
  drawEyeCircle(LEFT_EYE_X, EYE_Y, 18, 7, eyeColor, pupilColor);
  drawEyeCircle(RIGHT_EYE_X, EYE_Y, 18, 7, eyeColor, pupilColor);
}

void drawWideEyes(uint16_t eyeColor, uint16_t pupilColor) {
  drawEyeCircle(LEFT_EYE_X, EYE_Y, 22, 8, eyeColor, pupilColor);
  drawEyeCircle(RIGHT_EYE_X, EYE_Y, 22, 8, eyeColor, pupilColor);
}

void drawAngryBrows(uint16_t color) {
  tft.drawLine(LEFT_EYE_X - 24, EYE_Y - 22, LEFT_EYE_X + 20, EYE_Y - 10, color);
  tft.drawLine(LEFT_EYE_X - 24, EYE_Y - 23, LEFT_EYE_X + 20, EYE_Y - 11, color);

  tft.drawLine(RIGHT_EYE_X - 20, EYE_Y - 10, RIGHT_EYE_X + 24, EYE_Y - 22, color);
  tft.drawLine(RIGHT_EYE_X - 20, EYE_Y - 11, RIGHT_EYE_X + 24, EYE_Y - 23, color);
}

void drawConfusedBrows(uint16_t color) {
  tft.drawLine(LEFT_EYE_X - 20, EYE_Y - 24, LEFT_EYE_X + 20, EYE_Y - 14, color);
  tft.drawLine(RIGHT_EYE_X - 20, EYE_Y - 14, RIGHT_EYE_X + 20, EYE_Y - 24, color);
}

void drawSmile(uint16_t color) {
  tft.drawRoundRect(120, MOUTH_Y - 6, 80, 30, 14, color);
  tft.fillRect(121, MOUTH_Y - 6, 78, 14, ILI9341_BLACK);
}

void drawBigSmile(uint16_t color) {
  tft.drawRoundRect(110, MOUTH_Y - 8, 100, 42, 16, color);
  tft.fillRect(111, MOUTH_Y - 8, 98, 18, ILI9341_BLACK);
}

void drawFlatMouth(uint16_t color) {
  tft.fillRect(125, MOUTH_Y + 8, 70, 6, color);
}

void drawSadMouth(uint16_t color) {
  tft.drawRoundRect(120, MOUTH_Y + 8, 80, 30, 14, color);
  tft.fillRect(121, MOUTH_Y + 24, 78, 13, ILI9341_BLACK);
}

void drawOpenMouth(uint16_t color) {
  tft.drawRoundRect(134, MOUTH_Y - 4, 52, 42, 14, color);
  tft.fillRoundRect(138, MOUTH_Y, 44, 34, 10, color);
}

void drawZzz(uint16_t color) {
  tft.setTextColor(color);
  tft.setTextSize(2);
  tft.setCursor(238, 68);
  tft.print("Z");
  tft.setCursor(255, 53);
  tft.print("Z");
  tft.setCursor(272, 40);
  tft.print("Z");
}

void drawHeart(int x, int y, uint16_t color) {
  tft.fillCircle(x - 8, y - 6, 9, color);
  tft.fillCircle(x + 8, y - 6, 9, color);
  tft.fillTriangle(x - 17, y - 3, x + 17, y - 3, x, y + 18, color);
}

void renderEmotion(uint8_t idx) {
  const char* e = EMOTIONS[idx];

  // HTML-like style: black background + white vector lines.
  if (strcmp(e, "angry") == 0) {
    tft.fillScreen(ILI9341_BLACK);
    drawHeader(e, ILI9341_WHITE);
    drawFaceOutline(ILI9341_WHITE);
    tft.drawLine(82, 72, 128, 96, ILI9341_WHITE);
    tft.drawLine(238, 72, 192, 96, ILI9341_WHITE);
    tft.drawLine(83, 73, 129, 97, ILI9341_WHITE);
    tft.drawLine(237, 73, 191, 97, ILI9341_WHITE);
    tft.drawLine(108, 174, 133, 165, ILI9341_WHITE);
    tft.drawLine(133, 165, 159, 174, ILI9341_WHITE);
    tft.drawLine(159, 174, 185, 165, ILI9341_WHITE);
    tft.drawLine(185, 165, 208, 174, ILI9341_WHITE);
  }
  else if (strcmp(e, "confused") == 0) {
    tft.fillScreen(ILI9341_BLACK);
    drawHeader(e, ILI9341_WHITE);
    drawFaceOutline(ILI9341_WHITE);
    drawNeutralEyes(ILI9341_WHITE, ILI9341_BLACK);
    drawConfusedBrows(ILI9341_WHITE);
    tft.drawCircle(160, MOUTH_Y + 18, 12, ILI9341_WHITE);
    tft.drawFastVLine(160, MOUTH_Y + 6, 10, ILI9341_WHITE);
  }
  else if (strcmp(e, "happy") == 0) {
    tft.fillScreen(ILI9341_BLACK);
    drawHeader(e, ILI9341_WHITE);
    drawFaceOutline(ILI9341_WHITE);
    tft.drawLine(78, 95, 105, 66, ILI9341_WHITE);
    tft.drawLine(105, 66, 132, 95, ILI9341_WHITE);
    tft.drawLine(188, 95, 215, 66, ILI9341_WHITE);
    tft.drawLine(215, 66, 242, 95, ILI9341_WHITE);
    tft.drawCircle(58, 55, 4, ILI9341_WHITE);
    tft.drawCircle(262, 60, 4, ILI9341_WHITE);
    tft.drawCircle(280, 128, 3, ILI9341_WHITE);
    tft.drawLine(146, 173, 160, 185, ILI9341_WHITE);
    tft.drawLine(160, 185, 174, 173, ILI9341_WHITE);
  }
  else if (strcmp(e, "ideal") == 0) {
    tft.fillScreen(ILI9341_BLACK);
    drawHeader(e, ILI9341_WHITE);
    drawFaceOutline(ILI9341_WHITE);
    drawNeutralEyes(ILI9341_WHITE, ILI9341_BLACK);
    drawFlatMouth(ILI9341_WHITE);
  }
  else if (strcmp(e, "love") == 0) {
    tft.fillScreen(ILI9341_BLACK);
    drawHeader(e, ILI9341_WHITE);
    drawFaceOutline(ILI9341_WHITE);
    drawHeart(LEFT_EYE_X, EYE_Y, ILI9341_WHITE);
    drawHeart(RIGHT_EYE_X, EYE_Y, ILI9341_WHITE);
    drawSmile(ILI9341_WHITE);
  }
  else if (strcmp(e, "panic") == 0) {
    tft.fillScreen(ILI9341_BLACK);
    drawHeader(e, ILI9341_WHITE);
    drawFaceOutline(ILI9341_WHITE);
    drawWideEyes(ILI9341_WHITE, ILI9341_BLACK);
    drawOpenMouth(ILI9341_WHITE);
    tft.drawTriangle(260, 65, 275, 95, 245, 95, ILI9341_WHITE);
    tft.drawLine(260, 73, 260, 86, ILI9341_BLACK);
    tft.fillCircle(260, 91, 2, ILI9341_BLACK);
  }
  else if (strcmp(e, "reconnecting") == 0) {
    tft.fillScreen(ILI9341_BLACK);
    drawHeader(e, ILI9341_WHITE);
    drawFaceOutline(ILI9341_WHITE);
    drawBlinkEyes(ILI9341_WHITE);
    tft.drawCircle(160, MOUTH_Y + 8, 16, ILI9341_WHITE);
    tft.fillTriangle(160, MOUTH_Y - 12, 172, MOUTH_Y - 2, 148, MOUTH_Y - 2, ILI9341_BLACK);

    tft.drawCircle(272, 118, 12, ILI9341_WHITE);
    tft.drawFastVLine(272, 106, 8, ILI9341_WHITE);
  }
  else if (strcmp(e, "rizz") == 0) {
    tft.fillScreen(ILI9341_BLACK);
    drawHeader(e, ILI9341_WHITE);
    drawFaceOutline(ILI9341_WHITE);
    drawNeutralEyes(ILI9341_WHITE, ILI9341_BLACK);
    tft.drawLine(120, MOUTH_Y + 16, 196, MOUTH_Y + 6, ILI9341_WHITE);
    tft.drawLine(120, MOUTH_Y + 17, 196, MOUTH_Y + 7, ILI9341_WHITE);
  }
  else if (strcmp(e, "sad") == 0) {
    tft.fillScreen(ILI9341_BLACK);
    drawHeader(e, ILI9341_WHITE);
    drawFaceOutline(ILI9341_WHITE);
    tft.drawLine(78, 86, 105, 112, ILI9341_WHITE);
    tft.drawLine(105, 112, 132, 86, ILI9341_WHITE);
    tft.drawLine(188, 86, 215, 112, ILI9341_WHITE);
    tft.drawLine(215, 112, 242, 86, ILI9341_WHITE);
    drawSadMouth(ILI9341_WHITE);
    tft.fillCircle(105, 150, 3, ILI9341_WHITE);
    tft.fillCircle(215, 150, 3, ILI9341_WHITE);
  }
  else if (strcmp(e, "search-thinking") == 0) {
    tft.fillScreen(ILI9341_BLACK);
    drawHeader(e, ILI9341_WHITE);
    drawFaceOutline(ILI9341_WHITE);
    drawNeutralEyes(ILI9341_WHITE, ILI9341_BLACK);
    tft.drawCircle(LEFT_EYE_X, EYE_Y, 22, ILI9341_WHITE);
    tft.drawLine(116, 124, 130, 138, ILI9341_WHITE);
    drawFlatMouth(ILI9341_WHITE);

    tft.setTextColor(ILI9341_WHITE);
    tft.setTextSize(2);
    tft.setCursor(248, 118);
    tft.print("?");
  }
  else if (strcmp(e, "shy") == 0) {
    tft.fillScreen(ILI9341_BLACK);
    drawHeader(e, ILI9341_WHITE);
    drawFaceOutline(ILI9341_WHITE);
    drawBlinkEyes(ILI9341_WHITE);
    drawSmile(ILI9341_WHITE);
    tft.drawCircle(78, 145, 10, ILI9341_WHITE);
    tft.drawCircle(242, 145, 10, ILI9341_WHITE);
  }
  else if (strcmp(e, "sleep") == 0) {
    tft.fillScreen(ILI9341_BLACK);
    drawHeader(e, ILI9341_WHITE);
    drawFaceOutline(ILI9341_WHITE);
    drawBlinkEyes(ILI9341_WHITE);
    drawFlatMouth(ILI9341_WHITE);
    drawZzz(ILI9341_WHITE);
  }
  else if (strcmp(e, "speeking") == 0) {
    tft.fillScreen(ILI9341_BLACK);
    drawHeader(e, ILI9341_WHITE);
    drawFaceOutline(ILI9341_WHITE);
    drawNeutralEyes(ILI9341_WHITE, ILI9341_BLACK);
    drawOpenMouth(ILI9341_WHITE);

    tft.drawCircle(250, 150, 9, ILI9341_WHITE);
    tft.drawCircle(267, 143, 14, ILI9341_WHITE);
    tft.drawCircle(288, 132, 19, ILI9341_WHITE);
  }
  else if (strcmp(e, "surprised") == 0) {
    tft.fillScreen(ILI9341_BLACK);
    drawHeader(e, ILI9341_WHITE);
    drawFaceOutline(ILI9341_WHITE);
    drawWideEyes(ILI9341_WHITE, ILI9341_BLACK);
    drawOpenMouth(ILI9341_WHITE);
  }
}

void showCurrentEmotion() {
  Serial.print("Emotion: ");
  Serial.println(EMOTIONS[currentEmotion]);
  renderEmotion(currentEmotion);
}

void setup() {
  Serial.begin(115200);

  // Start SPI with custom ESP32 pins
  SPI.begin(TFT_SCLK, TFT_MISO, TFT_MOSI);

  // Initialize TFT
  tft.begin();
  tft.setRotation(1);
  tft.fillScreen(ILI9341_BLACK);

  showCurrentEmotion();
  lastSwitchMs = millis();
}

void loop() {
  uint32_t now = millis();
  if (now - lastSwitchMs >= EMOTION_INTERVAL_MS) {
    lastSwitchMs = now;
    currentEmotion = (currentEmotion + 1) % EMOTION_COUNT;
    showCurrentEmotion();
  }
}
