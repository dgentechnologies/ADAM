// ═══════════════════════════════════════════════════════════════
// adam_emotions.h  —  ADAM v3 TFT Emotion Renderer
// Dgen Technologies Pvt. Ltd.  |  May 2026
//
// Target  : ESP32-CAM + ILI9341 320×240 (landscape)
// Library : TFT_eSPI (≥2.5)  —  configure User_Setup.h:
//
//   #define ILI9341_DRIVER
//   #define TFT_CS   15
//   #define TFT_DC    2
//   #define TFT_MOSI 13
//   #define TFT_SCLK 14
//   #define TFT_MISO 12   ← already used; keep for SPI bus
//   #define SPI_FREQUENCY 40000000
//   #define TFT_WIDTH  240
//   #define TFT_HEIGHT 320
//
// Sprite double-buffer needs ~150 KB PSRAM.
// ESP32-CAM (AI-Thinker) has 4 MB PSRAM — fine.
// Call psramInit() in setup() before initEmotions().
//
// USAGE (in your .ino):
//
//   #include <TFT_eSPI.h>
//   #include "adam_emotions.h"
//
//   TFT_eSPI    tft;
//   TFT_eSprite spr(&tft);
//
//   void setup() {
//     psramInit();
//     tft.init();
//     initEmotions(tft, spr);
//     setEmotion(EMO_IDLE);
//   }
//   void loop() {
//     updateEmotion();
//     // On JSON from Pi: setEmotion(nameToEmotion("happy"));
//   }
//
// Pi sends JSON:  {"emotion":"happy"}
// ESP32-CAM calls: setEmotion(nameToEmotion(jsonStr));
// ═══════════════════════════════════════════════════════════════

#pragma once
#include <TFT_eSPI.h>
#include <math.h>
#include <string.h>

// ─── Color palette ───────────────────────────────────────────
// White is the primary facial color. Accents used sparingly.
#define COL_BG      0x0000   // Black background
#define COL_WHITE   0xFFFF   // Face elements
#define COL_DIM     0x8410   // 50% grey  (mouth idle, reconnect)
#define COL_DIM2    0x4208   // 25% grey  (faint effects)
#define COL_PINK    0xFC18   // Blush / shy
#define COL_BLUE    0x841F   // Tears / sweat drops
#define COL_RED     0xF800   // Angry vein accent
#define COL_YELLOW  0xFFE0   // Sparkle warm
#define COL_ORANGE  0xFD20   // Heat shimmer

// ─── Layout (landscape 320×240) ──────────────────────────────
#define SCR_W     320
#define SCR_H     240
#define EYE_L_X   105
#define EYE_R_X   215
#define EYE_Y      88
#define MOUTH_Y   148
#define FACE_CX   160

// ─── Emotion enum ────────────────────────────────────────────
enum AdamEmotion : uint8_t {
  EMO_IDLE = 0,
  EMO_SPEAKING,
  EMO_HAPPY,
  EMO_SAD,
  EMO_ANGRY,
  EMO_PANIC,
  EMO_SURPRISED,
  EMO_SHY,
  EMO_SLEEP,
  EMO_THINKING,
  EMO_RECONNECTING,
  EMO_LOVE,
  EMO_CONFUSED,
  EMO_RIZZ,
  EMO_COUNT
};

static const char* _EMO_NAMES[EMO_COUNT] = {
  "idle","speaking","happy","sad","angry",
  "panic","surprised","shy","sleep","thinking",
  "reconnecting","love","confused","rizz"
};

AdamEmotion nameToEmotion(const char* name) {
  for (int i = 0; i < EMO_COUNT; i++)
    if (strcmp(name, _EMO_NAMES[i]) == 0) return (AdamEmotion)i;
  return EMO_IDLE;
}

// ─── Internal state ──────────────────────────────────────────
static AdamEmotion _emo      = EMO_IDLE;
static uint32_t    _lastFrame = 0;
static TFT_eSprite* _spr     = nullptr;

void initEmotions(TFT_eSPI& tft, TFT_eSprite& spr) {
  _spr = &spr;
  tft.setRotation(1);            // landscape
  tft.fillScreen(COL_BG);
  spr.createSprite(SCR_W, SCR_H);
  spr.setSwapBytes(true);
}

void setEmotion(AdamEmotion e) { _emo = e; }
AdamEmotion getEmotion()       { return _emo; }

// ═══════════════════════════════════════════════════════════════
// DRAWING UTILITIES  (all draw into *_spr, never direct to TFT)
// ═══════════════════════════════════════════════════════════════

// Grey shade from brightness 0.0–1.0
static inline uint16_t _grey(float b) {
  uint8_t v = (uint8_t)(constrain(b, 0.0f, 1.0f) * 255);
  return _spr->color565(v, v, v);
}

// Draw arc of ellipse (rx, ry) centred at (cx, cy).
// Angle convention: 0°=right  90°=DOWN(screen)  270°=UP(screen)
// Draws startDeg → endDeg inclusive (no wrap-around, keep startDeg < endDeg).
static void _arc(int cx, int cy, int rx, int ry,
                 float startDeg, float endDeg,
                 uint16_t col, int thick = 4) {
  float step = max(0.8f, 50.0f / (float)max(rx, ry));
  float pr   = startDeg * DEG_TO_RAD;
  float px   = cx + rx * cosf(pr);
  float py   = cy + ry * sinf(pr);
  for (float a = startDeg + step; a <= endDeg + 0.01f; a += step) {
    float ar = a * DEG_TO_RAD;
    float nx = cx + rx * cosf(ar);
    float ny = cy + ry * sinf(ar);
    int h = thick / 2;
    for (int t = -h; t <= h; t++) {
      _spr->drawLine((int)px, (int)py + t, (int)nx, (int)ny + t, col);
      _spr->drawLine((int)px + t, (int)py, (int)nx + t, (int)ny, col);
    }
    px = nx; py = ny;
  }
}

// Thick line
static void _line(int x0,int y0,int x1,int y1,uint16_t col,int thick=4) {
  int h = thick / 2;
  for (int t = -h; t <= h; t++) {
    _spr->drawLine(x0, y0+t, x1, y1+t, col);
    _spr->drawLine(x0+t, y0, x1+t, y1, col);
  }
}

// Rounded-rectangle eye
static void _rectEye(int cx,int cy,int w,int h,uint16_t col) {
  _spr->fillRoundRect(cx-w/2, cy-h/2, w, h, h/2, col);
}

// Filled heart — two circles + filled triangle
static void _heartFill(int cx, int cy, int s, uint16_t col) {
  int r = s / 2;
  _spr->fillCircle(cx-r, cy-r/2, r, col);
  _spr->fillCircle(cx+r, cy-r/2, r, col);
  _spr->fillTriangle(cx-s, cy-r/2, cx+s, cy-r/2, cx, cy+s, col);
}

// Outline heart using parametric curve
static void _heartOutline(int cx,int cy,int s,uint16_t col,int thick=3) {
  float sc = s / 17.0f;
  float prevX=cx, prevY=cy+s;
  for (float t = 0.0f; t <= 2*PI+0.05f; t += 0.08f) {
    float x = cx + sc * 16.0f * powf(sinf(t), 3.0f);
    float y = cy - sc * (13.0f*cosf(t) - 5.0f*cosf(2*t)
                        - 2.0f*cosf(3*t) - cosf(4*t));
    int h = thick/2;
    for (int k=-h; k<=h; k++)
      _spr->drawLine((int)prevX,(int)prevY+k,(int)x,(int)y+k,col);
    prevX=x; prevY=y;
  }
}

// Sparkle cross at (sx,sy), arm length arm, brightness 0.0–1.0
static void _sparkle(int sx,int sy,int arm,float bright,uint16_t col=COL_WHITE) {
  if (bright <= 0.02f) return;
  uint16_t c  = _grey(bright);
  uint16_t cd = _grey(bright * 0.55f);
  int da = arm * 7 / 10;
  _spr->drawLine(sx,sy-arm, sx,sy+arm, c);
  _spr->drawLine(sx-arm,sy, sx+arm,sy, c);
  _spr->drawLine(sx-da,sy-da, sx+da,sy+da, cd);
  _spr->drawLine(sx+da,sy-da, sx-da,sy+da, cd);
}

// ═══════════════════════════════════════════════════════════════
// EMOTION DRAW FUNCTIONS
// All angles: 0°=right  90°=screen-down  180°=left  270°=screen-up
//   Happy ^ arch  = 180→360  (curves UP on screen)
//   Sad ∪  arch  =   0→180  (curves DOWN on screen)
// ═══════════════════════════════════════════════════════════════

// ── IDLE ─────────────────────────────────────────────────────
static void _drawIdle(uint32_t ms) {
  bool   blink  = (ms % 6000) < 130;
  float  breathe = sinf(ms * 0.001047f);        // 6 s
  int    by      = (int)(breathe * 1.5f);

  _spr->fillSprite(COL_BG);

  if (blink) {
    _spr->fillRect(EYE_L_X-21, EYE_Y-1+by, 42, 3, COL_WHITE);
    _spr->fillRect(EYE_R_X-21, EYE_Y-1+by, 42, 3, COL_WHITE);
  } else {
    _rectEye(EYE_L_X, EYE_Y+by, 42, 10, COL_WHITE);
    _rectEye(EYE_R_X, EYE_Y+by, 42, 10, COL_WHITE);
  }
  // Tiny dim mouth, gently pulsing
  float mScale = 1.0f + 0.04f * breathe;
  int   mw     = (int)(24 * mScale);
  _spr->fillRoundRect(FACE_CX-mw/2, MOUTH_Y+by, mw, 5, 2, COL_DIM);
}

// ── SPEAKING ─────────────────────────────────────────────────
static void _drawSpeaking(uint32_t ms) {
  bool blink = (ms % 3800) < 100;
  // Mouth cycles wide → mid → narrow, then repeats
  float ph = (ms % 420) / 420.0f;
  int   mw = (ph < 0.33f) ? 52 : (ph < 0.66f) ? 34 : 18;

  _spr->fillSprite(COL_BG);

  if (blink) {
    _spr->fillRect(EYE_L_X-21, EYE_Y-1, 42, 3, COL_WHITE);
    _spr->fillRect(EYE_R_X-21, EYE_Y-1, 42, 3, COL_WHITE);
  } else {
    _rectEye(EYE_L_X, EYE_Y, 42, 10, COL_WHITE);
    _rectEye(EYE_R_X, EYE_Y, 42, 10, COL_WHITE);
  }
  _spr->fillRoundRect(FACE_CX-mw/2, MOUTH_Y-3, mw, 8, 4, COL_WHITE);
}

// ── HAPPY ────────────────────────────────────────────────────
static void _drawHappy(uint32_t ms) {
  bool  blink  = (ms % 4000) < 110;
  float bounce = sinf(ms * 0.002856f) * 2.5f;  // 2.2 s

  _spr->fillSprite(COL_BG);

  // Three orbiting sparkles (white → warm yellow)
  auto _sp = [&](uint32_t offset, int sx, int sy, int arm) {
    uint32_t t = (ms + offset) % 2200;
    float bright = 0.0f;
    if (t > 300 && t < 1400)
      bright = (t < 850) ? (t-300)/550.0f : (1400-t)/550.0f;
    if (bright < 0.02f) return;
    // White base with yellow tint at peak
    uint8_t y = (uint8_t)(bright * 255);
    uint16_t col = _spr->color565(y, y, (uint8_t)(y * 0.6f));
    _sparkle(sx, sy, arm, bright, col);
  };
  _sp(0,    58,  55, 7);
  _sp(600,  262, 60, 7);
  _sp(1100, 280, 128, 5);

  if (blink) {
    _spr->fillRect(EYE_L_X-27, EYE_Y, 54, 4, COL_WHITE);
    _spr->fillRect(EYE_R_X-27, EYE_Y, 54, 4, COL_WHITE);
  } else {
    _arc(EYE_L_X, EYE_Y+5, 27, 22, 180, 360, COL_WHITE, 5);  // ^ arch
    _arc(EYE_R_X, EYE_Y+5, 27, 22, 180, 360, COL_WHITE, 5);
  }
  // Smile (downward arch)
  int my = MOUTH_Y + (int)bounce;
  _arc(FACE_CX, my-10, 28, 14, 0, 180, COL_WHITE, 4);
}

// ── SAD ──────────────────────────────────────────────────────
static void _drawSad(uint32_t ms) {
  float sag = sinf(ms * 0.001571f) * 4.0f;    // 4 s
  float tp1 = fmodf(ms * 0.001f + 0.2f, 1.5f);
  float tp2 = fmodf(ms * 0.001f + 1.0f, 1.5f);

  _spr->fillSprite(COL_BG);

  // Droopy ∪ eyes
  int ey = EYE_Y - 5 + (int)sag;
  _arc(EYE_L_X, ey, 27, 22, 0, 180, COL_WHITE, 5);
  _arc(EYE_R_X, ey, 27, 22, 0, 180, COL_WHITE, 5);

  // Frown (upward arch = 180→360)
  _arc(FACE_CX, MOUTH_Y+8+(int)sag, 26, 12, 180, 360, COL_WHITE, 4);

  // Blue-tinted tears
  auto _tear = [&](float tp, int ex) {
    if (tp < 0.55f) return;
    float d  = (tp - 0.55f) / 0.95f;
    int   ty = EYE_Y + 28 + (int)(d * 35);
    float a  = (tp < 1.2f) ? 1.0f : (1.5f-tp)/0.3f;
    uint8_t av = (uint8_t)(a * 210);
    uint16_t tc = _spr->color565(av/3, av/2, av);
    _spr->fillEllipse(ex, ty, 5, 7, tc);
    if (d > 0.08f)
      _spr->drawLine(ex-1, ty+6, ex-2, ty+7+(int)(d*16), tc);
  };
  _tear(tp1, EYE_L_X);
  _tear(tp2, EYE_R_X);
}

// ── ANGRY ────────────────────────────────────────────────────
static void _drawAngry(uint32_t ms) {
  int t  = ms % 100;
  int sx = (t<25)?-2:(t<75)?2:0;
  int sy = (t<50)?1:-1;

  _spr->fillSprite(COL_BG);

  // Diagonal slash eyes
  _line(82+sx,  72+sy, 128+sx, 96+sy, COL_WHITE, 5);
  _line(238+sx, 72+sy, 192+sx, 96+sy, COL_WHITE, 5);

  // Zigzag mouth
  const int mp[9][2] = {
    {108,142},{120,131},{133,142},{146,153},
    {159,142},{172,131},{185,142},{197,153},{208,142}
  };
  for (int i = 0; i < 8; i++)
    _line(mp[i][0]+sx,mp[i][1]+sy, mp[i+1][0]+sx,mp[i+1][1]+sy, COL_WHITE, 4);

  // Pulsing vein marks — hint of red tint at peak
  float vp = (ms % 600) / 600.0f;
  float vb = vp < 0.5f ? vp*2 : (1-vp)*2;
  uint16_t vc = _spr->color565((uint8_t)(200*vb+55), (uint8_t)(60*vb), (uint8_t)(60*vb));
  _spr->drawLine(88,58, 82,50, vc); _spr->drawLine(82,50, 86,42, vc); _spr->drawLine(86,42, 94,44, vc);
  _spr->drawLine(232,58, 238,50, vc); _spr->drawLine(238,50, 234,42, vc); _spr->drawLine(234,42, 226,44, vc);

  // Heat shimmer rising from eyes (orange tint)
  uint32_t ht = ms % 1200;
  if (ht < 960) {
    float hd = ht / 960.0f;
    float hb = (hd < 0.3f) ? hd/0.3f : (hd>0.7f) ? (1-hd)/0.3f : 1.0f;
    uint16_t hc = _spr->color565((uint8_t)(160*hb), (uint8_t)(80*hb), 0);
    int hy = EYE_Y - 4 - (int)(hd*26);
    _spr->drawLine(100, hy, 100, hy-7, hc);
    _spr->drawLine(108, hy-3, 107, hy-9, hc);
    _spr->drawLine(220, hy, 220, hy-7, hc);
    _spr->drawLine(212, hy-3, 213, hy-9, hc);
  }
}

// ── PANIC ────────────────────────────────────────────────────
static void _drawPanic(uint32_t ms) {
  int t  = ms % 70;
  int sx = (t<20)?-2:(t<40)?2:-1;
  int sy = (t<35)?1:-1;

  _spr->fillSprite(COL_BG);

  // Stress marks (flash, cyan tint)
  float sf = (ms%550)/550.0f;
  float sb = sf<0.5f ? sf*2 : (1-sf)*2;
  uint16_t sc = _spr->color565((uint8_t)(200*sb),(uint8_t)(230*sb),(uint8_t)(255*sb));
  _spr->drawLine(64,70, 57,79, sc); _spr->drawLine(57,79, 61,88, sc);
  _spr->drawLine(56,76, 52,78, sc); _spr->drawLine(52,78, 55,83, sc);
  _spr->drawLine(256,70, 263,79, sc); _spr->drawLine(263,79, 259,88, sc);

  // Wide oval eyes (double-stroke for weight)
  for (int d = 0; d < 2; d++) {
    _spr->drawEllipse(EYE_L_X+sx, EYE_Y+sy, 23-d, 29-d, COL_WHITE);
    _spr->drawEllipse(EYE_R_X+sx, EYE_Y+sy, 23-d, 29-d, COL_WHITE);
  }

  // Darting pupils
  float pp = (ms%300)/300.0f;
  int px2 = (pp<0.45f)?0:(pp<0.55f)?4:(pp<0.65f)?-4:0;
  int py2 = (pp<0.45f)?0:(pp<0.55f)?-2:(pp<0.65f)?2:0;
  _spr->fillCircle(EYE_L_X+sx+px2, EYE_Y+sy+py2, 5, COL_WHITE);
  _spr->fillCircle(EYE_R_X+sx+px2, EYE_Y+sy+py2, 5, COL_WHITE);

  // Panic zigzag mouth
  const int mz[9][2] = {
    {128,143},{138,135},{148,143},{158,151},
    {168,143},{178,135},{188,143},{196,149},{200,145}
  };
  for (int i = 0; i < 8; i++)
    _line(mz[i][0]+sx,mz[i][1]+sy, mz[i+1][0]+sx,mz[i+1][1]+sy, COL_WHITE, 4);

  // Blue sweat drops
  auto _sweat = [&](float phase, int ex) {
    float sw = fmodf(phase, 2.2f);
    if (sw < 0.35f) return;
    float d  = (sw - 0.35f) / 1.85f;
    int   swy = EYE_Y - 28 + (int)(d * 44);
    float a  = (sw<0.45f)?(sw-0.35f)/0.1f:(sw>2.0f)?(2.2f-sw)/0.2f:1.0f;
    uint8_t av = (uint8_t)(a * 200);
    _spr->fillEllipse(ex, swy, 4, 6, _spr->color565(av/4, av/3, av));
  };
  _sweat(fmodf(ms*0.001f+0.2f, 100.0f), EYE_L_X);
  _sweat(fmodf(ms*0.001f+1.3f, 100.0f), EYE_R_X);
}

// ── SURPRISED ────────────────────────────────────────────────
static void _drawSurprised(uint32_t ms) {
  float phase = fmodf(ms * 0.001f, 2.8f);
  float eyeS  = (phase < 0.18f) ? phase/0.18f : 1.0f;
  int   erx   = (int)(23 * eyeS);
  int   ery   = (int)(31 * eyeS);

  _spr->fillSprite(COL_BG);

  // Shock lines radiating outward
  if (phase > 0.25f && phase < 1.9f) {
    float lb = min(1.0f, (phase-0.25f)/0.2f) * (phase>1.7f?(1.9f-phase)/0.2f:1.0f);
    uint16_t lc = _grey(lb * 0.55f);
    _spr->drawLine(68,52, 48,38, lc);  _spr->drawLine(60,65, 36,60, lc);  _spr->drawLine(72,78, 50,82, lc);
    _spr->drawLine(252,52,272,38, lc); _spr->drawLine(260,65,284,60, lc); _spr->drawLine(248,78,270,82, lc);
  }

  // Wide oval eyes pop in
  if (erx > 2 && ery > 2) {
    for (int d = 0; d < 2; d++) {
      _spr->drawEllipse(EYE_L_X, EYE_Y, erx-d, ery-d, COL_WHITE);
      _spr->drawEllipse(EYE_R_X, EYE_Y, erx-d, ery-d, COL_WHITE);
    }
    // Eye glint flash at pop moment
    if (phase < 0.22f) {
      float gb = (0.22f-phase)/0.22f * 0.35f;
      uint16_t gc = _grey(gb);
      _spr->fillCircle(EYE_L_X+8, EYE_Y-10, 6, gc);
      _spr->fillCircle(EYE_R_X+8, EYE_Y-10, 6, gc);
    }
  }

  // O-shaped mouth scales in
  if (phase > 0.18f) {
    float ms2 = min(1.0f, (phase-0.18f)/0.2f);
    int   mry = max(2, (int)(7*ms2));
    _spr->drawEllipse(FACE_CX, MOUTH_Y-3, 8,   mry,   COL_WHITE);
    _spr->drawEllipse(FACE_CX, MOUTH_Y-3, 7, mry-1,   COL_WHITE);
  }
}

// ── SHY ──────────────────────────────────────────────────────
static void _drawShy(uint32_t ms) {
  bool  blink   = (ms % 5000) < 110;
  float blushI  = 0.35f + 0.3f*(sinf(ms*0.002513f)*0.5f+0.5f);
  float bob     = sinf(ms*0.001047f) * 5.0f;  // gentle dip
  int   by      = (int)bob;

  _spr->fillSprite(COL_BG);

  // Soft pink blush ellipses (layered for glow effect)
  for (int r = 22; r > 0; r -= 4) {
    float ratio = (float)r / 22.0f;
    uint8_t ri = (uint8_t)(blushI * ratio * 200);
    uint16_t bc = _spr->color565(ri, ri/5, ri/4);
    _spr->drawEllipse(82,  116+by, r, r/2, bc);
    _spr->drawEllipse(238, 116+by, r, r/2, bc);
  }

  if (blink) {
    _spr->fillRect(EYE_L_X-27, EYE_Y+by, 54, 4, COL_WHITE);
    _spr->fillRect(EYE_R_X-27, EYE_Y+by, 54, 4, COL_WHITE);
  } else {
    _arc(EYE_L_X, EYE_Y+5+by, 27, 22, 180, 360, COL_WHITE, 5); // ^ happy eyes
    _arc(EYE_R_X, EYE_Y+5+by, 27, 22, 180, 360, COL_WHITE, 5);
  }
  // Tiny smile
  _arc(FACE_CX, MOUTH_Y-2+by, 10, 8, 0, 180, COL_WHITE, 4);

  // Floating heart (top-right, pink-tinted)
  float hf = fmodf(ms*0.001f + 1.5f, 3.0f);
  if (hf > 0.1f && hf < 2.6f) {
    float d  = hf / 2.6f;
    int   hx = 272 + (int)(8*d);
    int   hy = 78  - (int)(28*d);
    float a  = (hf<0.3f)?hf/0.3f:(hf>2.2f)?(2.6f-hf)/0.4f:1.0f;
    uint8_t av=(uint8_t)(a*160);
    _heartOutline(hx, hy, 7, _spr->color565(av, av/5, av/4), 2);
  }
}

// ── SLEEP ────────────────────────────────────────────────────
static void _drawSleep(uint32_t ms) {
  float breathe = sinf(ms * 0.001257f);    // 5 s
  int   by      = (int)(breathe * 2);

  _spr->fillSprite(COL_BG);

  // Closed ∪ eyes
  _arc(EYE_L_X, EYE_Y-5+by, 27, 22, 0, 180, COL_WHITE, 5);
  _arc(EYE_R_X, EYE_Y-5+by, 27, 22, 0, 180, COL_WHITE, 5);

  // ZZZ cascade — each char rises and fades, staggered
  const int   ZSZ[3] = {3, 2, 1};
  const uint32_t ZOFF[3] = {0, 500, 1000};
  for (int i = 0; i < 3; i++) {
    float zp = fmodf((ms + ZOFF[i]) * 0.001f, 4.0f);
    if (zp < 0.1f || zp > 3.8f) continue;
    float za = (zp<0.25f)?zp/0.25f:(zp>3.5f)?(4.0f-zp)/0.5f:1.0f;
    int   zy  = 76 - i*18 - (int)(zp * 9.5f);
    int   zx  = 250 + (int)(zp * 3.5f);
    uint16_t zc = _grey(za * 0.85f);
    _spr->drawChar(zx, zy, 'z', zc, COL_BG, ZSZ[i]);
  }
}

// ── THINKING / SEARCH ────────────────────────────────────────
static void _drawThinking(uint32_t ms) {
  float scan = sinf(ms * 0.003927f) * 6.0f;  // 1.6 s, ±6 px
  int   so   = (int)scan;

  _spr->fillSprite(COL_BG);

  // Wave/zigzag eyes scanning in opposite directions
  const int lw[7][2] = {{70,88},{80,78},{92,88},{104,98},{116,88},{128,78},{138,88}};
  const int rw[7][2] = {{180,88},{192,78},{204,88},{216,98},{228,88},{240,78},{252,88}};
  for (int i = 0; i < 6; i++) {
    _line(lw[i][0]+so, lw[i][1], lw[i+1][0]+so, lw[i+1][1], COL_WHITE, 4);
    _line(rw[i][0]-so, rw[i][1], rw[i+1][0]-so, rw[i+1][1], COL_WHITE, 4);
  }

  // Flat mouth, subtle pulse
  float mp = sinf(ms * 0.003142f) * 0.5f + 0.5f;
  int   mw = 50 + (int)(mp * 8);
  _spr->fillRoundRect(FACE_CX-mw/2, MOUTH_Y-2, mw, 4, 2, _grey(0.6f+mp*0.4f));

  // Blinking cursor (right of mouth)
  if ((ms / 900) % 2 == 0)
    _spr->fillRect(FACE_CX+mw/2+3, MOUTH_Y-6, 3, 12, COL_WHITE);
}

// ── RECONNECTING ─────────────────────────────────────────────
static void _drawReconnecting(uint32_t ms) {
  float angle = (ms % 1200) / 1200.0f * 2*PI;
  float ringP = (ms % 1200) / 1200.0f;

  _spr->fillSprite(COL_BG);

  // Expanding pulse rings (cyan-tinted)
  float rs = 1.0f + ringP * 1.3f;
  float rb = max(0.0f, 0.4f - ringP * 0.4f);
  if (rb > 0.04f) {
    uint16_t rc = _spr->color565(
      (uint8_t)(rb*160), (uint8_t)(rb*230), (uint8_t)(rb*255));
    int rr = (int)(22 * rs);
    _spr->drawEllipse(EYE_L_X, EYE_Y, rr, rr, rc);
    _spr->drawEllipse(EYE_R_X, EYE_Y, rr, rr, rc);
  }

  // Rotating star eyes (CW left, CCW right)
  const float arms[4] = {0, PI/2, PI, 3*PI/2};
  for (int a = 0; a < 4; a++) {
    int al = 22, thick = (a % 2 == 0) ? 4 : 2;
    float al_  = angle + arms[a];
    float ar_  = -angle + arms[a];
    _line(EYE_L_X+(int)(cosf(al_)*al), EYE_Y+(int)(sinf(al_)*al),
          EYE_L_X-(int)(cosf(al_)*al), EYE_Y-(int)(sinf(al_)*al), COL_WHITE, thick);
    _line(EYE_R_X+(int)(cosf(ar_)*al), EYE_Y+(int)(sinf(ar_)*al),
          EYE_R_X-(int)(cosf(ar_)*al), EYE_Y-(int)(sinf(ar_)*al), COL_WHITE, thick);
  }

  // Pulsing flat mouth
  float mp = sinf(ms * 0.004488f) * 0.5f + 0.5f;
  _spr->fillRoundRect(FACE_CX-30, MOUTH_Y-2, 60, 4, 2, _grey(0.5f+mp*0.5f));

  // Loading dots (white active, dim inactive)
  int dp = (ms / 470) % 3;
  for (int d = 0; d < 3; d++)
    _spr->fillCircle(FACE_CX-10+d*10, MOUTH_Y+22, 4, _grey(d==dp ? 1.0f : 0.15f));
}

// ── LOVE ─────────────────────────────────────────────────────
static void _drawLove(uint32_t ms) {
  // Double-beat heartbeat envelope
  float t = fmodf(ms * 0.000909f, 1.0f);  // 1.1 s
  float sc = 1.0f;
  if      (t < 0.14f) sc = 1.0f + t/0.14f * 0.22f;
  else if (t < 0.28f) sc = 1.22f - (t-0.14f)/0.14f * 0.22f;
  else if (t < 0.42f) sc = 1.0f  + (t-0.28f)/0.14f * 0.12f;
  else if (t < 0.56f) sc = 1.12f - (t-0.42f)/0.14f * 0.12f;

  _spr->fillSprite(COL_BG);

  // Pulse rings (pink tint)
  float rp = fmodf(ms * 0.000909f, 1.0f);
  float rb = max(0.0f, 0.45f - rp * 0.45f);
  if (rb > 0.04f) {
    uint16_t rc = _spr->color565((uint8_t)(rb*255), (uint8_t)(rb*80), (uint8_t)(rb*100));
    float rs = 1.0f + rp * 1.5f;
    int rr = (int)(24 * rs);
    _spr->drawEllipse(EYE_L_X, EYE_Y, rr, rr, rc);
    _spr->drawEllipse(EYE_R_X, EYE_Y, rr, rr, rc);
  }

  // Heart eyes (white, heartbeat-scaled)
  _heartFill(EYE_L_X, EYE_Y, (int)(28*sc), COL_WHITE);
  _heartFill(EYE_R_X, EYE_Y, (int)(28*sc), COL_WHITE);

  // Smile
  float sb = sinf(ms * 0.003142f) * 2.0f;
  _arc(FACE_CX, MOUTH_Y-8+(int)sb, 30, 12, 0, 180, COL_WHITE, 4);

  // Floating outline hearts (pink-tinted, 3 positions)
  auto _fh = [&](float phase, int sx2, int sy) {
    float p = fmodf(phase, 2.5f);
    if (p < 0.1f || p > 2.4f) return;
    float d  = p / 2.5f;
    int   hx = sx2 + (int)(10*d);
    int   hy = sy  - (int)(50*d);
    float a  = (p<0.4f)?p/0.4f:(p>2.0f)?(2.5f-p)/0.5f:1.0f;
    int   hs = max(3, (int)(9*(1.0f-d*0.4f)));
    uint8_t av = (uint8_t)(a * 180);
    _heartOutline(hx, hy, hs, _spr->color565(av, av/4, av/4), 2);
  };
  _fh(fmodf(ms*0.001f+0.5f,100.0f), 258, 140);
  _fh(fmodf(ms*0.001f+1.2f,100.0f), 50,  155);
  _fh(fmodf(ms*0.001f+1.8f,100.0f), 278, 102);
}

// ── CONFUSED ─────────────────────────────────────────────────
static void _drawConfused(uint32_t ms) {
  // Head-tilt approximated as horizontal pixel offset
  float tilt = sinf(ms * 0.001571f);     // 4 s: +1°…-5°
  int   to   = (int)(tilt * 5);

  _spr->fillSprite(COL_BG);

  // Archimedean spiral eyes, counter-rotating
  auto _spiral = [&](int cx, int cy, bool cw) {
    float base = (ms % 2400) / 2400.0f * 2*PI * (cw ? 1 : -1);
    float prevX = cx, prevY = cy;
    for (float a = 0.1f; a < 7.3f; a += 0.1f) {
      float r   = a / 7.3f * 22.0f;
      float ang = base + a * (cw ? 1 : -1);
      float nx  = cx + r * cosf(ang) + to;
      float ny  = cy + r * sinf(ang);
      if (a > 0.2f) _spr->drawLine((int)prevX,(int)prevY,(int)nx,(int)ny,COL_WHITE);
      prevX = nx; prevY = ny;
    }
    _spr->fillCircle(cx+to, cy, 3, COL_WHITE);
  };
  _spiral(EYE_L_X, EYE_Y, true);
  _spiral(EYE_R_X, EYE_Y, false);

  // Flat mouth with pulse
  float mp = sinf(ms * 0.003142f) * 0.5f + 0.5f;
  int   mw = (int)(54 * (1.0f - 0.18f*mp));
  _spr->fillRoundRect(FACE_CX-mw/2+to, MOUTH_Y-2, mw, 5, 2, COL_WHITE);

  // Bouncing question mark
  int qy = 46 - (int)(sinf(ms * 0.002513f) * 10.0f);
  _spr->setTextSize(4);
  _spr->drawChar(238, qy, '?', COL_WHITE, COL_BG, 4);
}

// ── RIZZ ─────────────────────────────────────────────────────
static void _drawRizz(uint32_t ms) {
  float phase   = fmodf(ms * 0.000357f, 1.0f);  // 2.8 s
  bool  winking = (phase > 0.10f && phase < 0.45f);
  int   leanO   = (int)(sinf(ms * 0.002244f) * 3.0f);

  _spr->fillSprite(COL_BG);

  // Left eye: thick when open, thin line when winking
  int lh = winking ? 3 : 7;
  _spr->fillRoundRect(EYE_L_X+leanO-25, EYE_Y-lh/2, 50, lh, lh/2, COL_WHITE);

  // Right eye: confident ^ arc, slight squint during wink
  int rh = winking ? 13 : 22;
  _arc(EYE_R_X+leanO, EYE_Y+5, 27, rh, 180, 360, COL_WHITE, 5);

  // Smirk — flat left, right curl
  _line(142+leanO, 140, 163+leanO, 136, COL_WHITE, 4);
  // Right curl: two segments approximating a bezier
  _line(163+leanO, 136, 174+leanO, 130, COL_WHITE, 4);
  _line(174+leanO, 130, 183+leanO, 136, COL_WHITE, 4);

  // Post-wink gleam sparkle (white-to-yellow)
  if (phase > 0.50f && phase < 0.80f) {
    float gp = (phase-0.50f)/0.30f;
    float gs = sinf(gp * PI);
    uint8_t ga = (uint8_t)(gs * 240);
    uint16_t gc = _spr->color565(ga, ga, (uint8_t)(ga*0.5f)); // warm white
    _sparkle(260, 58, 9, gs, gc);
  }
}

// ═══════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════

// Call in loop() — rate-limited to ~30 fps, flicker-free via sprite.
void updateEmotion() {
  if (!_spr) return;
  uint32_t now = millis();
  if (now - _lastFrame < 33) return;
  _lastFrame = now;

  switch (_emo) {
    case EMO_IDLE:         _drawIdle(now);         break;
    case EMO_SPEAKING:     _drawSpeaking(now);     break;
    case EMO_HAPPY:        _drawHappy(now);        break;
    case EMO_SAD:          _drawSad(now);          break;
    case EMO_ANGRY:        _drawAngry(now);        break;
    case EMO_PANIC:        _drawPanic(now);        break;
    case EMO_SURPRISED:    _drawSurprised(now);    break;
    case EMO_SHY:          _drawShy(now);          break;
    case EMO_SLEEP:        _drawSleep(now);        break;
    case EMO_THINKING:     _drawThinking(now);     break;
    case EMO_RECONNECTING: _drawReconnecting(now); break;
    case EMO_LOVE:         _drawLove(now);         break;
    case EMO_CONFUSED:     _drawConfused(now);     break;
    case EMO_RIZZ:         _drawRizz(now);         break;
    default:               _drawIdle(now);         break;
  }
  _spr->pushSprite(0, 0);   // atomic blit → zero flicker
}