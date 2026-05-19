/*
 * picopixel_lvgl_ui.ino
 * ADAM Face UI on ESP32 + ILI9341 via LVGL v8
 *
 * Wiring (ESP32 -> ILI9341):
 * CS=5, DC=2, RST=4, MOSI=23, SCLK=18, MISO=19
 *
 * Required Arduino libraries:
 * - LVGL (v8.x)
 * - Adafruit GFX Library
 * - Adafruit ILI9341
 */

#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>

#include <lvgl.h>

// UI header stub - define minimal UI structure
// extern "C" {
// #include "ui.h"
// }

// TFT pins (from your working test)
#define TFT_CS    5
#define TFT_RST   4
#define TFT_DC    2
#define TFT_MOSI 23
#define TFT_SCLK 18
#define TFT_MISO 19

// ILI9341 resolution
static const uint16_t TFT_HOR_RES = 320;
static const uint16_t TFT_VER_RES = 240;

Adafruit_ILI9341 tft(TFT_CS, TFT_DC, TFT_RST);

// LVGL draw buffer. 40 lines gives a good balance between RAM and speed.
static lv_disp_draw_buf_t draw_buf;
static lv_color_t buf_1[TFT_HOR_RES * 40];

static void lvgl_flush_cb(lv_disp_drv_t *disp, const lv_area_t *area, lv_color_t *color_p) {
  (void)disp;

  const int32_t w = area->x2 - area->x1 + 1;
  const int32_t h = area->y2 - area->y1 + 1;

#if LV_COLOR_DEPTH == 16
  tft.startWrite();
  tft.setAddrWindow(area->x1, area->y1, w, h);
  tft.writePixels((uint16_t *)color_p, (uint32_t)(w * h), true, false);
  tft.endWrite();
#else
  // Fallback conversion for non-565 LVGL configs.
  static uint16_t line[TFT_HOR_RES];
  tft.startWrite();
  for (int32_t y = 0; y < h; y++) {
    for (int32_t x = 0; x < w; x++) {
      lv_color_t c = color_p[(y * w) + x];
      line[x] = ((uint16_t)c.ch.red & 0xF8) << 8 |
                ((uint16_t)c.ch.green & 0xFC) << 3 |
                ((uint16_t)c.ch.blue >> 3);
    }
    tft.setAddrWindow(area->x1, area->y1 + y, w, 1);
    tft.writePixels(line, (uint32_t)w, true, false);
  }
  tft.endWrite();
#endif

  lv_disp_flush_ready(disp);
}

void setup() {
  Serial.begin(115200);

  SPI.begin(TFT_SCLK, TFT_MISO, TFT_MOSI);
  tft.begin();
  tft.setRotation(1);
  tft.fillScreen(ILI9341_BLACK);

  lv_init();

  lv_disp_draw_buf_init(&draw_buf, buf_1, NULL, TFT_HOR_RES * 40);

  static lv_disp_drv_t disp_drv;
  lv_disp_drv_init(&disp_drv);
  disp_drv.hor_res = TFT_HOR_RES;
  disp_drv.ver_res = TFT_VER_RES;
  disp_drv.flush_cb = lvgl_flush_cb;
  disp_drv.draw_buf = &draw_buf;
  lv_disp_drv_register(&disp_drv);

  // UI initialization skipped - no ui.h generated yet
  // ui_init() would be called here once UI is designed

  Serial.println("LVGL initialized on ESP32.");
}

void loop() {
  static uint32_t lastMs = 0;
  uint32_t now = millis();
  uint32_t delta = now - lastMs;
  lastMs = now;

  if (delta > 0) {
    lv_tick_inc(delta);
  }

  lv_timer_handler();
  // ui_tick() would be called here for UI animations
  delay(5);
}
