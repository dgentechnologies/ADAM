// ═════════════════════════════════════════════════════════════════════════════
// User_Setup_ADAM_S3.h — TFT_eSPI configuration for the ADAM Head Motherboard
// Dgen Technologies Pvt. Ltd.  |  October 2026
//
// Target: ESP32-S3-WROOM-1-N16R8 + 2.4" ILI9341 320x240 on a 20-pin FPC
// Pin map source of truth: Hardware_Docs/PCB_Docs/ADAM_Head_PinMap_v2_N16R8.md
//
// ─── INSTALLATION (required — TFT_eSPI cannot take pins from the sketch) ─────
// TFT_eSPI reads its pin configuration at COMPILE time from a header inside the
// library folder. There is no constructor argument for pins. So:
//
//   1. Copy this file into the TFT_eSPI library folder:
//        Arduino/libraries/TFT_eSPI/User_Setups/User_Setup_ADAM_S3.h
//
//   2. Edit  Arduino/libraries/TFT_eSPI/User_Setup_Select.h
//      - comment out the default:
//            //#include <User_Setup.h>
//      - add:
//            #include <User_Setups/User_Setup_ADAM_S3.h>
//
//   3. Rebuild. The sketch prints the pins it actually got at boot and will
//      refuse to compile if MOSI/SCK are still 35/36.
//
// Keeping this file in the repo (rather than only in the library folder) is
// deliberate: the library folder is not version-controlled, so without a copy
// here the board's display config would exist only on one developer's machine.
// ═════════════════════════════════════════════════════════════════════════════

#define USER_SETUP_ID 9001

// ─── Driver ─────────────────────────────────────────────────────────────────
#define ILI9341_DRIVER

// Panel is 240x320 in its native portrait orientation. The sketch calls
// setRotation(1) to use it as 320x240 landscape, which is what the face
// renderer in adam_emotions.h is drawn for.
#define TFT_WIDTH  240
#define TFT_HEIGHT 320

// ─── Pins ───────────────────────────────────────────────────────────────────
// GPIO21 (MOSI) and GPIO38 (SCK) are the OCTAL-PSRAM CORRECTION.
//
// The PCB reference document assigns the display bus to GPIO35/36. On an N16R8
// (or N8R8) module those two pins are wired internally to the octal PSRAM die
// and are NOT available externally — Espressif's ESP32-S3-WROOM-1 datasheet
// states GPIO33-37 are reserved on octal-PSRAM modules. Driving the display
// there does not fail cleanly; it corrupts PSRAM, which shows up later as
// camera framebuffer garbage and random crashes.
//
// Do not change these back without also changing the module to a quad-PSRAM
// part (e.g. N8R2), which costs 6 MB of PSRAM and half its bandwidth.
#define TFT_MOSI 21
#define TFT_SCLK 38
#define TFT_DC   40
#define TFT_CS   41
#define TFT_RST  47

// MISO is not connected on this board — the 20-pin FPC leaves pin 6 open and
// the panel is write-only. -1 tells TFT_eSPI not to configure a read line.
#define TFT_MISO -1

// Backlight is hardwired to +3V3 through a 10 ohm resistor (J_TFT pins 12/13).
// There is no GPIO for it, so TFT_BL is deliberately left undefined: the
// backlight is always on and brightness is not software-controllable.

// ─── Fonts ──────────────────────────────────────────────────────────────────
// The face is drawn with geometry, not text, so only a small font is kept for
// diagnostics. Dropping the rest saves flash and, more usefully, build time.
#define LOAD_GLCD
#define LOAD_FONT2
#define LOAD_GFXFF
#define SMOOTH_FONT

// ─── SPI ────────────────────────────────────────────────────────────────────
// 40 MHz is the reliable ceiling for the ILI9341 over a 20-pin FPC of this
// length. 80 MHz works on short rigid traces but shows tearing/noise through a
// flex cable that also runs beside the camera bus.
#define SPI_FREQUENCY       40000000
#define SPI_READ_FREQUENCY  20000000

// DMA is what makes the sprite double-buffer cheap enough to animate the face
// at a usable rate while the camera is also running.
#define ESP32_DMA
