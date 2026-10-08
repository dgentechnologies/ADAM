# ADAM Head Board — Corrected Pin Map v2 (ESP32-S3-WROOM-1-**N16R8**)

**Company:** Dgen Technologies Pvt. Ltd. — Kolkata, India
**Board:** Head Motherboard (unified ESP32-S3: camera + display + touch + tilt)
**Module:** **ESP32-S3-WROOM-1-N16R8** — 16 MB quad flash, **8 MB octal PSRAM**
**Revision:** v2.0 — octal-PSRAM pin conflict corrected
**Supersedes:** the head-board pin tables in `ADAM_PCB_Reference_v1.pdf` §5.3 and
`ADAM_Head_Motherboard_BOM_ESP32S3.md` §4
**Date:** 2026-10-07

---

## 1. Why this document exists

Both existing pin tables assign the display SPI bus to **GPIO35 and GPIO36**.
On a module with **octal** SPI PSRAM — which `N8R8` and `N16R8` both are —
those pins are not available. They are wired inside the module to the PSRAM
die.

> Espressif, *ESP32-S3-WROOM-1 datasheet*, pin table note: GPIO33–GPIO37 are
> used for the internal connection between the ESP32-S3 and the SPI
> flash/PSRAM on modules that embed **octal** SPI flash or octal SPI PSRAM, and
> are **not available for external use**.

Routing `TFT_MOSI`/`TFT_SCK` to GPIO35/36 on an N16R8 therefore puts the
display on the PSRAM's own data lines. The practical result is not a clean
failure but an ugly one: PSRAM corruption, camera framebuffer garbage, random
`Guru Meditation` crashes, or a display that works only until PSRAM is first
touched. It is the kind of fault that gets blamed on the panel for a week.

**This board needs its PSRAM.** 8 MB is what makes an OV2640 JPEG framebuffer
and a 320×240 TFT sprite double-buffer (~150 KB) coexist. Dropping to quad
PSRAM to free GPIO33–37 would also halve PSRAM bandwidth, which is exactly the
wrong trade on a board whose job is pushing pixels.

So the fix is to move two display signals. **Two traces, no BOM change.**

---

## 2. What changed from the v1 tables

| Signal | v1 pin | **v2 pin** | Why |
|---|---|---|---|
| `TFT_MOSI` | GPIO35 | **GPIO21** | GPIO35 is an octal-PSRAM data line — unavailable on N16R8 |
| `TFT_SCK` | GPIO36 | **GPIO38** | GPIO36 is an octal-PSRAM data line — unavailable on N16R8 |

Everything else in `ADAM_PCB_Reference_v1.pdf` §5.3 is kept, **including** that
document's own corrections over the older BOM `.md`:

| Signal | Pin | Note |
|---|---|---|
| `TFT_DC` | GPIO40 | already moved off GPIO45 (VDD_SPI strap) by the PDF — correct |
| `TFT_CS` | GPIO41 | already moved off GPIO46 (boot-log strap) by the PDF — correct |
| `TOUCH3` | GPIO42 | already moved off GPIO3 (JTAG-source strap) by the PDF — correct |

**Note the BOM `.md` and the PDF disagree** on five nets (`TOUCH3`, `CAM_SIOD`,
`CAM_SIOC`, `TFT_DC`, `TFT_CS`). **The PDF wins** — it is the later consolidated
review. This document follows the PDF, plus the two changes above.

---

## 3. Authoritative pin map (v2)

### 3.1 Camera — OV2640, 8-bit DVP

| Signal | GPIO | J_CAM pin | Note |
|---|---|---|---|
| `CAM_XCLK` | 15 | 12 | 20 MHz master clock out |
| `CAM_PCLK` | 13 | 18 | pixel clock in |
| `CAM_VSYNC` | 6 | 14 | |
| `CAM_HREF` | 7 | 16 | |
| `CAM_SIOD` | 4 | 10 | SCCB data, 2.2 kΩ pull-up |
| `CAM_SIOC` | 5 | 8 | SCCB clock, 2.2 kΩ pull-up |
| `CAM_D0` | 11 | 17 | |
| `CAM_D1` | 9 | 15 | |
| `CAM_D2` | 8 | 13 | |
| `CAM_D3` | 10 | 11 | |
| `CAM_D4` | 12 | 9 | |
| `CAM_D5` | 18 | 7 | |
| `CAM_D6` | 17 | 5 | |
| `CAM_D7` | 16 | 3 | |
| `PWDN` | — | 23 | **hardwired** 10 kΩ pulldown — firmware uses `-1` |
| `RESET` | — | 6 | **hardwired** 10 kΩ + 100 nF RC delay — firmware uses `-1` |

`PWDN` and `RESET` being hardwired is deliberate and must be reflected in
firmware as `pin_pwdn = -1`, `pin_reset = -1`. Declaring a GPIO the board does
not route is a silent init failure.

### 3.2 Display — ILI9341 320×240, SPI

| Signal | GPIO | J_TFT pin | Note |
|---|---|---|---|
| `TFT_MOSI` | **21** | 5 | **CHANGED from 35** |
| `TFT_SCK` | **38** | 8 | **CHANGED from 36** |
| `TFT_DC` | 40 | 7 | |
| `TFT_CS` | 41 | 9 | |
| `TFT_RESET` | 47 | 10 | |
| backlight | — | 12, 13 | hardwired 3V3 via 10 Ω — no GPIO, always on |
| MISO | — | 6 | not connected — display is write-only |

### 3.3 Touch — 4 channels

| Signal | GPIO | Position | Native touch channel |
|---|---|---|---|
| `TOUCH1` | 1 | left cheek | TOUCH1 |
| `TOUCH2` | 2 | right cheek | TOUCH2 |
| `TOUCH3` | **42** | petting-A | *none — GPIO42 has no touch peripheral* |
| `TOUCH4` | 14 | petting-B | TOUCH14 |

⚠️ **Consequence of the PDF's GPIO3 → GPIO42 move:** ESP32-S3 native capacitive
touch exists only on GPIO1–GPIO14. **GPIO42 cannot do native touch.** The
"populate copper pads *or* TTP223 modules" choice in the BOM therefore no
longer applies to channel 3 — with this pin map, **TOUCH3 must be an external
TTP223 module** (a digital input). If native-pad mode matters for all four
channels, move TOUCH3 to **GPIO39** instead… which also has no touch
peripheral. There is no free GPIO ≤14 left, so native-touch-on-all-four is not
achievable on this pin map. Firmware treats all four as digital TTP223 inputs.

### 3.4 Actuation, link, programming

| Signal | GPIO | Note |
|---|---|---|
| `SERVO_PWM` | 48 | LEDC, MG90S tilt, 50 Hz |
| `HEAD_TX` | 43 | to Body Pi RX — 921600 baud |
| `HEAD_RX` | 44 | from Body Pi TX — 921600 baud |
| `USB_DM` | 19 | native USB — pogo pad, flashing + debug CDC |
| `USB_DP` | 20 | native USB — pogo pad |
| `BOOT` | 0 | strap, 10 kΩ pull-up, pogo pad only |

---

## 4. Reserved — do not route to anything

| GPIO | Reserved for | Consequence if used |
|---|---|---|
| **26–32** | internal quad SPI **flash** | board will not boot |
| **33–37** | internal **octal PSRAM** (N16R8) | PSRAM corruption / random crashes |
| **0** | boot-mode strap | cannot enter download mode reliably |
| **3** | JTAG-source strap | left floating on purpose — leave open |
| **45** | VDD_SPI flash-voltage strap | wrong flash voltage at reset |
| **46** | ROM boot-log strap | boot behaviour changes |
| 19, 20 | native USB | lose flashing/debug |

**Spare and genuinely free: GPIO39 only.**

ESP32-S3 has no GPIO22–25 — those numbers do not exist on this part. Any
schematic net claiming them is an error.

### Layout DRC checklist

1. GPIO33–37 pads: **no copper, no track, no pour.** Add this as an explicit
   DRC rule — it is the failure this revision exists to prevent.
2. GPIO3, 45, 46 pads: unconnected, floating. Internal pull-ups govern boot.
3. `CAM_D0`–`D7` + `PCLK` over an unbroken ground plane, length-matched ±2.5 mm.
4. `C_SERVO` (220 µF) and the flyback diode immediately at `J_SERVO` pins.
5. Neck cable braided shield grounded at the **body** end only.

---

## 5. UART0 and the boot-log question

`HEAD_TX`/`HEAD_RX` sit on GPIO43/44, which are the ESP32-S3's default
`U0TXD`/`U0RXD`. Two consequences the Pi side must live with:

1. **The ROM bootloader prints on GPIO43 at every reset** (115200 8N1, before
   any application code runs). The Pi will receive a burst of junk on the
   protocol line each time the head resets. Because GPIO46 is left floating,
   ROM logging is enabled.
2. Therefore firmware must **not** also log debug text there.

The firmware handles this by:

- driving the Pi protocol on a **UART1** peripheral routed to GPIO43/44 via the
  GPIO matrix, at 921600 — not on UART0;
- sending all human-readable debug to **USB CDC** on the pogo pads instead;
- emitting a `HELLO:` sync line after boot so the Pi can discard everything
  before it.

The Pi's `esp32_link.py` already discards unparseable bytes, so the ROM burst is
harmless — but it must stay that way. Do not add a parser there that treats
unexpected bytes as fatal.

---

## 6. Firmware constants

These are the values in `MP-MC codes/esp32_s3_head/esp32_s3_head.ino`. If the
schematic changes, change it there — it is the single place pins are declared.

```c
// Camera (OV2640 DVP)
#define CAM_XCLK 15
#define CAM_PCLK 13
#define CAM_VSYNC 6
#define CAM_HREF  7
#define CAM_SIOD  4
#define CAM_SIOC  5
#define CAM_D0 11
#define CAM_D1  9
#define CAM_D2  8
#define CAM_D3 10
#define CAM_D4 12
#define CAM_D5 18
#define CAM_D6 17
#define CAM_D7 16
#define CAM_PWDN  -1   // hardwired pulldown
#define CAM_RESET -1   // hardwired RC delay

// Display (ILI9341) — 21/38 are the octal-PSRAM corrections
#define TFT_MOSI_PIN 21
#define TFT_SCK_PIN  38
#define TFT_DC_PIN   40
#define TFT_CS_PIN   41
#define TFT_RST_PIN  47

// Touch (all four are external TTP223 digital inputs)
#define TOUCH1_PIN 1
#define TOUCH2_PIN 2
#define TOUCH3_PIN 42
#define TOUCH4_PIN 14

// Tilt servo + neck link
#define SERVO_PIN  48
#define NECK_TX    43
#define NECK_RX    44
#define NECK_BAUD  921600
```

---

## 7. Open items for the hardware team

1. **Apply the GPIO21/GPIO38 change to the schematic and layout** before fab.
   Nothing else on this board is blocking.
2. **Decide TOUCH3's mode.** With GPIO42 it is TTP223-only; native copper-pad
   touch is impossible on that pin (§3.3). If all-native touch is required,
   the pin map needs a free GPIO ≤14 and there isn't one — that would mean
   giving up a camera data line, i.e. a different architecture.
3. **Update the two source documents** so they stop disagreeing with each other
   and with this one: `ADAM_Head_Motherboard_BOM_ESP32S3.md` §4 and
   `ADAM_PCB_Reference_v1.pdf` §5.3.
4. `N16R8` vs `N8R8`: both are octal-PSRAM, so this pin map applies to either.
   N16R8's extra flash is what allows dual OTA partitions **plus** LittleFS for
   face-animation frames; N8R8 forces a choice between them.
