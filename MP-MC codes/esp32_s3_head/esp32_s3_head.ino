// ═════════════════════════════════════════════════════════════════════════════
// esp32_s3_head.ino — ADAM Head Controller (production, custom PCB)
// Dgen Technologies Pvt. Ltd., Kolkata  |  October 2026
//
// Target : ESP32-S3-WROOM-1-N16R8  (16 MB flash, 8 MB OCTAL PSRAM)
// Board  : ADAM Head Motherboard — unified camera + display + touch + tilt
// Pin map: Hardware_Docs/PCB_Docs/ADAM_Head_PinMap_v2_N16R8.md   <-- AUTHORITATIVE
//
// This single MCU replaces the prototype's TWO chips (ESP32-CAM + RP2040 Pico).
// The Pico is gone: this firmware drives the ILI9341 face directly, so there is
// no "relay the emotion onward" step any more.
//
// ─── REQUIRED LIBRARIES ─────────────────────────────────────────────────────
//   TFT_eSPI        (>=2.5)  Bodmer     — display driver for the face renderer
//   NimBLE-Arduino  (>=1.4)  h2zero     — BLE stack (~30 KB lighter than
//                                         Bluedroid; RAM matters here because
//                                         the camera also wants DMA buffers)
//   esp32 Arduino core (>=3.0) — provides esp_camera
//
// TFT_eSPI takes its pins from a compile-time header, NOT from the constructor,
// so its config cannot live in this file. Copy `User_Setup_ADAM_S3.h` (shipped
// beside this sketch) into the TFT_eSPI library folder and select it in
// `User_Setup_Select.h`. If the display stays black, that step was skipped.
//
// ─── BOARD SETTINGS (Arduino IDE) ───────────────────────────────────────────
//   Board            : ESP32S3 Dev Module
//   Flash Size       : 16MB (128Mb)
//   PSRAM            : OPI PSRAM          <-- must be OPI, not QSPI, on N16R8
//   Partition Scheme : 16M Flash (3MB APP/9.9MB FATFS)  (or a dual-OTA scheme)
//   USB CDC On Boot  : Enabled            <-- debug goes to USB, see below
//   Upload Mode      : UART0 / Hardware CDC
//
// ─── WHY DEBUG GOES TO USB AND NOT SERIAL ───────────────────────────────────
// The Pi link sits on GPIO43/44, which are the chip's default U0TXD/U0RXD. Two
// consequences, both deliberate here:
//
//   1. The ROM bootloader prints its own banner on GPIO43 at 115200 on every
//      reset, before any of this code runs. The Pi therefore receives a burst
//      of junk on the protocol line at each head reset. That is unavoidable
//      (GPIO46, the boot-log strap, is intentionally left floating) — the Pi
//      side discards unparseable bytes, and must keep doing so.
//   2. If this firmware ALSO logged to UART0 it would corrupt the binary
//      protocol continuously. So it does not.
//
// Instead: the Pi protocol runs on the UART1 peripheral, routed to GPIO43/44
// through the GPIO matrix, and every human-readable message goes to USB CDC on
// the pogo pads. `DBG()` is the only logging macro; it never touches PiLink.
//
// ─── WIRE PROTOCOL (unchanged from the prototype — the Pi needs no edits) ────
// Outbound to Pi (binary, tag-prefixed):
//   'F' <uint32 len LE> <JPEG bytes>   camera frame
//   'T' <t1><t2><t3><t4>               raw touch state, 0/1 each
//   'G' <code>                         gesture: 0 none 1 angry 2 petting 3 stop
//
// Inbound from Pi (newline-terminated text):
//   "TILT:<deg>\n"   move tilt servo
//   "CAM:ON\n"       power up sensor, resume frames
//   "CAM:OFF\n"      deinit sensor, stop frames
//   "EMO:<name>\n"   set the face  (rendered HERE now, not relayed)
//
// Outbound, new in this firmware:
//   "HELLO:S3 <ver>\n"      sync marker so the Pi can discard the ROM banner
//   "PROV:WIFI:<ssid>\t<pass>\n" | "PROV:KEY:<key>\n" | "PROV:MODE:<m>\n"
//   "PROV:LOC:<json>\n"     | "PROV:END\n"
// Inbound, new in this firmware:
//   "PROVSTAT:connecting\n" | "PROVSTAT:connected:<ip>\n"
//   "PROVSTAT:failed:<reason>\n"
//
// NOTE the asymmetry: PROV: lines are TEXT on the same link that carries binary
// frames. That is safe because every text line this firmware emits is sent only
// while provisioning, which happens before the Pi ever enables the camera.
// ═════════════════════════════════════════════════════════════════════════════

#include <Arduino.h>
#include <esp_camera.h>
#include <esp_heap_caps.h>
#include <driver/ledc.h>
#include <TFT_eSPI.h>
#include <NimBLEDevice.h>
#include "adam_emotions.h"

// ─────────────────────────────────────────────────────────────────────────────
// VERSION
// ─────────────────────────────────────────────────────────────────────────────
#define FW_NAME     "adam-head-s3"
#define FW_VERSION  "1.0.0"

// ─────────────────────────────────────────────────────────────────────────────
// PIN MAP — the single place pins are declared.
// Source of truth: ADAM_Head_PinMap_v2_N16R8.md
//
// GPIO21 / GPIO38 for the display bus are the OCTAL-PSRAM CORRECTION. The PCB
// reference assigned GPIO35/36, which on an N16R8 module are wired internally
// to the PSRAM die and are not available externally. Routing the display there
// corrupts PSRAM instead of failing cleanly. Do not "restore" them.
// ─────────────────────────────────────────────────────────────────────────────
#define CAM_XCLK_PIN   15
#define CAM_PCLK_PIN   13
#define CAM_VSYNC_PIN   6
#define CAM_HREF_PIN    7
#define CAM_SIOD_PIN    4   // SCCB data,  2.2k pull-up on board
#define CAM_SIOC_PIN    5   // SCCB clock, 2.2k pull-up on board
#define CAM_D0_PIN     11
#define CAM_D1_PIN      9
#define CAM_D2_PIN      8
#define CAM_D3_PIN     10
#define CAM_D4_PIN     12
#define CAM_D5_PIN     18
#define CAM_D6_PIN     17
#define CAM_D7_PIN     16
// PWDN is held low by a 10k pulldown; RESET has a 10k+100nF RC delay. Neither
// is routed to a GPIO, so both MUST be -1. Declaring a pin the board does not
// drive makes esp_camera_init() fail in a way that looks like a dead sensor.
#define CAM_PWDN_PIN   -1
#define CAM_RESET_PIN  -1

#define TOUCH1_PIN      1   // left cheek
#define TOUCH2_PIN      2   // right cheek
#define TOUCH3_PIN     42   // petting-A   (moved off GPIO3 JTAG strap)
#define TOUCH4_PIN     14   // petting-B

// All four are EXTERNAL TTP223 modules read as digital inputs. Native
// capacitive touch is not an option on this pin map: the S3's touch peripheral
// only exists on GPIO1..GPIO14, and TOUCH3 lives on GPIO42. See the pin-map
// document for why there is no free pin <=14 to move it to.
#define TOUCH1_ACTIVE_LOW 0
#define TOUCH2_ACTIVE_LOW 0
#define TOUCH3_ACTIVE_LOW 0
#define TOUCH4_ACTIVE_LOW 0

#define SERVO_PIN      48   // MG90S tilt, LEDC
#define NECK_TX_PIN    43   // -> Pi RX
#define NECK_RX_PIN    44   // <- Pi TX
#define NECK_BAUD      921600

// ─────────────────────────────────────────────────────────────────────────────
// LOGGING — USB CDC only. Never the Pi link. See the header note.
// ─────────────────────────────────────────────────────────────────────────────
#define DBG(...)   do { Serial.printf(__VA_ARGS__); } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// TRANSPORT
// ─────────────────────────────────────────────────────────────────────────────
HardwareSerial PiLink(1);   // UART1 peripheral, matrixed onto GPIO43/44

TFT_eSPI    tft;
TFT_eSprite spr(&tft);

// ─────────────────────────────────────────────────────────────────────────────
// PROTOCOL CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
static const uint8_t TAG_FRAME   = 'F';
static const uint8_t TAG_TOUCH   = 'T';
static const uint8_t TAG_GESTURE = 'G';

static const uint8_t GESTURE_NONE    = 0;
static const uint8_t GESTURE_ANGRY   = 1;   // cheek slap  (touch1 or touch2)
static const uint8_t GESTURE_PETTING = 2;   // touch3 + touch4 together
static const uint8_t GESTURE_STOP    = 3;   // touch3 alone

// ─────────────────────────────────────────────────────────────────────────────
// RUNTIME STATE
// ─────────────────────────────────────────────────────────────────────────────
static bool     cameraOn        = false;
static bool     cameraInited    = false;
static uint32_t lastFrameMs     = 0;
static uint32_t lastTouchSendMs = 0;
static uint32_t lastCamOkMs     = 0;

static const uint32_t FRAME_INTERVAL_MS = 1000;  // ~1 fps, matches the Pi's duty cycle
static const uint32_t TOUCH_INTERVAL_MS = 50;
static const uint32_t CAM_WATCHDOG_MS   = 30000;

// ═════════════════════════════════════════════════════════════════════════════
// SECTION 1 — CAMERA
// ═════════════════════════════════════════════════════════════════════════════
static camera_config_t makeCameraConfig() {
    camera_config_t c = {};
    c.ledc_channel = LEDC_CHANNEL_0;
    c.ledc_timer   = LEDC_TIMER_0;
    c.pin_d0 = CAM_D0_PIN;  c.pin_d1 = CAM_D1_PIN;
    c.pin_d2 = CAM_D2_PIN;  c.pin_d3 = CAM_D3_PIN;
    c.pin_d4 = CAM_D4_PIN;  c.pin_d5 = CAM_D5_PIN;
    c.pin_d6 = CAM_D6_PIN;  c.pin_d7 = CAM_D7_PIN;
    c.pin_xclk  = CAM_XCLK_PIN;
    c.pin_pclk  = CAM_PCLK_PIN;
    c.pin_vsync = CAM_VSYNC_PIN;
    c.pin_href  = CAM_HREF_PIN;
    // esp32-camera renamed these fields: older headers spell them pin_sscb_*
    // (with the historical typo), newer ones pin_sccb_*. Picking one spelling
    // unguarded is a build failure on the other core version.
#if ESP_ARDUINO_VERSION_MAJOR >= 3
    c.pin_sccb_sda = CAM_SIOD_PIN;
    c.pin_sccb_scl = CAM_SIOC_PIN;
#else
    c.pin_sscb_sda = CAM_SIOD_PIN;
    c.pin_sscb_scl = CAM_SIOC_PIN;
#endif
    c.pin_pwdn  = CAM_PWDN_PIN;
    c.pin_reset = CAM_RESET_PIN;

    // 20 MHz, not 24: the OV2640 is on a 75 mm flex here for neck articulation,
    // and the extra length costs signal integrity on PCLK. 20 MHz is the
    // conservative setting that still clears 1 fps by a wide margin.
    c.xclk_freq_hz = 20000000;
    c.pixel_format = PIXFORMAT_JPEG;
    c.frame_size   = FRAMESIZE_VGA;   // 640x480 — what the Pi forwards to Gemini
    c.jpeg_quality = 12;              // lower number = better quality
    c.fb_count     = 2;
    c.fb_location  = CAMERA_FB_IN_PSRAM;
    c.grab_mode    = CAMERA_GRAB_LATEST;
    return c;
}

static bool startCamera() {
    if (cameraInited) { cameraOn = true; return true; }

    camera_config_t cfg = makeCameraConfig();
    esp_err_t err = esp_camera_init(&cfg);
    if (err != ESP_OK) {
        DBG("[cam] init FAILED 0x%x\n", err);
        cameraInited = false;
        cameraOn     = false;
        return false;
    }

    // Sensor tuning. The camera looks slightly downward from the head, so a
    // small brightness lift keeps faces out of shadow under desk lighting.
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_brightness(s, 1);
        s->set_saturation(s, 0);
        s->set_vflip(s, 0);
        s->set_hmirror(s, 0);
    }

    cameraInited = true;
    cameraOn     = true;
    lastCamOkMs  = millis();
    DBG("[cam] ON  (VGA JPEG q12, PSRAM fb x2)\n");
    return true;
}

static void stopCamera() {
    if (cameraInited) {
        // Full deinit, not just "stop sending". Deinit cuts the sensor clock and
        // its power draw, which is what actually keeps the head cool between
        // vision requests. Pausing the send loop alone does not.
        esp_camera_deinit();
        cameraInited = false;
    }
    cameraOn = false;
    DBG("[cam] OFF (sensor deinitialised)\n");
}

static void sendFrame(camera_fb_t *fb) {
    uint32_t len = fb->len;
    PiLink.write(TAG_FRAME);
    PiLink.write((uint8_t *)&len, 4);     // little-endian uint32
    PiLink.write(fb->buf, fb->len);
}

static void serviceCamera() {
    if (!cameraOn || !cameraInited) return;
    uint32_t now = millis();
    if (now - lastFrameMs < FRAME_INTERVAL_MS) return;
    lastFrameMs = now;

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        // A single miss is normal. A sustained run of them means the sensor has
        // wedged — recover by reinitialising rather than streaming nothing
        // forever and leaving the Pi to guess.
        if (now - lastCamOkMs > CAM_WATCHDOG_MS) {
            DBG("[cam] watchdog: no frame for %lums, reinitialising\n",
                (unsigned long)(now - lastCamOkMs));
            stopCamera();
            delay(100);
            startCamera();
        }
        return;
    }
    lastCamOkMs = now;
    sendFrame(fb);
    esp_camera_fb_return(fb);
}

// ═════════════════════════════════════════════════════════════════════════════
// SECTION 2 — TILT SERVO (LEDC)
// ═════════════════════════════════════════════════════════════════════════════
// Raw LEDC rather than a servo library: one timer, one channel, no dependency,
// and it lets the pulse width be stated in microseconds where the datasheet
// states it. MG90S wants 50 Hz with roughly 500..2400 us of pulse.
//
// The LEDC API CHANGED in ESP32 Arduino core 3.0: ledcSetup()+ledcAttachPin()
// became ledcAttach(), and ledcWrite() takes a PIN instead of a channel. Both
// spellings are guarded below so this sketch builds on 2.x and 3.x rather than
// failing with a confusing "not declared in this scope" on whichever core the
// next person has installed.
static const int      SERVO_LEDC_CH   = 4;      // channels 0..1 belong to the camera
static const int      SERVO_FREQ_HZ   = 50;
static const int      SERVO_RES_BITS  = 16;
static const uint32_t SERVO_MIN_US    = 500;
static const uint32_t SERVO_MAX_US    = 2400;

static int  tiltCurrentDeg = 90;
static bool tiltAttached   = false;
static uint32_t tiltMoveEndMs = 0;

// Detach after the move completes. A servo held under PWM hums at 50 Hz, and on
// a head that hum is picked up by the body's microphones — the same reason the
// pan servo on the body auto-detaches. Silence beats holding torque here.
static const uint32_t TILT_HOLD_MS = 600;

static inline void servoAttach() {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcAttach(SERVO_PIN, SERVO_FREQ_HZ, SERVO_RES_BITS);
#else
    ledcSetup(SERVO_LEDC_CH, SERVO_FREQ_HZ, SERVO_RES_BITS);
    ledcAttachPin(SERVO_PIN, SERVO_LEDC_CH);
#endif
}

static inline void servoDetach() {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcDetach(SERVO_PIN);
#else
    ledcDetachPin(SERVO_PIN);
#endif
}

static inline void servoDuty(uint32_t duty) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcWrite(SERVO_PIN, duty);          // 3.x addresses the PIN
#else
    ledcWrite(SERVO_LEDC_CH, duty);      // 2.x addresses the CHANNEL
#endif
}

static void servoWriteDeg(int deg) {
    deg = constrain(deg, 0, 180);
    uint32_t us = SERVO_MIN_US + (uint32_t)((SERVO_MAX_US - SERVO_MIN_US) * (deg / 180.0f));
    uint32_t maxDuty = (1u << SERVO_RES_BITS) - 1;
    uint32_t duty    = (uint32_t)((us / 20000.0f) * maxDuty);   // 20 ms period
    servoDuty(duty);
}

static void tiltTo(int deg) {
    deg = constrain(deg, 0, 180);
    if (!tiltAttached) {
        servoAttach();
        tiltAttached = true;
    }
    servoWriteDeg(deg);
    tiltCurrentDeg = deg;
    tiltMoveEndMs  = millis() + TILT_HOLD_MS;
    DBG("[tilt] -> %d deg\n", deg);
}

static void serviceServo() {
    if (tiltAttached && tiltMoveEndMs && millis() > tiltMoveEndMs) {
        servoDetach();
        pinMode(SERVO_PIN, INPUT);     // high-Z, no hum
        tiltAttached  = false;
        tiltMoveEndMs = 0;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SECTION 3 — TOUCH + GESTURES
// ═════════════════════════════════════════════════════════════════════════════
// Logic carried over unchanged from the prototype, because it is already tuned
// against real TTP223 modules sitting next to a switching camera bus:
//
//   1. MAJORITY VOTE  — three back-to-back reads per pin, keep the majority.
//      Kills single-cycle glitches coupled from XCLK/PCLK.
//   2. STATE DEBOUNCE — a channel must read touched continuously for
//      TOUCH_DEBOUNCE_MS before it counts. A finger lasts far longer than that;
//      noise does not.
//
// Neither layer fixes a module left in TOGGLE/LATCH mode — that is a solder
// jumper on the TTP223 board itself. If a touch sticks on until touched again,
// fix the jumper; no amount of firmware will help.
#define TOUCH_DEBOUNCE_MS 60

static const uint8_t TOUCH_PINS[4]      = { TOUCH1_PIN, TOUCH2_PIN, TOUCH3_PIN, TOUCH4_PIN };
static const uint8_t TOUCH_ACTIVE_LOW[4] = { TOUCH1_ACTIVE_LOW, TOUCH2_ACTIVE_LOW,
                                             TOUCH3_ACTIVE_LOW, TOUCH4_ACTIVE_LOW };

static int      touchStable[4]   = { 0, 0, 0, 0 };
static int      touchCandidate[4] = { 0, 0, 0, 0 };
static uint32_t touchSinceMs[4]  = { 0, 0, 0, 0 };

static int rawTouched(int i) {
    int a = digitalRead(TOUCH_PINS[i]);
    int b = digitalRead(TOUCH_PINS[i]);
    int c = digitalRead(TOUCH_PINS[i]);
    int lvl = (a + b + c) >= 2 ? HIGH : LOW;      // majority of three
    return TOUCH_ACTIVE_LOW[i] ? (lvl == LOW) : (lvl == HIGH);
}

static void readTouch(int out[4]) {
    uint32_t now = millis();
    for (int i = 0; i < 4; i++) {
        int raw = rawTouched(i);
        if (raw != touchCandidate[i]) {
            touchCandidate[i] = raw;
            touchSinceMs[i]   = now;
        } else if (raw != touchStable[i] && (now - touchSinceMs[i]) >= TOUCH_DEBOUNCE_MS) {
            touchStable[i] = raw;
        }
        out[i] = touchStable[i];
    }
}

static void sendTouch(int t[4]) {
    uint8_t payload[4] = { (uint8_t)t[0], (uint8_t)t[1], (uint8_t)t[2], (uint8_t)t[3] };
    PiLink.write(TAG_TOUCH);
    PiLink.write(payload, 4);
}

static void sendGesture(uint8_t code) {
    PiLink.write(TAG_GESTURE);
    PiLink.write(code);
}

// Edge-triggered: one event per gesture, not one per poll. The Pi treats a
// gesture as a discrete interaction (slap -> angry face, petting -> happy), so
// repeating it 20x/second while a finger rests would be wrong.
static uint8_t lastGesture = GESTURE_NONE;

static void processGestures(int t[4]) {
    uint8_t g = GESTURE_NONE;
    if (t[2] && t[3])            g = GESTURE_PETTING;   // both petting pads
    else if (t[0] || t[1])       g = GESTURE_ANGRY;     // either cheek
    else if (t[2])               g = GESTURE_STOP;      // petting-A alone

    if (g != lastGesture) {
        if (g != GESTURE_NONE) {
            sendGesture(g);
            DBG("[touch] gesture %u\n", g);
        }
        lastGesture = g;
    }
}

static void serviceTouch() {
    uint32_t now = millis();
    if (now - lastTouchSendMs < TOUCH_INTERVAL_MS) return;
    lastTouchSendMs = now;

    int t[4];
    readTouch(t);
    processGestures(t);
    sendTouch(t);
}

// ═════════════════════════════════════════════════════════════════════════════
// SECTION 4 — BLE PROVISIONING
// ═════════════════════════════════════════════════════════════════════════════
// The Pi has no Bluetooth: `dtoverlay=disable-bt` reassigns the PL011 UART to
// the neck link so this board can run at 921600. That is why the ESP32 owns the
// radio for setup. Design detail: pi/docs/mobile_ble_sync.md
//
// BLE is used ONLY to hand over Wi-Fi credentials and the AI config. All normal
// traffic is LAN (the Pi's HTTP API on 8766). Advertising stops the moment the
// Pi reports it has joined a network — a provisioning service that stays
// discoverable forever is a door left open.
#define BLE_SVC_UUID        "19B10000-E8F2-537E-4F6C-D104768A1214"
#define BLE_CHR_WIFI_UUID   "0000AD01-0000-1000-8000-00805F9B34FB"
#define BLE_CHR_STATUS_UUID "0000AD02-0000-1000-8000-00805F9B34FB"
#define BLE_CHR_PROOF_UUID  "0000AD03-0000-1000-8000-00805F9B34FB"
#define BLE_CHR_AICFG_UUID  "0000AD04-0000-1000-8000-00805F9B34FB"
#define BLE_CHR_LOC_UUID    "0000AD05-0000-1000-8000-00805F9B34FB"

static NimBLECharacteristic *chrStatus = nullptr;
static bool     bleAdvertising = false;
static bool     provisioned    = false;
static char     deviceShortId[10] = {0};     // "ADAM-3F2A"
static char     pairingNonce[9]   = {0};

// ─── Chunked write reassembly ────────────────────────────────────────────────
// Default ATT MTU is 23 bytes => 20 bytes of payload. Every real payload here
// is bigger: a Gemini API key is ~39 chars, a WPA2 passphrase can be 63, and
// the Wi-Fi JSON clears 100 easily. A single write() would therefore TRUNCATE
// SILENTLY — the worst possible failure for a credential.
//
// So the app sends frames: {"i":<idx>,"n":<total>,"d":"<slice>"} and this
// reassembles them. An incomplete transfer is discarded after a timeout rather
// than being forwarded half-formed to nmcli.
static const size_t   CHUNK_ASM_MAX     = 1024;
static const uint32_t CHUNK_TIMEOUT_MS  = 10000;

struct ChunkAsm {
    String   buf;
    int      expect = -1;    // total chunk count, -1 = idle
    int      next   = 0;     // next index we require
    uint32_t startMs = 0;
    void reset() { buf = ""; expect = -1; next = 0; startMs = 0; }
};

static ChunkAsm asmWifi, asmAiCfg, asmLoc;

// Returns true when a complete payload is assembled into `out`.
static bool feedChunk(ChunkAsm &a, const std::string &raw, String &out) {
    uint32_t now = millis();
    if (a.expect >= 0 && (now - a.startMs) > CHUNK_TIMEOUT_MS) {
        DBG("[ble] chunk transfer timed out, discarding\n");
        a.reset();
    }

    String s(raw.c_str());

    // Un-chunked single write: accept it only if it is plainly a whole JSON
    // object and not a chunk frame. Keeps the simple case simple.
    if (s.indexOf("\"i\"") < 0 || s.indexOf("\"d\"") < 0) {
        out = s;
        a.reset();
        return true;
    }

    int i = -1, n = -1;
    int pi = s.indexOf("\"i\"");
    int pn = s.indexOf("\"n\"");
    int pd = s.indexOf("\"d\"");
    if (pi >= 0) i = s.substring(s.indexOf(':', pi) + 1).toInt();
    if (pn >= 0) n = s.substring(s.indexOf(':', pn) + 1).toInt();
    if (pd < 0 || i < 0 || n <= 0) { a.reset(); return false; }

    int q1 = s.indexOf('"', s.indexOf(':', pd));
    int q2 = (q1 >= 0) ? s.indexOf('"', q1 + 1) : -1;
    if (q1 < 0 || q2 < 0) { a.reset(); return false; }
    String piece = s.substring(q1 + 1, q2);

    if (i == 0) { a.reset(); a.expect = n; a.startMs = now; }
    if (a.expect != n || i != a.next) {
        DBG("[ble] chunk out of order (got %d, want %d) — restarting\n", i, a.next);
        a.reset();
        return false;
    }
    if (a.buf.length() + piece.length() > CHUNK_ASM_MAX) {
        DBG("[ble] chunk payload too large — discarding\n");
        a.reset();
        return false;
    }

    a.buf += piece;
    a.next = i + 1;
    if (a.next >= a.expect) { out = a.buf; a.reset(); return true; }
    return false;
}

// ─── Minimal JSON string extraction ─────────────────────────────────────────
// A full JSON parser is not worth 20 KB here. These payloads are produced by
// our own app and are flat objects of short strings.
static String jsonStr(const String &src, const char *key) {
    String pat = String("\"") + key + "\"";
    int k = src.indexOf(pat);
    if (k < 0) return "";
    int c = src.indexOf(':', k + pat.length());
    if (c < 0) return "";
    int q1 = src.indexOf('"', c);
    if (q1 < 0) return "";
    int q2 = q1 + 1;
    while (q2 < (int)src.length()) {                 // honour backslash escapes
        if (src[q2] == '\\') { q2 += 2; continue; }
        if (src[q2] == '"')  break;
        q2++;
    }
    if (q2 >= (int)src.length()) return "";
    return src.substring(q1 + 1, q2);
}

static void setProvStatus(const char *json) {
    if (!chrStatus) return;
    chrStatus->setValue((uint8_t *)json, strlen(json));
    chrStatus->notify();
}

// Credentials cross to the Pi here. They are NEVER logged: the debug UART is on
// the exposed pogo header, and a Wi-Fi password or API key printed once is a
// password leaked. Only lengths are reported.
static void relayWifi(const String &payload) {
    String ssid = jsonStr(payload, "ssid");
    String pass = jsonStr(payload, "password");
    if (ssid.length() == 0) { DBG("[prov] wifi payload had no ssid\n"); return; }

    // Tab separator, not ':' — an SSID may legally contain ':' and passwords
    // frequently do, so a colon-delimited line cannot be parsed back reliably.
    PiLink.printf("PROV:WIFI:%s\t%s\n", ssid.c_str(), pass.c_str());
    DBG("[prov] wifi relayed (ssid %u chars, pass %u chars)\n",
        ssid.length(), pass.length());
}

static void relayAiCfg(const String &payload) {
    String mode = jsonStr(payload, "mode");
    String key  = jsonStr(payload, "key");
    if (mode.length()) PiLink.printf("PROV:MODE:%s\n", mode.c_str());
    if (key.length())  PiLink.printf("PROV:KEY:%s\n",  key.c_str());
    DBG("[prov] ai cfg relayed (mode %s, key %u chars)\n",
        mode.length() ? mode.c_str() : "-", key.length());
}

static void relayLocation(const String &payload) {
    PiLink.printf("PROV:LOC:%s\n", payload.c_str());
    DBG("[prov] location relayed (%u chars)\n", payload.length());
}

class WifiCB : public NimBLECharacteristicCallbacks {
    void onWrite(NimBLECharacteristic *c) override {
        String whole;
        if (feedChunk(asmWifi, c->getValue(), whole)) {
            relayWifi(whole);
            PiLink.print("PROV:END\n");
            setProvStatus("{\"status\":\"connecting\"}");
        }
    }
};

class AiCfgCB : public NimBLECharacteristicCallbacks {
    void onWrite(NimBLECharacteristic *c) override {
        String whole;
        if (feedChunk(asmAiCfg, c->getValue(), whole)) relayAiCfg(whole);
    }
};

class LocCB : public NimBLECharacteristicCallbacks {
    void onWrite(NimBLECharacteristic *c) override {
        String whole;
        if (feedChunk(asmLoc, c->getValue(), whole)) relayLocation(whole);
    }
};

class ServerCB : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer *, NimBLEConnInfo &info) override {
        // Ask for a bigger MTU immediately. Chunking still runs regardless —
        // iOS negotiates its own MTU and will not honour this — but when the
        // peer agrees, a credential fits in one or two frames instead of six.
        NimBLEDevice::setMTU(256);
        DBG("[ble] central connected\n");
    }
    void onDisconnect(NimBLEServer *, NimBLEConnInfo &, int reason) override {
        DBG("[ble] central disconnected (reason %d)\n", reason);
        asmWifi.reset(); asmAiCfg.reset(); asmLoc.reset();
        if (!provisioned && bleAdvertising) NimBLEDevice::startAdvertising();
    }
};

static void makeIdentity() {
    uint8_t mac[6] = {0};
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    snprintf(deviceShortId, sizeof(deviceShortId), "ADAM-%02X%02X", mac[4], mac[5]);
    uint32_t r = esp_random();
    snprintf(pairingNonce, sizeof(pairingNonce), "%08x", (unsigned)r);
}

static void startBle() {
    makeIdentity();
    NimBLEDevice::init(deviceShortId);
    NimBLEDevice::setPower(ESP_PWR_LVL_P9);

    NimBLEServer *srv = NimBLEDevice::createServer();
    srv->setCallbacks(new ServerCB());

    NimBLEService *svc = srv->createService(BLE_SVC_UUID);

    NimBLECharacteristic *cWifi = svc->createCharacteristic(
        BLE_CHR_WIFI_UUID, NIMBLE_PROPERTY::WRITE);
    cWifi->setCallbacks(new WifiCB());

    chrStatus = svc->createCharacteristic(
        BLE_CHR_STATUS_UUID, NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);
    chrStatus->setValue("{\"status\":\"idle\"}");

    // Proof of possession: only someone physically holding the unit can read
    // this. Being within Bluetooth range is not authorisation, so the backend
    // requires this nonce on /devices/claim.
    NimBLECharacteristic *cProof = svc->createCharacteristic(
        BLE_CHR_PROOF_UUID, NIMBLE_PROPERTY::READ);
    {
        char proof[96];
        snprintf(proof, sizeof(proof), "{\"serial\":\"%s\",\"nonce\":\"%s\"}",
                 deviceShortId, pairingNonce);
        cProof->setValue((uint8_t *)proof, strlen(proof));
    }

    NimBLECharacteristic *cAi = svc->createCharacteristic(
        BLE_CHR_AICFG_UUID, NIMBLE_PROPERTY::WRITE);
    cAi->setCallbacks(new AiCfgCB());

    NimBLECharacteristic *cLoc = svc->createCharacteristic(
        BLE_CHR_LOC_UUID, NIMBLE_PROPERTY::WRITE);
    cLoc->setCallbacks(new LocCB());

    svc->start();

    NimBLEAdvertising *adv = NimBLEDevice::getAdvertising();
    adv->addServiceUUID(BLE_SVC_UUID);
    adv->setName(deviceShortId);
    adv->enableScanResponse(true);
    NimBLEDevice::startAdvertising();
    bleAdvertising = true;

    DBG("[ble] advertising as %s\n", deviceShortId);
}

static void stopBleAdvertising() {
    if (!bleAdvertising) return;
    NimBLEDevice::stopAdvertising();
    bleAdvertising = false;
    DBG("[ble] advertising stopped (provisioned)\n");
}

// ═════════════════════════════════════════════════════════════════════════════
// SECTION 5 — INBOUND COMMANDS FROM THE PI
// ═════════════════════════════════════════════════════════════════════════════
static void handleLine(String line) {
    line.trim();
    if (line.length() == 0) return;

    if (line.startsWith("EMO:")) {
        // Rendered here. The prototype forwarded this to a Pico over a second
        // UART; on this board the display is ours, so the relay is gone.
        String name = line.substring(4);
        name.trim();
        if (name.length()) {
            setEmotion(nameToEmotion(name.c_str()));
            DBG("[emo] %s\n", name.c_str());
        }
        return;
    }

    if (line.startsWith("TILT:")) {
        tiltTo(line.substring(5).toInt());
        return;
    }

    if (line.startsWith("CAM:")) {
        String v = line.substring(4);
        v.trim();
        v.toUpperCase();
        if (v == "ON")       startCamera();
        else if (v == "OFF") stopCamera();
        return;
    }

    if (line.startsWith("PROVSTAT:")) {
        // Provisioning result from the Pi -> straight out as a BLE notify.
        String rest = line.substring(9);
        rest.trim();
        if (rest.startsWith("connected")) {
            int c = rest.indexOf(':');
            String ip = (c >= 0) ? rest.substring(c + 1) : "";
            char j[96];
            snprintf(j, sizeof(j), "{\"status\":\"connected\",\"ip\":\"%s\"}", ip.c_str());
            setProvStatus(j);
            provisioned = true;
            stopBleAdvertising();
        } else if (rest.startsWith("failed")) {
            int c = rest.indexOf(':');
            String why = (c >= 0) ? rest.substring(c + 1) : "unknown";
            char j[128];
            snprintf(j, sizeof(j), "{\"status\":\"failed\",\"err\":\"%s\"}", why.c_str());
            setProvStatus(j);
            // Advertising deliberately stays UP on failure. Dropping the only
            // channel the user has the instant the password is wrong strands
            // them with no way to retype it.
        } else {
            setProvStatus("{\"status\":\"connecting\"}");
        }
        return;
    }

    DBG("[pi] unknown line: %s\n", line.c_str());
}

static void servicePiLink() {
    static String rx;
    while (PiLink.available()) {
        char c = (char)PiLink.read();
        if (c == '\n') { handleLine(rx); rx = ""; }
        else if (c != '\r') {
            // Bound the buffer. The Pi only ever sends short lines; anything
            // longer is noise (e.g. a reset burst) and must not grow forever.
            if (rx.length() < 512) rx += c;
            else rx = "";
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SETUP / LOOP
// ═════════════════════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);               // USB CDC — debug only
    delay(300);
    DBG("\n%s %s booting\n", FW_NAME, FW_VERSION);

    PiLink.begin(NECK_BAUD, SERIAL_8N1, NECK_RX_PIN, NECK_TX_PIN);

    for (int i = 0; i < 4; i++) pinMode(TOUCH_PINS[i], INPUT);

    if (!psramFound()) {
        // Not fatal for the face, but the camera cannot work without it. Say so
        // loudly: the usual cause is the board menu set to QSPI PSRAM on a
        // module that needs OPI.
        DBG("[boot] *** PSRAM NOT FOUND — check 'PSRAM: OPI PSRAM' in board menu\n");
    } else {
        DBG("[boot] PSRAM %u KB free\n",
            (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    }

    tft.init();
    tft.setRotation(1);                // landscape 320x240
    tft.fillScreen(TFT_BLACK);
    initEmotions(tft, spr);
    setEmotion(nameToEmotion("idle"));

    // Printed from TFT_eSPI's OWN macros, not from this sketch. If these do not
    // read 21/38/40/41/47 then User_Setup_ADAM_S3.h was not installed/selected,
    // and the display will be blank or garbled. This is the fastest way to tell
    // that apart from a dead panel.
    DBG("[tft] ILI9341 ready (MOSI=%d SCK=%d DC=%d CS=%d RST=%d)\n",
        TFT_MOSI, TFT_SCLK, TFT_DC, TFT_CS, TFT_RST);
#if (TFT_MOSI == 35) || (TFT_SCLK == 36)
#error "TFT_eSPI is configured for GPIO35/36 - those are the N16R8 octal-PSRAM lines. Install User_Setup_ADAM_S3.h (MOSI=21, SCK=38)."
#endif

    tiltTo(90);                        // centre, then auto-detach
    startBle();

    // Sync marker: everything the Pi saw before this line is ROM-bootloader
    // noise and can be discarded.
    PiLink.printf("HELLO:%s %s\n", FW_NAME, FW_VERSION);
    DBG("[boot] ready. camera starts on CAM:ON from the Pi.\n");
}

void loop() {
    servicePiLink();
    serviceTouch();
    serviceCamera();
    serviceServo();
    updateEmotion();                   // drives the face animation
}
