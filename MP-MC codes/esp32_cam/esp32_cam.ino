// ═════════════════════════════════════════════════════════════════════════════
// esp32_cam.ino — ADAM Head Controller (PROTOTYPE, AI-Thinker ESP32-CAM)
// Dgen Technologies Pvt. Ltd., Kolkata  |  October 2026
//
// Target : AI-Thinker ESP32-CAM (ESP32-S0WD, 4 MB PSRAM)
// Role   : camera + touch + tilt on the breadboard prototype head
//
// This is the PROTOTYPE sibling of `esp32_s3_head/esp32_s3_head.ino`. Both speak
// the SAME wire protocol to the Pi, so the Pi code is identical either way. The
// two differ only where the hardware forces it:
//
//                        │ ESP32-CAM (this file) │ ESP32-S3 (production)
//   ─────────────────────┼───────────────────────┼──────────────────────────
//   TFT face             │ NO — relayed to a     │ YES — driven directly
//                        │ separate RP2040 Pico  │ (Pico eliminated)
//   BLE provisioning     │ NO — see note below   │ YES
//   Free GPIOs           │ almost none           │ comfortable
//   PSRAM                │ 4 MB quad             │ 8 MB octal
//   Touch channels       │ 3, or 4 on some boards│ 4 always
//
// ─── WHY NO BLE HERE ────────────────────────────────────────────────────────
// Not an oversight. The ESP32-CAM runs the camera DMA, two UARTs and the Pico
// relay out of 320 KB of usable internal SRAM. Adding a BLE stack (~40 KB even
// with NimBLE) on top of camera framebuffer descriptors is where this board
// starts failing allocations intermittently — the worst kind of bug. Mobile-app
// provisioning is a PRODUCTION feature and lives on the S3 board, which has the
// headroom. On the prototype, set Wi-Fi in the Pi's `~/adam/.env` by hand.
//
// ─── REQUIRED ───────────────────────────────────────────────────────────────
//   Board package: esp32 by Espressif (2.x or 3.x — both are handled)
//   Board        : "AI Thinker ESP32-CAM"
//   PSRAM        : Enabled
//   No external libraries. The tilt servo uses raw LEDC, so ESP32Servo is NOT
//   needed any more — that dependency was dropped in this revision.
//
// ─── WIRE PROTOCOL (identical to the S3 build — the Pi needs no changes) ─────
// Outbound to Pi (binary, tag-prefixed):
//   'F' <uint32 len LE> <JPEG bytes>   camera frame
//   'T' <t1><t2><t3><t4>               raw touch state, 0/1 each
//   'G' <code>                         gesture: 0 none 1 angry 2 petting 3 stop
//
// Inbound from Pi (newline-terminated text):
//   "TILT:<deg>\n"   move tilt servo
//   "CAM:ON\n"       init sensor, resume frames
//   "CAM:OFF\n"      deinit sensor, stop frames
//   "EMO:<name>\n"   relayed to the Pico with the "EMO:" prefix STRIPPED
//
// ─── GPIO12 BOOT HAZARD — READ BEFORE WIRING TOUCH1 ─────────────────────────
// TOUCH1 is on GPIO12, which is the ESP32's MTDI strapping pin. Its level at
// reset selects the internal flash voltage: LOW = 3.3 V (correct), HIGH = 1.8 V
// (the board will not boot). A TTP223 in ACTIVE-HIGH mode drives its output HIGH
// while touched — so holding the left cheek pad during power-up can stop the
// board booting. It is not broken; let go and re-power. Keep the module in
// active-high mode (the default) and simply don't hold that pad while resetting.
// ═════════════════════════════════════════════════════════════════════════════

#include <Arduino.h>
#include "esp_camera.h"

// ─────────────────────────────────────────────────────────────────────────────
// VERSION
// ─────────────────────────────────────────────────────────────────────────────
#define FW_NAME     "adam-head-cam"
#define FW_VERSION  "2.0.0"

// ─────────────────────────────────────────────────────────────────────────────
// BOARD VARIANT
// ─────────────────────────────────────────────────────────────────────────────
// Some AI-Thinker clones wire PSRAM's chip-select to GPIO16, which makes GPIO16
// unusable for UART2 RX. On those boards set this to 0: UART2 RX falls back to
// GPIO2 and the 4th touch pad is given up (there is no third option — the
// ESP32-CAM simply has no spare pin). With 3 pads, petting is detected as a
// sustained hold on TOUCH3 instead of TOUCH3+TOUCH4 together.
#define PSRAM_SAFE_BOARD 1

// ─────────────────────────────────────────────────────────────────────────────
// PIN MAP
// ─────────────────────────────────────────────────────────────────────────────
#define TOUCH1_PIN  12   // left cheek   — see GPIO12 boot hazard above
#define TOUCH2_PIN  14   // right cheek
#define TOUCH3_PIN  15   // stop / petting-A

#if PSRAM_SAFE_BOARD
  #define TOUCH4_PIN    2   // petting-B
  #define UART2_RX_PIN 16
#else
  #define UART2_RX_PIN  2
  #define NO_TOUCH4     1
#endif

// TTP223 output polarity, per channel. Most boards are ACTIVE-HIGH (output HIGH
// when touched) — leave these at 0. Set a channel to 1 only if that module's
// jumper is set to active-low. Symptom of getting it wrong: the pad reads
// "touched" when idle and "untouched" when you touch it.
#define TOUCH1_ACTIVE_LOW 0
#define TOUCH2_ACTIVE_LOW 0
#define TOUCH3_ACTIVE_LOW 0
#define TOUCH4_ACTIVE_LOW 0

// Master enable. Set false to run the head with no touch modules attached —
// otherwise floating inputs generate phantom gestures.
static const bool TOUCH_ENABLED = false;

#define TILT_PIN        13
#define UART2_TX_PIN     4    // -> Pi GPIO15 (RXD)
#define UART2_BAUD  921600

#define PICO_RELAY_TX_PIN   3      // repurposed U0RXD -> Pico UART0
#define PICO_RELAY_BAUD 115200

// Camera — standard AI-Thinker ESP32-CAM module. Y2..Y9 are D0..D7.
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35   // D7
#define Y8_GPIO_NUM     34   // D6
#define Y7_GPIO_NUM     39   // D5
#define Y6_GPIO_NUM     36   // D4
#define Y5_GPIO_NUM     21   // D3
#define Y4_GPIO_NUM     19   // D2
#define Y3_GPIO_NUM     18   // D1
#define Y2_GPIO_NUM      5   // D0
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22

// ─────────────────────────────────────────────────────────────────────────────
// TRANSPORT
// ─────────────────────────────────────────────────────────────────────────────
// Serial  (UART0) : boot-time debug only. Shares pins with the Pico relay, so
//                   nothing may be printed to it after setup() — see below.
// PiLink  (UART2) : binary protocol to the Pi, 921600.
// PicoRelay(UART1): emotion names to the RP2040, 115200, one-directional.
HardwareSerial PiLink(2);
HardwareSerial PicoRelay(1);

// Debug goes to UART0 ONLY during setup(), before PicoRelay claims GPIO3.
// After that this macro is a no-op: printing would corrupt the Pico link.
static bool dbgOpen = true;
#define DBG(...)   do { if (dbgOpen) Serial.printf(__VA_ARGS__); } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// PROTOCOL CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
static const uint8_t TAG_FRAME   = 'F';
static const uint8_t TAG_TOUCH   = 'T';
static const uint8_t TAG_GESTURE = 'G';

static const uint8_t GESTURE_NONE    = 0;
static const uint8_t GESTURE_ANGRY   = 1;   // cheek slap (touch1 or touch2)
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

static const uint32_t FRAME_INTERVAL_MS = 1000;   // ~1 fps, matches the Pi
static const uint32_t TOUCH_INTERVAL_MS = 50;
static const uint32_t CAM_WATCHDOG_MS   = 30000;

// ═════════════════════════════════════════════════════════════════════════════
// SECTION 1 — CAMERA
// ═════════════════════════════════════════════════════════════════════════════
static camera_config_t makeCameraConfig() {
    camera_config_t c = {};
    c.ledc_channel = LEDC_CHANNEL_0;
    c.ledc_timer   = LEDC_TIMER_0;
    c.pin_d0 = Y2_GPIO_NUM;  c.pin_d1 = Y3_GPIO_NUM;
    c.pin_d2 = Y4_GPIO_NUM;  c.pin_d3 = Y5_GPIO_NUM;
    c.pin_d4 = Y6_GPIO_NUM;  c.pin_d5 = Y7_GPIO_NUM;
    c.pin_d6 = Y8_GPIO_NUM;  c.pin_d7 = Y9_GPIO_NUM;
    c.pin_xclk  = XCLK_GPIO_NUM;
    c.pin_pclk  = PCLK_GPIO_NUM;
    c.pin_vsync = VSYNC_GPIO_NUM;
    c.pin_href  = HREF_GPIO_NUM;
    // esp32-camera renamed these: older headers use pin_sscb_* (historical
    // typo), newer ones pin_sccb_*. Guarded so both core versions build.
#if ESP_ARDUINO_VERSION_MAJOR >= 3
    c.pin_sccb_sda = SIOD_GPIO_NUM;
    c.pin_sccb_scl = SIOC_GPIO_NUM;
#else
    c.pin_sscb_sda = SIOD_GPIO_NUM;
    c.pin_sscb_scl = SIOC_GPIO_NUM;
#endif
    c.pin_pwdn  = PWDN_GPIO_NUM;
    c.pin_reset = RESET_GPIO_NUM;

    c.xclk_freq_hz = 20000000;
    c.pixel_format = PIXFORMAT_JPEG;
    c.frame_size   = FRAMESIZE_VGA;    // 640x480 — what the Pi sends to Gemini
    c.jpeg_quality = 12;
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
    DBG("[cam] ON (VGA JPEG q12)\n");
    return true;
}

static void stopCamera() {
    if (cameraInited) {
        // Full deinit, not just "stop sending frames". Deinit cuts the sensor
        // clock and its current draw, which is what actually stops the head
        // warming up between vision requests. Pausing the send loop does not.
        esp_camera_deinit();
        cameraInited = false;
    }
    cameraOn = false;
    DBG("[cam] OFF (deinitialised)\n");
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
        // One miss is normal. A sustained run means the sensor has wedged —
        // reinitialise rather than silently streaming nothing and leaving the
        // Pi to infer that vision died.
        if (now - lastCamOkMs > CAM_WATCHDOG_MS) {
            DBG("[cam] watchdog: no frame for %lums, reinit\n",
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
// SECTION 2 — TILT SERVO (raw LEDC, no library)
// ═════════════════════════════════════════════════════════════════════════════
// The LEDC API changed in ESP32 Arduino core 3.0 (ledcSetup+ledcAttachPin ->
// ledcAttach, and ledcWrite takes a pin instead of a channel). Both spellings
// are guarded so this builds on 2.x and 3.x.
//
// The servo is DETACHED after each move: PWM held on an MG90S makes it hum at
// 50 Hz, and that hum is picked up by the body's microphones. Same reason the
// body's pan servo auto-detaches.
static const int      SERVO_LEDC_CH  = 4;    // 0..1 belong to the camera
static const int      SERVO_FREQ_HZ  = 50;
static const int      SERVO_RES_BITS = 16;
static const uint32_t SERVO_MIN_US   = 500;
static const uint32_t SERVO_MAX_US   = 2400;
static const uint32_t TILT_HOLD_MS   = 600;

static bool     tiltAttached  = false;
static uint32_t tiltMoveEndMs = 0;

static inline void servoAttach() {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcAttach(TILT_PIN, SERVO_FREQ_HZ, SERVO_RES_BITS);
#else
    ledcSetup(SERVO_LEDC_CH, SERVO_FREQ_HZ, SERVO_RES_BITS);
    ledcAttachPin(TILT_PIN, SERVO_LEDC_CH);
#endif
}

static inline void servoDetach() {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcDetach(TILT_PIN);
#else
    ledcDetachPin(TILT_PIN);
#endif
}

static inline void servoDuty(uint32_t duty) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcWrite(TILT_PIN, duty);
#else
    ledcWrite(SERVO_LEDC_CH, duty);
#endif
}

static void tiltTo(int deg) {
    deg = constrain(deg, 0, 180);
    if (!tiltAttached) { servoAttach(); tiltAttached = true; }
    uint32_t us      = SERVO_MIN_US + (uint32_t)((SERVO_MAX_US - SERVO_MIN_US) * (deg / 180.0f));
    uint32_t maxDuty = (1u << SERVO_RES_BITS) - 1;
    servoDuty((uint32_t)((us / 20000.0f) * maxDuty));    // 20 ms period
    tiltMoveEndMs = millis() + TILT_HOLD_MS;
    DBG("[tilt] -> %d\n", deg);
}

static void serviceServo() {
    if (tiltAttached && tiltMoveEndMs && millis() > tiltMoveEndMs) {
        servoDetach();
        pinMode(TILT_PIN, INPUT);        // high-Z, no hum
        tiltAttached  = false;
        tiltMoveEndMs = 0;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SECTION 3 — TOUCH + GESTURES
// ═════════════════════════════════════════════════════════════════════════════
// Two layers of noise rejection in front of the gesture machine, both tuned
// against real TTP223 modules sitting beside a switching camera bus:
//
//   1. MAJORITY VOTE  — three back-to-back reads per pin, keep the majority.
//      Kills single-cycle glitches coupled from XCLK/PCLK.
//   2. STATE DEBOUNCE — a channel must read touched continuously for
//      TOUCH_DEBOUNCE_MS before it counts. A finger lasts far longer; noise
//      does not.
//
// Neither fixes a module left in TOGGLE/LATCH mode — that is a solder jumper on
// the TTP223 board. If a touch sticks on until touched again, fix the jumper.
//
// Pins use INPUT_PULLDOWN: the ESP32 has real internal pulldowns, so an
// unconnected channel reads a solid 0 instead of floating and inventing touches.
#define TOUCH_DEBOUNCE_MS 60

#ifdef NO_TOUCH4
  #define TOUCH_CHANNELS 3
#else
  #define TOUCH_CHANNELS 4
#endif

static const uint8_t TOUCH_PINS[4] = {
    TOUCH1_PIN, TOUCH2_PIN, TOUCH3_PIN,
#ifdef NO_TOUCH4
    0xFF                      // absent on this board variant
#else
    TOUCH4_PIN
#endif
};
static const uint8_t TOUCH_ACTIVE_LOW[4] = {
    TOUCH1_ACTIVE_LOW, TOUCH2_ACTIVE_LOW, TOUCH3_ACTIVE_LOW, TOUCH4_ACTIVE_LOW
};

static int      touchStable[4]    = { 0, 0, 0, 0 };
static int      touchCandidate[4] = { 0, 0, 0, 0 };
static uint32_t touchSinceMs[4]   = { 0, 0, 0, 0 };

static int rawTouched(int i) {
    if (TOUCH_PINS[i] == 0xFF) return 0;
    int a = digitalRead(TOUCH_PINS[i]);
    int b = digitalRead(TOUCH_PINS[i]);
    int c = digitalRead(TOUCH_PINS[i]);
    int lvl = (a + b + c) >= 2 ? HIGH : LOW;          // majority of three
    return TOUCH_ACTIVE_LOW[i] ? (lvl == LOW) : (lvl == HIGH);
}

static void readTouch(int out[4]) {
    uint32_t now = millis();
    for (int i = 0; i < 4; i++) {
        if (!TOUCH_ENABLED || TOUCH_PINS[i] == 0xFF) { out[i] = 0; continue; }
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
// gesture as a discrete interaction, so repeating it 20x/second while a finger
// rests on a pad would be wrong.
static uint8_t  lastGesture   = GESTURE_NONE;
static uint32_t touch3HeldMs  = 0;

static void processGestures(int t[4]) {
    uint8_t g = GESTURE_NONE;

#ifdef NO_TOUCH4
    // 3-pad fallback: there is no petting-B channel, so a SUSTAINED hold on
    // TOUCH3 means petting while a short press means stop.
    if (t[2]) {
        if (touch3HeldMs == 0) touch3HeldMs = millis();
        g = (millis() - touch3HeldMs > 700) ? GESTURE_PETTING : GESTURE_STOP;
    } else {
        touch3HeldMs = 0;
        if (t[0] || t[1]) g = GESTURE_ANGRY;
    }
#else
    if (t[2] && t[3])      g = GESTURE_PETTING;   // both petting pads
    else if (t[0] || t[1]) g = GESTURE_ANGRY;     // either cheek
    else if (t[2])         g = GESTURE_STOP;      // petting-A alone
#endif

    if (g != lastGesture) {
        if (g != GESTURE_NONE) sendGesture(g);
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
// SECTION 4 — EMOTION RELAY TO THE PICO
// ═════════════════════════════════════════════════════════════════════════════
// The prefix MUST be stripped. The Pico's firmware matches the bare word
// against its own emotion table ("happy", not "EMO:happy"), so forwarding the
// line verbatim makes every relayed emotion silently fail to match.
//
// The S3 board has no equivalent of this function: it owns the display.
static void relayEmotionToPico(const String &line) {
    String bare = line;
    if (bare.startsWith("EMO:")) bare = bare.substring(4);
    bare.trim();
    if (bare.length() == 0) return;
    PicoRelay.print(bare);
    PicoRelay.print('\n');
}

// ═════════════════════════════════════════════════════════════════════════════
// SECTION 5 — INBOUND COMMANDS FROM THE PI
// ═════════════════════════════════════════════════════════════════════════════
static void handleLine(String line) {
    line.trim();
    if (line.length() == 0) return;

    if (line.startsWith("EMO:"))  { relayEmotionToPico(line);     return; }
    if (line.startsWith("TILT:")) { tiltTo(line.substring(5).toInt()); return; }

    if (line.startsWith("CAM:")) {
        String v = line.substring(4);
        v.trim();
        v.toUpperCase();
        if (v == "ON")       startCamera();
        else if (v == "OFF") stopCamera();
        return;
    }
}

static void servicePiLink() {
    static String rx;
    while (PiLink.available()) {
        char c = (char)PiLink.read();
        if (c == '\n') { handleLine(rx); rx = ""; }
        else if (c != '\r') {
            // Bound the buffer — the Pi only sends short lines, so anything
            // longer is noise and must not be allowed to grow without limit.
            if (rx.length() < 256) rx += c;
            else rx = "";
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SETUP / LOOP
// ═════════════════════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    delay(200);
    DBG("\n%s %s booting\n", FW_NAME, FW_VERSION);
    DBG("[boot] touch %s, %d channel(s), PSRAM_SAFE_BOARD=%d\n",
        TOUCH_ENABLED ? "ENABLED" : "disabled", TOUCH_CHANNELS, PSRAM_SAFE_BOARD);

    for (int i = 0; i < 4; i++) {
        if (TOUCH_PINS[i] != 0xFF) pinMode(TOUCH_PINS[i], INPUT_PULLDOWN);
    }

    PiLink.begin(UART2_BAUD, SERIAL_8N1, UART2_RX_PIN, UART2_TX_PIN);
    tiltTo(90);                       // centre, then auto-detach

    DBG("[boot] ready. camera starts on CAM:ON from the Pi.\n");

    // LAST THING in setup(): claim GPIO3 for the Pico relay and close the debug
    // port. UART0's RX pin and the relay's TX pin are the same physical GPIO, so
    // any Serial.print() after this would be written onto the Pico's link and
    // corrupt it. DBG() becomes a no-op from here on, by design.
    Serial.flush();
    dbgOpen = false;
    PicoRelay.begin(PICO_RELAY_BAUD, SERIAL_8N1, -1, PICO_RELAY_TX_PIN);
}

void loop() {
    servicePiLink();
    serviceTouch();
    serviceCamera();
    serviceServo();
}
