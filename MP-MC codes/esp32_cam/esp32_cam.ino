// ============================================================
// ADAM v32 — ESP32-CAM WIRED Vision + Touch + Emotion-Relay Node
// DGEN Technologies Pvt. Ltd., Kolkata
// ============================================================
// CHANGE FROM PREVIOUS BUILD: No WiFi image transfer. Camera frames,
// touch states, and emotion relay all travel over UART. There are now
// TWO separate hardware UARTs on the ESP32-CAM:
//
//   UART2 (existing) — Pi  <-> ESP32-CAM   (frames, touch, gestures,
//                       tilt cmds, cam power cmds, AND now inbound
//                       emotion commands from the Pi)
//   UART1 (NEW)      — ESP32-CAM -> Pico   (emotion relay, one-way)
//
// ──────────────────────────────────────────────────────────────
// *** FIX APPLIED IN THIS REVISION — FALSE TOUCH2/TOUCH3 TRIGGERS ***
// ──────────────────────────────────────────────────────────────
// SYMPTOM: TOUCH2 (GPIO14) and TOUCH3 (GPIO15) were registering
// touches with nobody touching them.
//
// ROOT CAUSE (most likely, in order of probability):
//   1. GPIO14 and GPIO15 are ESP32 JTAG strapping pins. They are
//      electrically "noisier" than a typical GPIO and more sensitive
//      to nearby switching noise (camera XCLK/PCLK toggling at MHz
//      rates lives right next to these in the pin map) and to being
//      left floating.
//   2. The old readTouch() did a single raw digitalRead() per poll,
//      every 20ms, with NO internal pull config specified beyond
//      pinMode(..., INPUT) — leaving the pin's idle state dependent
//      entirely on whatever the touch pad/wire's own parasitic
//      capacitance settles to. A floating digital input WILL read
//      garbage intermittently — that's not a code bug, it's physics.
//   3. Zero debounce — a single noisy HIGH sample was instantly
//      forwarded as a real gesture event.
//
// FIX:
//   - All touch pins now use INPUT_PULLDOWN (ESP32 has real internal
//     pulldowns, unlike classic AVR boards) so each pin idles firmly
//     at LOW instead of floating.
//   - Each poll cycle takes 3 quick samples per pin and majority-votes
//     them, filtering single-cycle glitches.
//   - A pin must read consistently HIGH for TOUCH_DEBOUNCE_MS straight
//     before it's considered "touched" (state-change debounce, applied
//     BEFORE the existing gesture state machine — gesture logic itself
//     is untouched).
//
// IF GHOST TOUCHES PERSIST after flashing this: it's hardware, not
// firmware. Add a physical 10kΩ pull-down resistor directly at each
// touch pad (GPIO -> 10k -> GND), keep touch wires short, and route
// them away from the camera ribbon cable if possible.
//
// ──────────────────────────────────────────────────────────────
// *** UPDATE — CONFIRMED HARDWARE: TTP223 TOUCH SWITCH MODULES ***
// ──────────────────────────────────────────────────────────────
// The touch pads are TTP223 breakout modules, not bare capacitive
// pads. This changes the diagnosis for "touch sticks on for a long
// time until touched again":
//
//   - The TTP223 has a solder-jumper-selectable LATCH/TOGGLE mode.
//     Many boards ship with this ENABLED by default: one touch sets
//     the output HIGH and it STAYS HIGH until touched again to flip
//     it back LOW. That is almost certainly what was happening —
//     it's not a firmware debounce problem, it's the module doing
//     exactly what its jumper tells it to do.
//   - FIX (hardware, on the module itself): find the solder pad
//     usually labeled "A" near the TTP223 IC and bridge/cut it to
//     select MOMENTARY mode instead of TOGGLE mode — output HIGH
//     only while actively touched, LOW when released.
//   - The TTP223 also has a separate jumper for active-HIGH vs
//     active-LOW output polarity, independent of the latch setting.
//     Firmware below now supports EITHER polarity per-pin via the
//     TOUCHx_ACTIVE_LOW defines — flip the relevant one if a pin's
//     sense is inverted (reads "touched" when idle, or vice versa)
//     rather than re-soldering the module.
//
// Since these are TTP223 chip outputs (actively driven HIGH or LOW
// by the module, not a floating bare pad), pinMode is back to plain
// INPUT — INPUT_PULLDOWN is no longer appropriate here and could
// fight a legitimate active-LOW driven signal. The majority-vote +
// debounce logic is kept as cheap insurance against any residual
// electrical noise, but the real fix for the "sticks on" symptom is
// the module's own latch-mode jumper described above.
//
// ──────────────────────────────────────────────────────────────
// WHY A RELAY, NOT A DIRECT PI→PICO WIRE
// ──────────────────────────────────────────────────────────────
// The Pico's firmware (adam_pico_tft.py) is hard-wired to listen on
// GP1 RX expecting the SENDER to be "ESP32-CAM GPIO 4" — that's in
// the Pico file's own header comment and its physical UART0 pin
// assignment. Rewiring that to come from the Pi directly would mean
// re-wiring a physical connector AND changing the Pico's pin config.
// Relaying through the ESP32-CAM instead means zero Pico changes and
// zero new wiring beyond one extra jumper (ESP32 -> Pico, one-way).
//
// COST CHECK — is relaying "heavy" for the ESP32-CAM? No. Emotion
// commands are tiny ASCII strings ("happy\n") sent only when the
// emotion actually changes (at most a few times per minute, not per
// frame). Forwarding one is a single PicoRelay.print() call — a few
// microseconds, no extra RAM, doesn't touch the camera pipeline at
// all. This runs on the same loop() that already polls touch every
// 20ms with zero issues, so an occasional string forward is nothing.
//
// ──────────────────────────────────────────────────────────────
// PROTOCOL TRANSLATION — IMPORTANT
// ──────────────────────────────────────────────────────────────
// The Pi sends emotion commands as "EMO:happy\n" (matches its TILT:/
// CAM: command style). The Pico's firmware expects BARE words with no
// prefix — "happy\n", checked against `if cmd in EMOTIONS`. Relaying
// the line verbatim (with the "EMO:" prefix still attached) would
// make every single command silently fail on the Pico side, because
// "emo:happy" is never a key in its EMOTIONS dict. This sketch strips
// the "EMO:" prefix before forwarding — see relayEmotionToPico().
//
// ──────────────────────────────────────────────────────────────
// CAMERA DUTY-CYCLING (heat/wear protection) — unchanged from prior rev
// ──────────────────────────────────────────────────────────────
//   "CAM:ON\n"   → power up camera (esp_camera_init) and resume
//                  1 FPS frame sending
//   "CAM:OFF\n"  → stop sending frames AND fully deinit the camera
//                  peripheral (esp_camera_deinit), cutting sensor
//                  clock/power draw to near-zero between uses
//
// ──────────────────────────────────────────────────────────────
// WIRING — FULL PIN MAP
// ──────────────────────────────────────────────────────────────
//   UART2 (Pi <-> ESP32-CAM, existing):
//     ESP32-CAM GPIO4  (TX) → Pi GPIO15 (RXD, physical pin 10)
//     ESP32-CAM GPIO16 (RX) → Pi GPIO14 (TXD, physical pin 8)
//     ESP32-CAM GND          → Pi GND (physical pin 6)
//
//   UART1 (ESP32-CAM -> Pico, NEW — emotion relay, one-directional):
//     ESP32-CAM GPIO3 (TX, repurposed U0RXD) → Pico GP1 (physical pin 2)
//     ESP32-CAM GND                           → Pico GND
//
//     NOTE: GPIO3 is normally the USB-serial flashing RX pin (U0RXD).
//     This sketch repurposes it as a plain outbound UART1 TX to the
//     Pico ONLY after the robot is assembled and running. Physically
//     disconnect this wire whenever you reflash the ESP32-CAM over
//     USB-serial, or you'll get bus contention during flashing.
//
//   Tilt servo            → GPIO13
//   Touch1 (left cheek)   → GPIO12
//   Touch2 (right cheek)  → GPIO14  [ESP32-CAM's GPIO14 — unrelated
//                                    to the Pi's GPIO14, different chip]
//   Touch3 (stop/petting) → GPIO15
//   Touch4 (petting)      → GPIO2   (or omitted, see PSRAM_SAFE_BOARD)
//
// PSRAM_SAFE_BOARD note: UART2 RX lives on GPIO16, which is the PSRAM
// chip-select pin on most AI-Thinker boards with populated PSRAM.
// Verify your specific board: if it uses PSRAM on GPIO16, set
// PSRAM_SAFE_BOARD to 0 below (falls back to 3-pad touch + UART2 RX
// on GPIO2 instead).
//
#define PSRAM_SAFE_BOARD 1   // set to 0 if your board uses PSRAM on GPIO16
                             // (falls back to 3-pad wiring, see readTouch())
//
// ──────────────────────────────────────────────────────────────
// LIBRARIES REQUIRED
// ──────────────────────────────────────────────────────────────
//   - ESP32Servo
//   - Board package: esp32 by Espressif Systems (v2.0.x recommended)
//
// ──────────────────────────────────────────────────────────────
// CONFIGURATION CHECKLIST
// ──────────────────────────────────────────────────────────────
//   [ ] Wire ESP32-CAM GPIO4 → Pi GPIO15 (pin 10), GPIO16 → Pi GPIO14
//       (pin 8), GND → Pi GND (pin 6). DOUBLE-CHECK TX→RX / RX→TX
//       crossover — the #1 wired-UART mistake.
//   [ ] Wire ESP32-CAM GPIO3 → Pico GP1 (physical pin 2), GND → Pico GND.
//       Disconnect this wire whenever reflashing over USB-serial.
//   [ ] Confirm PSRAM_SAFE_BOARD matches your actual ESP32-CAM module.
//   [ ] Wire TOUCH1→GPIO12, TOUCH2→GPIO14, TOUCH3→GPIO15, TOUCH4→GPIO2.
//   [ ] UART2_BAUD below MUST match PI_UART_BAUD in adam_main_wifi.py
//       (default 921600 — drop to 460800 if you see corrupted frames).
//   [ ] PICO_RELAY_BAUD below MUST match the Pico's UART0 baudrate
//       (adam_pico_tft.py uses 115200 — do not change unless both
//       sides are updated together).
//   [ ] Camera defaults OFF at boot — confirm the Pi sends "CAM:ON\n"
//       once vision is actually needed.
//   [ ] Pico must be running LIVE MODE (TESTING_MODE = False) to
//       actually respond to relayed commands instead of auto-cycling.
//   [ ] If ghost touches persist after this firmware fix, add a
//       physical 10kΩ pull-down resistor at each touch pad (see note
//       at top of file).
// ============================================================

#include "esp_camera.h"
#include <ESP32Servo.h>

// ── Direct touch GPIO wiring — TTP223 modules (PCF8574T removed) ──
#define TOUCH1_PIN 12   // left cheek
#define TOUCH2_PIN 14   // right cheek
#define TOUCH3_PIN 15   // stop / petting-A

// ── TTP223 output polarity, per channel ───────────────────────
// Most TTP223 boards default to ACTIVE-HIGH (output HIGH when
// touched, LOW when idle) — set the relevant define to 0 for that
// (default below). If a specific module's polarity jumper is set to
// active-LOW instead (output LOW when touched), set that channel's
// define to 1. Symptom of getting this wrong: that pad reads
// "touched" when idle and "not touched" when you actually touch it
// (inverted from what you'd expect).
#define TOUCH1_ACTIVE_LOW 0
#define TOUCH2_ACTIVE_LOW 0
#define TOUCH3_ACTIVE_LOW 0
#define TOUCH4_ACTIVE_LOW 0

#if PSRAM_SAFE_BOARD
  #define TOUCH4_PIN   2   // petting-B
  #define UART2_RX_PIN 16  // PSRAM CS pin on most boards — see note above
#else
  // Fallback: board uses PSRAM on GPIO16 — cannot use it for UART2.
  // Keep UART2 RX on GPIO2 and drop to a 3-pad layout: Touch3 alone
  // still works as STOP, but there is no separate Touch4 pad — treat
  // a long-hold on Touch3 as "petting" instead in processGestures().
  #define UART2_RX_PIN 2
  #define NO_TOUCH4 1
#endif

#define TILT_PIN      13
#define UART2_TX_PIN   4    // → Pi GPIO15 (RXD)
#define UART2_BAUD     921600

// ── UART1 — ESP32-CAM -> Pico (emotion relay, one-directional) ───
#define PICO_RELAY_TX_PIN  3     // repurposed U0RXD, see wiring note above
#define PICO_RELAY_BAUD    115200

// Camera pins — standard AI-Thinker ESP32-CAM module
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

Servo tiltServo;
HardwareSerial PiLink(2);        // UART2 — dedicated wired link to Pi
HardwareSerial PicoRelay(1);     // UART1 — dedicated wired link to Pico

// ── Camera power state (duty-cycling) ─────────────────────────
bool camera_on = false;   // starts OFF — Pi must send CAM:ON to enable

bool startCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk  = XCLK_GPIO_NUM;
    config.pin_pclk  = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href  = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn  = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format  = PIXFORMAT_JPEG;
    config.frame_size    = FRAMESIZE_VGA;   // drop to QVGA if UART is bottlenecked
    config.jpeg_quality  = 12;

    // Detect PSRAM properly instead of assuming — fb_count=2 needs it.
    if (psramFound()) {
        config.fb_count = 2;
        config.fb_location = CAMERA_FB_IN_PSRAM;
    } else {
        config.fb_count = 1;
        config.fb_location = CAMERA_FB_IN_DRAM;
        Serial.println("⚠️  No PSRAM detected — using 1 frame buffer in DRAM");
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init FAILED (0x%x)\n", err);
        return false;
    }
    Serial.println("Camera ON — streaming resumed");
    return true;
}

void stopCamera() {
    // Fully deinit — cuts sensor clock/power draw between uses instead
    // of just pausing frame sends. This is what actually reduces heat.
    esp_camera_deinit();
    Serial.println("Camera OFF — sensor deinitialized");
}

// ──────────────────────────────────────────────────────────────
// WIRE PROTOCOL — Pi <-> ESP32-CAM (over PiLink / UART2)
// ──────────────────────────────────────────────────────────────
// Outbound to Pi (binary, tag-prefixed):
//   'F' <uint32 length LE> <JPEG bytes>          — camera frame
//   'T' <4 bytes: t1 t2 t3 t4>  (0/1 each)        — raw touch state
//   'G' <1 byte gesture code>                     — decoded gesture event
//         gesture codes: 0=none 1=angry(slap) 2=petting 3=stop
//
// Inbound from Pi (text-line protocol, newline-terminated):
//   "TILT:90\n"     → move tilt servo to given angle
//   "CAM:ON\n"      → power up camera sensor, resume 1 FPS frame sends
//   "CAM:OFF\n"     → power down camera sensor, stop frame sends
//   "EMO:happy\n"   → relay to the Pico over PicoRelay (UART1), with
//                     the "EMO:" prefix STRIPPED before forwarding —
//                     see relayEmotionToPico()

const uint8_t TAG_FRAME   = 'F';
const uint8_t TAG_TOUCH   = 'T';
const uint8_t TAG_GESTURE = 'G';

const uint8_t GESTURE_NONE    = 0;
const uint8_t GESTURE_ANGRY   = 1;   // cheek slap (touch1 or touch2)
const uint8_t GESTURE_PETTING = 2;   // touch3 + touch4 together
const uint8_t GESTURE_STOP    = 3;   // touch3 alone

void sendFrame(camera_fb_t *fb) {
    uint32_t len = fb->len;
    PiLink.write(TAG_FRAME);
    PiLink.write((uint8_t*)&len, 4);   // little-endian uint32
    PiLink.write(fb->buf, fb->len);
}

void sendTouch(int t[4]) {
    PiLink.write(TAG_TOUCH);
    uint8_t payload[4] = {(uint8_t)t[0], (uint8_t)t[1], (uint8_t)t[2], (uint8_t)t[3]};
    PiLink.write(payload, 4);
}

void sendGesture(uint8_t code) {
    PiLink.write(TAG_GESTURE);
    PiLink.write(code);
}

// ──────────────────────────────────────────────────────────────
// EMOTION RELAY — Pi tells ESP32-CAM, ESP32-CAM tells Pico
// ──────────────────────────────────────────────────────────────
// Strips the "EMO:" prefix before forwarding, because the Pico's
// firmware checks the bare word against its EMOTIONS dict (e.g.
// "happy", not "EMO:happy" or "emo:happy"). Forwarding verbatim would
// make every relayed command silently fail to match on the Pico side.
void relayEmotionToPico(const String &line) {
    String bare = line;
    if (bare.startsWith("EMO:")) {
        bare = bare.substring(4);
    }
    bare.trim();
    if (bare.length() == 0) return;
    PicoRelay.print(bare);
    PicoRelay.print('\n');
}

// ──────────────────────────────────────────────────────────────
// TOUCH READING — TTP223 module outputs, polarity-aware + debounced
// ──────────────────────────────────────────────────────────────
// These pins are driven by TTP223 touch IC outputs, not bare floating
// pads — the module itself actively pulls the line HIGH or LOW. Two
// light layers of noise rejection still sit in front of the existing
// gesture state machine (processGestures() below is UNCHANGED — it
// still just consumes a clean int t[4] of 0/1 "touched" values, so
// nothing downstream needed to change):
//
//   1. MAJORITY-VOTE SAMPLING: each call takes 3 quick back-to-back
//      digitalRead()s per pin (a few microseconds apart) and keeps
//      the majority value. Kills single-cycle electrical glitches
//      (e.g. a spike from camera XCLK/PCLK switching).
//
//   2. STATE DEBOUNCE: a pin's raw (post-majority-vote, polarity-
//      corrected) reading must stay consistently "touched" for
//      TOUCH_DEBOUNCE_MS in a row before it's reported as touched to
//      the rest of the program. Cheap insurance against noise.
//
// NOTE: this does NOT fix a module left in TOGGLE/LATCH mode — that
// is a hardware jumper setting on the TTP223 board itself (see the
// header comment block near the top of this file). If touches are
// "sticking on" until touched again, fix the module's jumper first;
// this firmware layer only handles noise, not latch behavior.
//
// POLARITY: each channel's TOUCHx_ACTIVE_LOW define (set above, near
// the pin definitions) controls whether a driven LOW or driven HIGH
// from the module means "touched". _rawTouched() applies that
// correction before anything else sees the value, so every other
// piece of code below (majority vote, debounce, gesture machine,
// UART payloads) always deals in the same convention:
//   1 = touched, 0 = not touched — regardless of the module's wiring.
#define TOUCH_DEBOUNCE_MS 60   // finger touches last way longer than
                                 // this; noise glitches do not. Raise
                                 // this (e.g. 100-150) if false
                                 // triggers still slip through.

struct TouchDebounce {
    bool stable_state   = false;  // last confirmed, debounced state
    bool candidate_state = false; // raw state currently being confirmed
    unsigned long candidate_since = 0;
};
TouchDebounce _touchDb[4];

// Applies per-channel active-high/active-low polarity correction to
// a raw digitalRead(). Returns true if this reading means "touched".
static bool _rawTouched(int idx, int pin) {
    bool level = digitalRead(pin);
    bool active_low;
    switch (idx) {
        case 0: active_low = TOUCH1_ACTIVE_LOW; break;
        case 1: active_low = TOUCH2_ACTIVE_LOW; break;
        case 2: active_low = TOUCH3_ACTIVE_LOW; break;
        default: active_low = TOUCH4_ACTIVE_LOW; break;
    }
    return active_low ? !level : level;
}

// Majority-vote read of a single pin — 3 samples, few us apart —
// with polarity correction already applied.
static bool _readPinVoted(int idx, int pin) {
    int touched_count = 0;
    for (int i = 0; i < 3; i++) {
        if (_rawTouched(idx, pin)) touched_count++;
        delayMicroseconds(20);
    }
    return touched_count >= 2;   // majority
}

// Runs the vote + debounce pipeline for one channel, returns the
// current DEBOUNCED state (0/1) to use everywhere else in the sketch.
static int _debouncedTouch(int idx, int pin) {
    unsigned long now = millis();
    bool raw = _readPinVoted(idx, pin);

    TouchDebounce &d = _touchDb[idx];

    if (raw != d.candidate_state) {
        // Raw reading changed — restart the debounce timer for this
        // new candidate value.
        d.candidate_state = raw;
        d.candidate_since = now;
    } else {
        // Raw reading has been consistent — check if it's been
        // consistent long enough to promote to the stable state.
        if ((now - d.candidate_since) >= TOUCH_DEBOUNCE_MS) {
            d.stable_state = d.candidate_state;
        }
    }
    return d.stable_state ? 1 : 0;
}

void readTouch(int out[4]) {
    out[0] = _debouncedTouch(0, TOUCH1_PIN);
    out[1] = _debouncedTouch(1, TOUCH2_PIN);
    out[2] = _debouncedTouch(2, TOUCH3_PIN);
#ifdef NO_TOUCH4
    out[3] = 0;   // no physical 4th pad on this board variant — see
                  // PSRAM_SAFE_BOARD note; petting falls back to a
                  // long-hold on Touch3 inside processGestures()
#else
    out[3] = _debouncedTouch(3, TOUCH4_PIN);
#endif
}

// ──────────────────────────────────────────────────────────────
// GESTURE STATE MACHINE  (UNCHANGED — consumes clean debounced t[4])
// ──────────────────────────────────────────────────────────────
#define PETTING_DEBOUNCE_MS 120
#define SLAP_COOLDOWN_MS    600   // don't spam "angry" while cheek held
#define GESTURE_POLL_MS     20
#define LONGHOLD_PETTING_MS 500  // only used in NO_TOUCH4 fallback mode

unsigned long t3_only_since = 0;
bool t3_pending = false;
unsigned long last_slap_t = 0;
bool prev_petting = false;

void processGestures(int t[4]) {
    unsigned long now = millis();
    bool t1 = t[0], t2 = t[1], t3 = t[2], t4 = t[3];

    if ((t1 || t2) && (now - last_slap_t > SLAP_COOLDOWN_MS)) {
        sendGesture(GESTURE_ANGRY);
        last_slap_t = now;
    }

#ifdef NO_TOUCH4
    bool petting_now = t3 && (now - t3_only_since >= LONGHOLD_PETTING_MS) && t3_pending;
    if (t3) {
        if (!t3_pending) { t3_pending = true; t3_only_since = now; }
        if (petting_now && !prev_petting) sendGesture(GESTURE_PETTING);
        else if (!petting_now && (now - t3_only_since >= PETTING_DEBOUNCE_MS) && !prev_petting) {
            sendGesture(GESTURE_STOP);
        }
    } else {
        t3_pending = false;
    }
    prev_petting = petting_now;
#else
    bool petting_now = (t3 && t4);
    if (petting_now && !prev_petting) {
        sendGesture(GESTURE_PETTING);
    }
    prev_petting = petting_now;

    if (t3 && !t4) {
        if (!t3_pending) {
            t3_pending = true;
            t3_only_since = now;
        } else if (now - t3_only_since >= PETTING_DEBOUNCE_MS) {
            sendGesture(GESTURE_STOP);
            t3_pending = false;
        }
    } else {
        t3_pending = false;
    }
#endif
}

// ──────────────────────────────────────────────────────────────
// SETUP
// ──────────────────────────────────────────────────────────────
void setup() {
    // Serial (U0, GPIO1/3) is normally free for flashing/debug. GPIO3
    // is now ALSO wired to the Pico relay post-assembly — see wiring
    // note above. This Serial.begin() is boot-time debug logging only;
    // disconnect the Pico wire if you need to watch this monitor while
    // GPIO3 is also driving PicoRelay, to avoid confusing output.
    Serial.begin(115200);
    delay(300);
    Serial.println("ADAM ESP32-CAM (wired, Pi<->ESP32<->Pico relay) booting...");

    // ── Touch init — TTP223 modules, direct GPIO, no PCF8574T ────
    // Plain INPUT here (not INPUT_PULLDOWN): TTP223 outputs are
    // actively driven HIGH or LOW by the module itself, not a
    // floating bare pad, so an internal pulldown isn't appropriate
    // and could fight a legitimate active-LOW signal on a module
    // whose polarity jumper is set that way.
    pinMode(TOUCH1_PIN, INPUT);
    pinMode(TOUCH2_PIN, INPUT);
    pinMode(TOUCH3_PIN, INPUT);
#ifndef NO_TOUCH4
    pinMode(TOUCH4_PIN, INPUT);
#endif
    Serial.println("Touch: TTP223 modules, direct GPIO, polarity-aware + debounce");
    Serial.println("  NOTE: if a touch 'sticks on' until touched again, that's the");
    Serial.println("  TTP223's own LATCH/TOGGLE jumper — see file header for the fix.");

    // ── Servo init ───────────────────────────────────────────
    ESP32PWM::allocateTimer(0);
    tiltServo.setPeriodHertz(50);
    tiltServo.attach(TILT_PIN, 500, 2400);
    tiltServo.write(85);  // center

    // ── Wired UART2 link to Pi ────────────────────────────────
    PiLink.begin(UART2_BAUD, SERIAL_8N1, UART2_RX_PIN, UART2_TX_PIN);
    Serial.printf("UART2 to Pi ready @ %d baud (TX=GPIO%d RX=GPIO%d)\n",
                  UART2_BAUD, UART2_TX_PIN, UART2_RX_PIN);

    // ── Wired UART1 relay to Pico (emotion commands only) ─────
    PicoRelay.begin(PICO_RELAY_BAUD, SERIAL_8N1, -1, PICO_RELAY_TX_PIN);
    Serial.printf("UART1 relay to Pico ready @ %d baud (TX=GPIO%d)\n",
                  PICO_RELAY_BAUD, PICO_RELAY_TX_PIN);

    // Camera starts OFF — Pi commands CAM:ON when vision is needed.
    Serial.println("Camera starts OFF — waiting for CAM:ON from Pi");

    Serial.println("ADAM ESP32-CAM ready — frames/touch to Pi, emotion relay to Pico.");
}

// ──────────────────────────────────────────────────────────────
// MAIN LOOP
// ──────────────────────────────────────────────────────────────
unsigned long last_frame_t = 0;
unsigned long last_touch_poll = 0;
const unsigned long FRAME_INTERVAL_MS = 1000;   // 1 FPS to match Gemini Live cap

// ── Camera auto-off watchdog (safety net, independent of the Pi) ──────
// If the Pi process crashes/hangs/loses power uncleanly, it may never
// send a final CAM:OFF. The Pi's camera() task re-sends CAM:ON
// periodically as a keepalive while active, so if this ESP32 hears NO
// command at all (CAM:ON, CAM:OFF, TILT, or EMO) for CAM_WATCHDOG_MS,
// it force-powers the camera down on its own.
#define CAM_WATCHDOG_MS 30000
unsigned long last_pi_cmd_t = 0;

void loop() {
    unsigned long now = millis();

    // ── Send camera frame at 1 FPS — ONLY while camera_on ────────
    if (camera_on && now - last_frame_t >= FRAME_INTERVAL_MS) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb) {
            sendFrame(fb);
            esp_camera_fb_return(fb);
        } else {
            Serial.println("⚠️  esp_camera_fb_get() returned NULL — frame dropped");
        }
        last_frame_t = now;
    }

    // ── Poll touch + run gesture state machine (always active) ──
    if (now - last_touch_poll >= GESTURE_POLL_MS) {
        int t[4];
        readTouch(t);
        sendTouch(t);
        processGestures(t);
        last_touch_poll = now;
    }

    // ── Camera auto-off watchdog ──────────────────────────────────
    if (camera_on && last_pi_cmd_t != 0
            && (now - last_pi_cmd_t) > CAM_WATCHDOG_MS) {
        Serial.println("WATCHDOG: no Pi commands for 30s — forcing camera OFF");
        stopCamera();
        camera_on = false;
    }

    // ── Listen for incoming commands from Pi ─────────────────────
    // Text-line protocol inbound over PiLink (wired UART2 from Pi):
    //   "TILT:90\n"     → move tilt servo
    //   "CAM:ON\n"      → power up camera, resume frame sends
    //   "CAM:OFF\n"     → power down camera, stop frame sends
    //   "EMO:happy\n"   → relay to Pico (prefix stripped) over UART1
    static String inbuf;
    while (PiLink.available()) {
        char c = PiLink.read();
        if (c == '\n') {
            last_pi_cmd_t = now;   // any valid-looking line resets the watchdog
            if (inbuf.startsWith("TILT:")) {
                int angle = inbuf.substring(5).toInt();
                angle = constrain(angle, 50, 120);
                tiltServo.write(angle);
            } else if (inbuf == "CAM:ON") {
                if (!camera_on) {
                    camera_on = startCamera();
                    last_frame_t = 0;  // send a frame immediately on wake
                }
            } else if (inbuf == "CAM:OFF") {
                if (camera_on) {
                    stopCamera();
                    camera_on = false;
                }
            } else if (inbuf.startsWith("EMO:")) {
                relayEmotionToPico(inbuf);
            }
            inbuf = "";
        } else if (c != '\r') {
            inbuf += c;
            if (inbuf.length() > 64) {
                Serial.println("⚠️  inbuf overflow — dropping partial line");
                inbuf = "";  // guard against garbage
            }
        }
    }
}
