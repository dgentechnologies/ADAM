/*
  ADAM — Neck Servo Controller (v2 — Human-like eased motion)
  ============================================================
  Hardware:
    Pin 9  → Pan servo  (MG995, left/right)
    Pin 10 → Tilt servo (MG995, up/down)

  Serial protocol (9600 baud, newline-terminated):
    P<angle>          →  Pan  to angle (0–180, centre = 90)
    T<angle>          →  Tilt to angle (0–180, centre = 90)
    N<move>           →  Named movement: NOD, SHAKE, RESET, LOOK_UP, LOOK_DOWN,
                                         LOOK_LEFT, LOOK_RIGHT, TILT_CURIOUS
    SPEED<value>      →  Set global speed: 1 (slow) … 10 (fast), default 5
    S                 →  Status ping → replies "OK\n"

  Motion model (v2):
    - Sine ease-in/ease-out on every move (no more constant-speed linear steps)
    - Smooth micro-delay derived from the easing curve — fastest at midpoint,
      slowest at start and end, mimicking real mechanical inertia
    - Speed parameter scales the base step delay so ADAM can move at
      different rates per context (fast greeting snap vs slow curious tilt)
    - All animations (NOD, SHAKE, etc.) composed from the same eased primitive
      so every motion feels consistent
*/

#include <Servo.h>
#include <math.h>

Servo panServo;
Servo tiltServo;

// ── Neutral / limit positions ────────────────────────────────────
const int PAN_MIN    = 30;
const int PAN_MAX    = 150;
const int PAN_CENTER = 90;

const int TILT_MIN    = 50;
const int TILT_MAX    = 120;
const int TILT_CENTER = 85;

// Current positions
int currentPan  = PAN_CENTER;
int currentTilt = TILT_CENTER;

// Global speed multiplier: 1 (slowest) .. 10 (fastest)
// Base delay at speed=5 is ~6ms/step, scaled inversely
int globalSpeed = 5;   // default

// ── Eased single-axis move ────────────────────────────────────────
//
// Uses a sine ease-in/out curve:
//   progress = t / steps  (0.0 → 1.0)
//   eased    = (1 - cos(π × progress)) / 2   → smooth S-curve
//   position = start + (end - start) × eased
//
// The delay per step is also modulated so the servo physically
// slows at start and end, matching the position curve.
//
//  speedOverride: 0 = use globalSpeed, 1-10 = explicit
//
void easedMove(Servo &servo, int &current, int target, int speedOverride = 0) {
  target = constrain(target, 0, 180);
  if (current == target) return;

  int spd = (speedOverride > 0) ? speedOverride : globalSpeed;

  // Base step count: more steps = smoother arc, independent of distance
  // Longer distances use more steps so the easing looks consistent
  int dist  = abs(target - current);
  int steps = max(20, dist * 2);           // at least 20 steps, ~2 per degree

  // Base delay per step at speed=5: 7ms.  Scales: speed=1→35ms, speed=10→3ms
  // Formula: baseDelay = map(spd, 1, 10, 35, 3)
  float baseDelay = 35.0f - (spd - 1) * (35.0f - 3.0f) / 9.0f;

  int prevPos = current;

  for (int i = 1; i <= steps; i++) {
    float t       = (float)i / (float)steps;
    // Sine ease-in/out: t → 0..1 mapped through (1-cos(πt))/2
    float eased   = (1.0f - cos(PI * t)) / 2.0f;
    int   newPos  = current + (int)round((target - current) * eased);
    newPos = constrain(newPos, 0, 180);

    if (newPos != prevPos) {
      servo.write(newPos);
      prevPos = newPos;
    }

    // Delay modulation: slowest at start/end (easing derivative near 0),
    // fastest at midpoint (derivative near 1).
    // derivative of (1-cos(πt))/2 = (π/2)*sin(πt)
    float derivative = (PI / 2.0f) * sin(PI * t);           // 0..π/2..0
    derivative = max(0.05f, derivative);                      // floor to avoid ÷0
    float stepDelay  = baseDelay / derivative * 0.5f;         // scale to keep total time sane
    stepDelay = constrain(stepDelay, 2.0f, 80.0f);           // clamp: 2ms fast, 80ms very slow
    delay((int)round(stepDelay));
  }

  current = target;
}

void easedPan(int target, int spd = 0)  { easedMove(panServo,  currentPan,  target, spd); }
void easedTilt(int target, int spd = 0) { easedMove(tiltServo, currentTilt, target, spd); }

// ── Named movement animations ─────────────────────────────────────
// All use easedMove — each sub-move inherits the smooth curve

void doNod() {
  // Double-nod: confident, quick, snappy
  easedTilt(TILT_CENTER + 22, 8);
  delay(40);
  easedTilt(TILT_CENTER - 4, 8);
  delay(30);
  easedTilt(TILT_CENTER + 18, 8);
  delay(40);
  easedTilt(TILT_CENTER, 7);
}

void doNodFast() {
  easedTilt(TILT_CENTER + 18, 10);
  delay(20);
  easedTilt(TILT_CENTER, 10);
  delay(20);
  easedTilt(TILT_CENTER + 14, 10);
  delay(20);
  easedTilt(TILT_CENTER, 10);
}

void doShake() {
  // Slow, deliberate no-shake — weighted and human
  easedPan(PAN_CENTER - 30, 6);
  delay(80);
  easedPan(PAN_CENTER + 30, 6);
  delay(80);
  easedPan(PAN_CENTER - 22, 6);
  delay(60);
  easedPan(PAN_CENTER, 7);
}

void doReset() {
  // Settle gently back to neutral (both axes, tilt first)
  easedTilt(TILT_CENTER, 5);
  delay(30);
  easedPan(PAN_CENTER, 5);
}

void doLookUp()    { easedTilt(TILT_MIN + 5, 5); }
void doLookDown()  { easedTilt(TILT_MAX - 5, 5); }
void doLookLeft()  { easedPan(PAN_MIN + 10, 6); }
void doLookRight() { easedPan(PAN_MAX - 10, 6); }

void doCuriousTilt() {
  // Organic two-axis curious lean: pan and tilt offset simultaneously
  // Achieved by interleaving small steps of each axis
  int panTarget  = PAN_CENTER + 22;
  int tiltTarget = TILT_CENTER - 12;
  int panSteps   = 30;

  for (int i = 1; i <= panSteps; i++) {
    float t     = (float)i / panSteps;
    float eased = (1.0f - cos(PI * t)) / 2.0f;
    int np = PAN_CENTER  + (int)round((panTarget  - PAN_CENTER)  * eased);
    int nt = TILT_CENTER + (int)round((tiltTarget - TILT_CENTER) * eased);
    np = constrain(np, PAN_MIN, PAN_MAX);
    nt = constrain(nt, TILT_MIN, TILT_MAX);
    panServo.write(np);
    tiltServo.write(nt);
    currentPan  = np;
    currentTilt = nt;
    delay(10);
  }
  delay(700);
  // Return: both axes ease back together
  int fromPan  = currentPan;
  int fromTilt = currentTilt;
  for (int i = 1; i <= panSteps; i++) {
    float t     = (float)i / panSteps;
    float eased = (1.0f - cos(PI * t)) / 2.0f;
    int np = fromPan  + (int)round((PAN_CENTER  - fromPan)  * eased);
    int nt = fromTilt + (int)round((TILT_CENTER - fromTilt) * eased);
    np = constrain(np, PAN_MIN, PAN_MAX);
    nt = constrain(nt, TILT_MIN, TILT_MAX);
    panServo.write(np);
    tiltServo.write(nt);
    currentPan  = np;
    currentTilt = nt;
    delay(10);
  }
  currentPan  = PAN_CENTER;
  currentTilt = TILT_CENTER;
}

// ── Serial command parser ─────────────────────────────────────────
String inputBuffer = "";

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  // Status ping
  if (cmd == "S") {
    Serial.println("OK");
    return;
  }

  // SPEED<n>  →  set global speed 1-10
  if (cmd.startsWith("SPEED")) {
    int spd = constrain(cmd.substring(5).toInt(), 1, 10);
    globalSpeed = spd;
    Serial.print("SPEED:"); Serial.println(globalSpeed);
    return;
  }

  char type = cmd.charAt(0);

  if (type == 'P') {
    int angle = constrain(cmd.substring(1).toInt(), PAN_MIN, PAN_MAX);
    easedPan(angle);
    Serial.print("PAN:"); Serial.println(angle);
    return;
  }

  if (type == 'T') {
    int angle = constrain(cmd.substring(1).toInt(), TILT_MIN, TILT_MAX);
    easedTilt(angle);
    Serial.print("TILT:"); Serial.println(angle);
    return;
  }

  if (type == 'N') {
    String move = cmd.substring(1);
    move.toUpperCase();

    if      (move == "NOD")           doNod();
    else if (move == "NOD_FAST")      doNodFast();
    else if (move == "SHAKE")         doShake();
    else if (move == "RESET")         doReset();
    else if (move == "LOOK_UP")       doLookUp();
    else if (move == "LOOK_DOWN")     doLookDown();
    else if (move == "LOOK_LEFT")     doLookLeft();
    else if (move == "LOOK_RIGHT")    doLookRight();
    else if (move == "TILT_CURIOUS")  doCuriousTilt();
    else { Serial.print("UNKNOWN_MOVE:"); Serial.println(move); return; }

    Serial.print("DONE:"); Serial.println(move);
    return;
  }

  Serial.print("ERR:"); Serial.println(cmd);
}

// ── Setup ─────────────────────────────────────────────────────────
void setup() {
  Serial.begin(9600);
  panServo.attach(9);
  tiltServo.attach(10);

  // Boot to centre with a slow, settling ease
  panServo.write(PAN_CENTER);
  tiltServo.write(TILT_CENTER);
  currentPan  = PAN_CENTER;
  currentTilt = TILT_CENTER;

  delay(600);
  Serial.println("ADAM_SERVO_READY");
}

// ── Loop ──────────────────────────────────────────────────────────
void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }
}
