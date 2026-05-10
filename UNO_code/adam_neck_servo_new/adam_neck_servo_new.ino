/*
  ADAM — Neck Servo Controller
  ==============================
  Hardware:
    Pin 9  → Pan servo  (MG995, left/right)
    Pin 10 → Tilt servo (MG995, up/down)

  Serial protocol (9600 baud, newline-terminated):
    P<angle>   →  Pan  to angle  (0–180, centre = 90)
    T<angle>   →  Tilt to angle  (0–180, centre = 90)
    N<move>    →  Named movement: NOD, SHAKE, RESET, LOOK_UP, LOOK_DOWN, LOOK_LEFT, LOOK_RIGHT, TILT_CURIOUS
    S          →  Status ping → replies "OK\n"

  Example commands:
    "P90\n"     → pan to centre
    "T70\n"     → tilt slightly up
    "NNOD\n"    → perform nod animation
*/

#include <Servo.h>

Servo panServo;
Servo tiltServo;

// ── Neutral / limit positions ────────────────────────────────────
const int PAN_MIN    = 30;
const int PAN_MAX    = 150;
const int PAN_CENTER = 90;

const int TILT_MIN    = 50;   // looking up limit
const int TILT_MAX    = 120;  // looking down limit
const int TILT_CENTER = 85;   // slight tilt forward looks natural

// Current positions
int currentPan  = PAN_CENTER;
int currentTilt = TILT_CENTER;

// ── Smooth move helper ────────────────────────────────────────────
void smoothMove(Servo &servo, int &current, int target, int stepDelay = 8) {
  target = constrain(target, 0, 180);
  if (current == target) return;
  int step = (current < target) ? 1 : -1;
  while (current != target) {
    current += step;
    servo.write(current);
    delay(stepDelay);
  }
}

void smoothPan(int target, int spd = 8)  { smoothMove(panServo,  currentPan,  target, spd); }
void smoothTilt(int target, int spd = 8) { smoothMove(tiltServo, currentTilt, target, spd); }

// ── Named movement animations ─────────────────────────────────────
void doNod() {
  // Yes-nod: tilt down twice quickly
  smoothTilt(TILT_CENTER + 20, 5);
  delay(60);
  smoothTilt(TILT_CENTER, 5);
  delay(60);
  smoothTilt(TILT_CENTER + 18, 5);
  delay(60);
  smoothTilt(TILT_CENTER, 5);
}

void doShake() {
  // No-shake: pan left-right twice
  smoothPan(PAN_CENTER - 28, 4);
  delay(60);
  smoothPan(PAN_CENTER + 28, 4);
  delay(60);
  smoothPan(PAN_CENTER - 20, 4);
  delay(60);
  smoothPan(PAN_CENTER, 5);
}

void doReset() {
  smoothTilt(TILT_CENTER, 6);
  smoothPan(PAN_CENTER, 6);
}

void doLookUp()    { smoothTilt(TILT_MIN + 5, 7); }
void doLookDown()  { smoothTilt(TILT_MAX - 5, 7); }
void doLookLeft()  { smoothPan(PAN_MIN + 10, 7); }
void doLookRight() { smoothPan(PAN_MAX - 10, 7); }

void doCuriousTilt() {
  // Tilt head sideways (use pan to simulate — combined effect)
  smoothTilt(TILT_CENTER - 10, 8);
  smoothPan(PAN_CENTER + 18, 8);
  delay(600);
  smoothTilt(TILT_CENTER, 8);
  smoothPan(PAN_CENTER, 8);
}

// ── Serial command parser ─────────────────────────────────────────
String inputBuffer = "";

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  char type = cmd.charAt(0);

  if (type == 'S') {
    Serial.println("OK");
    return;
  }

  if (type == 'P') {
    int angle = constrain(cmd.substring(1).toInt(), PAN_MIN, PAN_MAX);
    smoothPan(angle, 6);
    Serial.print("PAN:"); Serial.println(angle);
    return;
  }

  if (type == 'T') {
    int angle = constrain(cmd.substring(1).toInt(), TILT_MIN, TILT_MAX);
    smoothTilt(angle, 6);
    Serial.print("TILT:"); Serial.println(angle);
    return;
  }

  if (type == 'N') {
    String move = cmd.substring(1);
    move.toUpperCase();

    if      (move == "NOD")           doNod();
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

  // Boot to centre
  panServo.write(PAN_CENTER);
  tiltServo.write(TILT_CENTER);
  currentPan  = PAN_CENTER;
  currentTilt = TILT_CENTER;

  delay(500);
  Serial.println("ADAM_SERVO_READY");
}

// ── Loop ──────────────────────────────────────────────────────────
void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleCommand(inputBuffer);gfd    qqqqqqqqqqqqqq  
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }
}
