"""
hardware.py — ADAM v40 servo & display actuators
==============================================================================
Thin actuator layer. Two physical output paths:

  • Pan servo — driven DIRECTLY by the Pi via gpiozero on GPIO 12 (hardware
    PWM). Initialized once at import; if gpiozero/pigpio isn't available the
    handle stays None and servo_pan() becomes a safe no-op (audio-only mode).
    Auto-DETACHES NECK_SERVO_HOLD_S after each move — see servo_pan() for the
    measurement that made that mandatory, and `servo_moving` for the event the
    audio path uses to ignore the mic while the head is actually turning.
  • Tilt servo + TFT face — driven INDIRECTLY: the Pi sends "TILT:<deg>" /
    "EMO:<emotion>" text lines over the UART to the ESP32-CAM, which relays
    them on to the Pico. So servo_tilt() and tft_set() are just esp_link
    sends, not GPIO.

Wiring constants (GPIO pin, pulse widths) come from config.py, verbatim from
the working build. tft_set() is the canonical definition; session re-exports
it so `from session import tft_set` keeps working for main.py.
"""

import threading

from config import (
    NECK_GPIO_PIN,
    NECK_SERVO_MIN_PW,
    NECK_SERVO_MAX_PW,
    NECK_SERVO_HOLD_S,
    NECK_SERVO_SETTLE_S,
)
from esp32_link import esp_link

pan_servo = None
try:
    from gpiozero import AngularServo
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        pan_servo = AngularServo(
            NECK_GPIO_PIN,
            initial_angle=None,          # Start DETACHED: gpiozero's default
                                         # initial_angle=0 emits a PWM pulse at
                                         # construction, snapping the servo to
                                         # center the instant this module is
                                         # imported (the boot-time jerk). None
                                         # sends NO pulse, so the servo stays
                                         # wherever it is until the first real
                                         # servo_pan() call (active tracking,
                                         # head gesture, or idle re-center).
            min_angle=-90, max_angle=90,
            min_pulse_width=NECK_SERVO_MIN_PW,
            max_pulse_width=NECK_SERVO_MAX_PW,
        )
    print(f"✅ Pan servo on GPIO {NECK_GPIO_PIN}")
except Exception as e:
    print(f"⚠️  Pan servo unavailable: {e}")


# Set while the pan servo is being driven, cleared once it has been released.
# The audio path reads this to keep servo noise out of the VAD's decisions: the
# noise burst while the head turns must neither open the mic gate nor be learned
# as the room's ambient level. Exposed as an Event (not a bool) so the listen
# loop can test it atomically from its own thread.
servo_moving = threading.Event()

_detach_timer: "threading.Timer | None" = None
_settle_timer: "threading.Timer | None" = None
_detach_lock = threading.Lock()


def _clear_moving() -> None:
    """Stop telling the audio path to distrust the mic."""
    global _settle_timer
    with _detach_lock:
        _settle_timer = None
    servo_moving.clear()


def _release_pan() -> None:
    """Stop driving PWM, then keep `servo_moving` set through the settle window.

    Detaching silences the coil, but NOT the microphones: measured with
    adam/_servodecay.py, the post-filter floor is still p50 1,697 (max 3,582) in
    the first second after detach() and p50 1,931 in the second, against a 1,039-
    1,245 baseline and an open_th of 1,800. Clearing the flag at detach time —
    which is what this used to do — handed the VAD roughly two seconds of
    servo-grade noise per head move and let the trackers learn it as the room.
    """
    global _detach_timer, _settle_timer
    with _detach_lock:
        _detach_timer = None
    try:
        if pan_servo is not None:
            pan_servo.detach()
    except Exception:
        pass
    if NECK_SERVO_SETTLE_S <= 0:
        servo_moving.clear()
        return
    with _detach_lock:
        if _settle_timer is not None:
            _settle_timer.cancel()
        _settle_timer = threading.Timer(NECK_SERVO_SETTLE_S, _clear_moving)
        _settle_timer.daemon = True
        _settle_timer.start()


def servo_pan(angle: int) -> None:
    """Move the pan servo, then RELEASE the pin NECK_SERVO_HOLD_S later.

    The release is not an optimisation, it is an audio fix. gpiozero holds the
    50 Hz pulse train indefinitely once `.angle` is assigned, so the servo stays
    energised and vibrates the board carrying both INMP441s. Re-measured with
    adam/_floorcal.py and adam/_servodecay.py, in the same post-filter RMS units
    the VAD gates on: p50 1,039-1,245 with the servo never pulsed, 4,658-4,666
    while it holds, and — the part that was missed first time round — still
    1,697 then 1,931 in the two seconds AFTER detach, before it returns to
    baseline. The gate opens at 1,800, so a holding servo alone pins it open
    forever, which is exactly how "ADAM speaks but never hears me" started:
    detached at boot (working), then the first head gesture attached it and the
    noise floor quadrupled for the rest of the session.

    Each call restarts the timer, so a burst of moves (DOA tracking, a gesture
    sequence) is one continuous hold followed by a single release, never a
    release part-way through the travel. A move arriving during the post-detach
    settle window also cancels that window's timer, so the flag stays set
    continuously instead of dropping between two nearby moves.
    """
    global _detach_timer, _settle_timer
    if pan_servo is None:
        return
    try:
        pan_servo.angle = max(-90, min(90, int(angle) - 90))
    except Exception:
        return
    if NECK_SERVO_HOLD_S <= 0:
        return                      # opt-out: keep torque, accept the noise
    servo_moving.set()
    with _detach_lock:
        if _settle_timer is not None:
            _settle_timer.cancel()
            _settle_timer = None
        if _detach_timer is not None:
            _detach_timer.cancel()
        _detach_timer = threading.Timer(NECK_SERVO_HOLD_S, _release_pan)
        _detach_timer.daemon = True
        _detach_timer.start()

def servo_tilt(angle: int) -> None:
    esp_link.send_line(f"TILT:{int(angle)}")

def tft_set(emotion: str) -> None:
    esp_link.send_line(f"EMO:{emotion}")
