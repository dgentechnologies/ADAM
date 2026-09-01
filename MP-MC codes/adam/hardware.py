"""
hardware.py — ADAM v40 servo & display actuators
==============================================================================
Thin actuator layer. Two physical output paths:

  • Pan servo — driven DIRECTLY by the Pi via gpiozero on GPIO 12 (hardware
    PWM). Initialized once at import; if gpiozero/pigpio isn't available the
    handle stays None and servo_pan() becomes a safe no-op (audio-only mode).
  • Tilt servo + TFT face — driven INDIRECTLY: the Pi sends "TILT:<deg>" /
    "EMO:<emotion>" text lines over the UART to the ESP32-CAM, which relays
    them on to the Pico. So servo_tilt() and tft_set() are just esp_link
    sends, not GPIO.

Wiring constants (GPIO pin, pulse widths) come from config.py, verbatim from
the working build. tft_set() is the canonical definition; session re-exports
it so `from session import tft_set` keeps working for main.py.
"""

from config import NECK_GPIO_PIN, NECK_SERVO_MIN_PW, NECK_SERVO_MAX_PW
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


def servo_pan(angle: int) -> None:
    if pan_servo is None:
        return
    try:
        pan_servo.angle = max(-90, min(90, int(angle) - 90))
    except Exception:
        pass

def servo_tilt(angle: int) -> None:
    esp_link.send_line(f"TILT:{int(angle)}")

def tft_set(emotion: str) -> None:
    esp_link.send_line(f"EMO:{emotion}")
