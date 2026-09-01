#!/usr/bin/env python3
"""ADAM Pi hardware test — pan servo + UART/ESP32-CAM link.

Exercises the REAL code paths (hardware.servo_pan + esp32_link.esp_link) so it
validates the code as well as the wiring. Run inside the venv on the Pi:
    GEMINI_API_KEY=dummy ~/adam/venv/bin/python ~/adam/adam_hwtest.py
"""
import os, sys, time

os.environ.setdefault("GEMINI_API_KEY", "dummy")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config
print("=== ADAM hardware test ===")
print(f"  UART {config.PI_UART_PORT} @ {config.PI_UART_BAUD} | servo GPIO {config.NECK_GPIO_PIN}")

# ── [1] Pan servo (GPIO 12) — gentle sweep, then detach ──────────────────
print("\n[1] Pan servo — gentle sweep around center (physical movement expected)")
import hardware
if hardware.pan_servo is None:
    print("  ❌ pan_servo is None — gpiozero pin factory unavailable")
else:
    for a in (90, 78, 90, 102, 90):          # center ± ~12°, end centered
        hardware.servo_pan(a)
        print(f"    servo_pan({a:3d})  gpiozero.angle={hardware.pan_servo.angle}")
        time.sleep(0.7)
    try:
        hardware.pan_servo.detach()           # stop holding torque / buzzing
        print("  ✅ sweep OK, servo detached (no idle buzz)")
    except Exception as e:
        print(f"  ⚠️  detach note: {e}")

# ── [2] UART / ESP32-CAM link ────────────────────────────────────────────
print("\n[2] UART / ESP32-CAM link")
from esp32_link import esp_link
esp_link.start()
if not esp_link.connected:
    print("  ❌ UART port did not open")
else:
    WAIT = 8
    print(f"  port open; listening {WAIT}s for ESP32 frames/touch/gesture...")
    time.sleep(WAIT)
    print(f"  receiving_data = {esp_link.receiving_data}")
    print(f"  frame_q={esp_link.frame_q.qsize()} "
          f"touch_q={esp_link.touch_q.qsize()} "
          f"gesture_q={esp_link.gesture_q.qsize()}")
    try:
        jpeg = esp_link.frame_q.get_nowait()
        print(f"  ✅ JPEG frame: {len(jpeg)} bytes "
              f"SOI={jpeg[:2].hex()} EOI={jpeg[-2:].hex()}")
    except Exception:
        print("  (no JPEG frame queued — ESP32 not sending or not wired)")
    esp_link.send_line("EMO:neutral")
    esp_link.send_line(f"TILT:{config.NECK_TILT_CENTER}")
    time.sleep(0.5)
    esp_link.stop()

print("\n=== hardware test done ===")
