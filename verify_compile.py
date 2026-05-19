#!/usr/bin/env python3
"""
Verify ESP32 LVGL sketch compiles using Arduino IDE command line (hidden mode)
"""
import subprocess
import sys
import os
import json
from pathlib import Path

SKETCH_PATH = r"d:\Dgen Technologies Pvt. Ltd\ADAM\UNO_code\picopixel_lvgl_ui\picopixel_lvgl_ui.ino"
BOARD_FQBN = "esp32:esp32:esp32"
BUILD_DIR = r"d:\temp\arduino_build"

def run_arduino_verify():
    """Run Arduino IDE compile verification"""
    os.makedirs(BUILD_DIR, exist_ok=True)
    
    # Try using Arduino IDE command line if available
    cmd = [
        "arduino-cli",
        "compile",
        "--fqbn", BOARD_FQBN,
        "--build-path", BUILD_DIR,
        SKETCH_PATH
    ]
    
    print(f"Attempting: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print("STDOUT:")
        print(result.stdout[:2000])
        print("\nSTDERR:")
        print(result.stderr[:2000])
        print(f"\nReturn code: {result.returncode}")
        return result.returncode == 0
    except FileNotFoundError:
        print("arduino-cli not found. Trying Arduino IDE (GUI)...")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = run_arduino_verify()
    if success is None:
        print("\nUse Arduino IDE GUI to verify compilation:")
        print(f"  1. Open: {SKETCH_PATH}")
        print(f"  2. Select Board: {BOARD_FQBN}")
        print("  3. Press Ctrl+R to compile")
    elif success:
        print("\n✓ Compilation successful!")
        sys.exit(0)
    else:
        print("\n✗ Compilation failed!")
        sys.exit(1)
