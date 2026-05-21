"""
ADAM — Autonomous Desktop AI Module (v30 Pi Wi-Fi Integration)
================================================================
- Connects to ESP32-CAM over Wi-Fi for Video, LiDAR, and Touch.
- Controls Pan Servo directly via Pi Hardware PWM (GPIO 12).
- Drives ILI9341 2.42" TFT directly via SPI.
- Native I2S Audio: 
    * 2x INMP441 Mics (Stereo -> GCC-PHAT Sound Localization -> Mono)
    * 1x MAX98357A Amp (Mono out)
"""

import asyncio
import os
import time
import datetime
import json
import threading
import struct
import queue
import random
import re
import warnings
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import pyaudio
import aiohttp
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Hardware Specific Imports ─────────────────────────────────────────────────
from gpiozero import AngularServo
from adam_tft import TFTEmotionRenderer  # Ensure adam_tft.py is in same folder

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

try:
    import webrtcvad as _webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False

DDGS = None
try:
    from duckduckgo_search import DDGS as _DDGS
    DDGS = _DDGS
except ImportError:
    DDGS = None

DDG_AVAILABLE = DDGS is not None

# ── Environment & Network Setup ───────────────────────────────────────────────
load_dotenv(dotenv_path=".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file.")

ESP32_IP = os.getenv("ESP32_IP")
if not ESP32_IP:
    raise ValueError("❌ ESP32_IP not found in .env file. Add it (e.g. ESP32_IP=192.168.1.100)")

print("✅ Environment loaded. Connecting to ESP32 at:", ESP32_IP)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — ALL TUNABLE PARAMETERS (EDIT HERE ONLY)
# ═════════════════════════════════════════════════════════════════════════════

# AI Model Configuration
LIVE_MODEL          = "gemini-3.1-flash-live-preview"
GEN_MODEL_CASCADE   = ["gemini-3.1-flash-lite-preview", "gemini-3.1-flash-live-preview"]
GEN_RETRIES         = 2
VOICE               = "Charon"

# System Instruction
ADAM_SYSTEM_INSTRUCTION = "You are ADAM. Keep answers short and natural."

# Audio Configuration
POST_SPEECH_MUTE_S  = 0.4
FORMAT              = pyaudio.paInt16
MIC_CHANNELS        = 2              # CRITICAL: 2 Mics = Stereo input
OUT_CHANNELS        = 1              # Amp is mono
SEND_SAMPLE_RATE    = 16000          # Hz
RECV_SAMPLE_RATE    = 24000          # Hz
CHUNK_SIZE          = 512            # Frames per buffer
VAD_FRAME_MS        = 20             # Voice Activity Detection frame size (ms)
VAD_WINDOW          = 10             # VAD voting window size (frames)

# Asyncio Queue Sizes
MIC_QUEUE_MAXSIZE   = 120            # Microphone input queue
OUT_QUEUE_MAXSIZE   = 200            # Audio output queue

# Camera Configuration
CAMERA_FPS_INTERVAL = 1.0            # Seconds between frame sends
DETECT_W            = 320            # Face detection width
DETECT_H            = 240            # Face detection height
FACE_DETECT_MIN_NEIGHBORS = 4        # Haar cascade min neighbors
CAMERA_STREAM_URL   = f"http://{{ESP32_IP}}/stream"  # Template
CAMERA_OPEN_FAIL_SLEEP_S = 0.1       # Retry interval if stream not ready

# Attention System
ATTENTION_TIMEOUT_S = 30             # Seconds before passive attention timeout
GAZE_AWAY_DEACTIVATE_S = 2.0         # Seconds to stay passive after gaze away
ENABLE_IDLE         = True           # Enable idle behavior
IDLE_TIMEOUT_S      = 90             # Idle state timeout

# Physical Neck (Pan Servo)
NECK_GPIO_PIN       = 12             # GPIO pin for servo
NECK_SERVO_MIN_PULSE = 0.0005        # Min pulse width (seconds)
NECK_SERVO_MAX_PULSE = 0.0025        # Max pulse width (seconds)
NECK_PAN_CENTER     = 90             # Pan center angle (degrees)
NECK_TILT_CENTER    = 85             # Tilt center angle (degrees)
NECK_PAN_MIN        = 30             # Pan min angle
NECK_PAN_MAX        = 150            # Pan max angle
NECK_TILT_MIN       = 50             # Tilt min angle
NECK_TILT_MAX       = 120            # Tilt max angle
NECK_PAN_DEADZONE   = 5              # Pan deadzone (degrees)
NECK_TILT_DEADZONE  = 6              # Tilt deadzone (degrees)
NECK_TRACK_INTERVAL = 0.12           # Tracking update interval (seconds)
NECK_MAX_STEP       = 3              # Max step per update (degrees)
NECK_RECENTER_HOLD_S = 1.5           # Hold recentering (seconds)
NECK_FACE_TRACK_ALPHA_PAN  = 0.32    # Face tracking smoothing (pan)
NECK_FACE_TRACK_ALPHA_TILT = 0.18    # Face tracking smoothing (tilt)
NECK_IDLE_START_AFTER_S = 4.0        # Idle behavior delay (seconds)
NECK_SMOOTH_ALPHA   = 0.3            # Servo movement smoothing factor
NECK_TILT_WIFI_TIMEOUT_S = 1.0       # Tilt WiFi command timeout

# Sound Localization (Direction of Arrival)
MIC_DISTANCE_M      = 0.065          # Distance between INMP441 mics (meters, 65mm)
DOA_ANGLE_DEADZONE  = 10             # Audio DoA deadzone (degrees)
SOUND_SPEED_MPS     = 343.0          # Speed of sound (m/s)
AUDIO_ACTIVE_WINDOW_S = 2.0          # Window to track active audio (seconds)

# ESP32 Communication
ESP32_SENSORS_URL   = f"http://{{ESP32_IP}}/sensors"  # Template
ESP32_SENSOR_TIMEOUT_S = 0.8         # Sensor poll timeout
ESP32_SENSOR_POLL_INTERVAL_S = 0.2   # Sensor poll interval
ESP32_TOUCH_DEFAULT = [0, 0, 0, 0]   # Default touch sensor array
ESP32_TOUCH_EMOTION = "happy"         # Emotion on touch

# Vision Processing Smoothing
NECK_AUDIO_SNAP_DELAY_S = 2.0        # Delay before snapping to audio direction

# Task Processing
TASK_PROCESS_INTERVAL_MS = 10         # Sleep interval in main loops (ms)
SEND_FAIL_RETRY_SLEEP_S = 0.01       # Retry interval on send failure
SPEAKER_TIMEOUT_S   = 0.3            # Speaker output timeout
TFT_EMOTION_SPEAKING = "speaking"     # TFT emotion on speech start

# Wake Words & Models
WAKE_WORDS          = ["adam", "hey adam", "ok adam", "okay adam"]
VOSK_MODEL_PATH     = "vosk-model-small-en-in-0.4"

# Memory Files
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE         = Path(BASE_DIR) / "adam_memory.json"
FACE_MEMORY_FILE    = Path(BASE_DIR) / "adam_faces.json"
CONV_MEMORY_FILE    = Path(BASE_DIR) / "adam_conversations.json"

# Conversation Memory
CONV_MAX_TURNS      = 40             # Max conversation turns
CONV_PROMPT_TURNS   = 20             # Turns before prompt reset

# Search Engine Config
SEARCH_CACHE_TTL_S  = 1800           # Cache TTL (seconds)
SEARCH_MIN_GAP_S    = 5.0            # Min gap between searches (seconds)

# Global State
_ddg_cache: dict[str, tuple[str, float]] = {}
_last_ddg_t: float = 0.0
global_audio_angle = [0.0]            # Current audio DoA angle
global_audio_active = [0.0]           # Last time audio was active

# ═════════════════════════════════════════════════════════════════════════════
# HARDWARE ABSTRACTION & INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════════

print("🤖 ADAM v30 Pi Wi-Fi Integration Starting...")

tft_renderer = None
current_tft_emotion = "idle"
pan_servo = None

try:
    print("📺 Initializing TFT renderer...")
    tft_renderer = TFTEmotionRenderer()
    print("✅ TFT renderer initialized")
except Exception as e:
    print(f"❌ TFT renderer init failed: {type(e).__name__}: {e}")
    tft_renderer = None

try:
    print(f"🦾 Initializing pan servo on GPIO {NECK_GPIO_PIN}...")
    pan_servo = AngularServo(
        NECK_GPIO_PIN, 
        min_angle=-90, 
        max_angle=90, 
        min_pulse_width=NECK_SERVO_MIN_PULSE, 
        max_pulse_width=NECK_SERVO_MAX_PULSE
    )
    print(f"✅ Pan servo initialized on GPIO {NECK_GPIO_PIN}")
except Exception as e:
    pan_servo = None
    print(f"❌ Pan servo init failed: {type(e).__name__}: {e}")

try:
    print("🎤 Initializing PyAudio...")
    pya = pyaudio.PyAudio()
    print("✅ PyAudio initialized")
except Exception as e:
    pya = None
    print(f"⚠️  PyAudio init failed: {type(e).__name__}: {e}")
    print("⚠️  ADAM will run without audio input/output (camera and servo still work)")

def neck_is_ready() -> bool:
    """Check if neck hardware is ready."""
    return pan_servo is not None

def pan(angle: int, speed: int = None) -> None:
    """Pan the servo to specified angle."""
    if not pan_servo:
        print(f"⚠️  Pan servo not available, skipping pan({angle})")
        return
    try:
        mapped_angle = max(-90, min(90, int(angle) - 90))
        pan_servo.angle = mapped_angle
        print(f"🦾 Pan servo → {angle}°")
    except Exception as e:
        print(f"❌ Pan error: {type(e).__name__}: {e}")

async def set_tilt_wifi(angle: int) -> None:
    """Send tilt command to ESP32 via WiFi."""
    url = f"http://{ESP32_IP}/tilt?angle={angle}"
    async with aiohttp.ClientSession() as session:
        try:
            await session.get(url, timeout=NECK_TILT_WIFI_TIMEOUT_S)
            print(f"🦾 Tilt WiFi command sent → {angle}°")
        except asyncio.TimeoutError:
            print(f"⚠️  Tilt WiFi timeout: {url}")
        except Exception as e:
            print(f"❌ Tilt WiFi error: {type(e).__name__}: {e}")

def tilt(angle: int, speed: int = None) -> None:
    """Queue tilt command to ESP32."""
    try:
        asyncio.create_task(set_tilt_wifi(angle))
    except Exception as e:
        print(f"❌ Tilt task creation failed: {type(e).__name__}: {e}")

def reset_neck() -> None:
    """Reset neck to center position."""
    print("🦾 Resetting neck to center...")
    pan(NECK_PAN_CENTER)
    tilt(NECK_TILT_CENTER)

def close_neck() -> None:
    """Close and cleanup neck hardware."""
    print("🦾 Closing neck hardware...")
    reset_neck()

# ═════════════════════════════════════════════════════════════════════════════
# SOUND LOCALIZATION (Direction of Arrival - GCC-PHAT)
# ═════════════════════════════════════════════════════════════════════════════

def get_doa_angle(left_channel: np.ndarray, right_channel: np.ndarray, 
                  sample_rate: int = SEND_SAMPLE_RATE, 
                  mic_dist: float = MIC_DISTANCE_M) -> float:
    """
    Calculates the angle of the sound source using Generalized Cross-Correlation (GCC-PHAT).
    Returns angle in degrees. Negative = Left, Positive = Right.
    """
    try:
        N = 1024
        L = np.fft.rfft(left_channel, n=N)
        R = np.fft.rfft(right_channel, n=N)
        
        R_cross = L * np.conj(R)
        cc = np.fft.irfft(R_cross / (np.abs(R_cross) + 1e-15))
        
        # Calculate max possible sample shift based on physical mic distance
        max_shift = int(sample_rate * mic_dist / SOUND_SPEED_MPS) + 1
        
        # Center the correlation
        cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
        shift = np.argmax(cc) - max_shift
        
        # Convert shift directly to angle
        val = (shift / sample_rate) * SOUND_SPEED_MPS / mic_dist
        val = np.clip(val, -1.0, 1.0)
        angle_deg = np.degrees(np.arcsin(val))
        
        print(f"📍 DoA calculated: {angle_deg:.1f}°")
        return angle_deg
    except Exception as e:
        print(f"❌ DoA calculation failed: {type(e).__name__}: {e}")
        return 0.0

# ═════════════════════════════════════════════════════════════════════════════
# VISION TRACKER
# ═════════════════════════════════════════════════════════════════════════════

class PersonTracker:
    """Multi-person face detection and tracking."""
    
    def __init__(self) -> None:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)
            self._available = not self._cascade.empty()
            if self._available:
                print(f"✅ Face cascade loaded: {cascade_path}")
            else:
                print(f"❌ Face cascade failed to load from: {cascade_path}")
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            print("✅ CLAHE contrast limiter initialized")
        except Exception as e:
            print(f"❌ PersonTracker init failed: {type(e).__name__}: {e}")
            self._available = False
            self._cascade = None
            self._clahe = None
        self._adam_speaking: bool = False

    def process_frame(self, frame: np.ndarray, vad_active: bool = True) -> dict:
        """Process frame for face detection."""
        result: dict = {"faces": [], "active_speaker_idx": None, "face_count": 0}
        
        if not self._available or frame is None:
            return result
        
        try:
            h_f, w_f = frame.shape[:2]
            gray_small = self._clahe.apply(
                cv2.cvtColor(cv2.resize(frame, (DETECT_W, DETECT_H)), cv2.COLOR_BGR2GRAY)
            )

            raw_det = self._cascade.detectMultiScale(
                gray_small, 
                scaleFactor=1.2, 
                minNeighbors=FACE_DETECT_MIN_NEIGHBORS, 
                minSize=(20, 20)
            )

            scale_x = w_f / DETECT_W
            scale_y = h_f / DETECT_H

            faces = [
                (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)) 
                for (x, y, w, h) in raw_det
            ]
            result["face_count"] = len(faces)
            
            if faces:
                print(f"📷 Detected {len(faces)} face(s)")
            
            for idx, (fx, fy, fw, fh) in enumerate(faces):
                cx, cy = (fx + fw / 2) / w_f, (fy + fh / 2) / h_f
                facing_camera = (fw / max(fh, 1) > 0.65)
                result["faces"].append({
                    "id": idx, 
                    "cx_norm": cx, 
                    "cy_norm": cy, 
                    "facing_camera": facing_camera
                })
            
            return result
        except Exception as e:
            print(f"❌ Frame processing failed: {type(e).__name__}: {e}")
            return result

    def build_context(self, tr: dict) -> str:
        """Build text context from vision results."""
        count = tr["face_count"]
        if count == 0:
            return "[CAMERA: No faces in frame.]"
        parts = []
        for i, f in enumerate(tr["faces"]):
            pos = "left" if f["cx_norm"] < 0.40 else "right" if f["cx_norm"] > 0.60 else "centre"
            facing = '→CAM' if f['facing_camera'] else '→AWAY'
            parts.append(f"P{i+1}({pos},{facing})")
        return f"[CAMERA: {count} people — {', '.join(parts)}.]"

    def set_adam_speaking(self, flag: bool) -> None:
        """Set flag for ADAM speaking state."""
        self._adam_speaking = flag

# ═════════════════════════════════════════════════════════════════════════════
# AUDIO VOICE ACTIVITY DETECTOR
# ═════════════════════════════════════════════════════════════════════════════

class VoiceActivityDetector:
    """Detects voice activity using WebRTC VAD."""
    
    def __init__(self) -> None:
        self._ready = False
        self._vad = None
        self._buf = b""
        self._votes: deque = deque(maxlen=VAD_WINDOW)
        self._frame_bytes = int(SEND_SAMPLE_RATE * (VAD_FRAME_MS / 1000.0)) * 2
        
        if WEBRTCVAD_AVAILABLE:
            try:
                self._vad = _webrtcvad.Vad(2)
                self._ready = True
                print("✅ Voice Activity Detector initialized (WebRTC VAD)")
            except Exception as e:
                print(f"⚠️  VAD init failed: {type(e).__name__}: {e}")
                self._ready = False
        else:
            print("⚠️  WebRTC VAD not available - voice detection disabled")
    
    def feed(self, pcm_chunk: bytes) -> None:
        """Feed PCM chunk to VAD."""
        if not self._ready:
            return
        
        try:
            self._buf += pcm_chunk
            while len(self._buf) >= self._frame_bytes:
                frame = self._buf[:self._frame_bytes]
                self._buf = self._buf[self._frame_bytes:]
                
                is_speech = 1 if self._vad.is_speech(frame, SEND_SAMPLE_RATE) else 0
                self._votes.append(is_speech)
        except Exception as e:
            print(f"❌ VAD feed error: {type(e).__name__}: {e}")
    
    def is_active(self) -> bool:
        """Check if voice is currently active."""
        if not self._ready or not self._votes:
            return True
        return sum(self._votes) > len(self._votes) // 2

class AttentionState:
    """Attention state constants."""
    PASSIVE = "passive"
    ATTENTIVE = "attentive"
    RESPONDING = "responding"

class AttentionManager:
    """Manages ADAM's attention state."""
    
    def __init__(self) -> None:
        self._state = AttentionState.PASSIVE
        self._last_t = 0.0
        self._lock = asyncio.Lock()
        print("✅ Attention Manager initialized")
    
    def is_active(self) -> bool:
        """Check if attention is currently active."""
        if self._state == AttentionState.ATTENTIVE:
            if time.time() - self._last_t > ATTENTION_TIMEOUT_S:
                self._state = AttentionState.PASSIVE
                print(f"🔔 Attention timeout: {ATTENTION_TIMEOUT_S}s")
                return False
            return True
        return False
    
    async def activate(self, reason: str = "") -> None:
        """Activate attention with optional reason."""
        async with self._lock:
            if self._state == AttentionState.RESPONDING:
                return
            old_state = self._state
            self._state = AttentionState.ATTENTIVE
            self._last_t = time.time()
            if old_state != AttentionState.ATTENTIVE:
                print(f"🔔 Attention activated: {reason}")
    
    async def set_responding(self, on: bool) -> None:
        """Set responding state."""
        async with self._lock:
            old_state = self._state
            self._state = AttentionState.RESPONDING if on else AttentionState.ATTENTIVE
            if not on:
                self._last_t = time.time()
            if old_state != self._state:
                status = "RESPONDING" if on else "LISTENING"
                print(f"🔔 Attention → {status}")
    
    def touch(self) -> None:
        """Update last activity timestamp."""
        if self._state in (AttentionState.ATTENTIVE, AttentionState.RESPONDING):
            self._last_t = time.time()

# ═════════════════════════════════════════════════════════════════════════════
# COMPONENT AVAILABILITY FLAGS (GRACEFUL DEGRADATION)
# ═════════════════════════════════════════════════════════════════════════════

# Track which components are available for graceful degradation
COMPONENTS_STATUS = {
    "camera": pan_servo is not None,
    "microphone": pya is not None,
    "tft": tft_renderer is not None,
    "servo": pan_servo is not None,
}

print("\n" + "📊 "*20)
print("COMPONENT AVAILABILITY STATUS:")
for component, available in COMPONENTS_STATUS.items():
    status = "✅ READY" if available else "⚠️  UNAVAILABLE"
    print(f"  {component.upper():<15} {status}")
print("📊 "*20 + "\n")

# ═════════════════════════════════════════════════════════════════════════════
# RUN SESSION - MAIN GEMINI LIVE CONNECTION
# ═════════════════════════════════════════════════════════════════════════════

async def run_session(client, resume_handle: str | None, stop: asyncio.Event, 
                     out_q: asyncio.Queue, attention: AttentionManager, 
                     vad: VoiceActivityDetector, tracker: PersonTracker) -> str | None:
    """Run a Gemini Live session with all I/O tasks."""
    print(f"🌐 Connecting to Gemini Live (model: {LIVE_MODEL})...")
    
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=ADAM_SYSTEM_INSTRUCTION,
        session_resumption=types.SessionResumptionConfig(handle=resume_handle),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
            )
        )
    )

    latest_handle: str | None = resume_handle

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            print("✅ Gemini Live session connected")
            
            mic_q = asyncio.Queue(maxsize=MIC_QUEUE_MAXSIZE)
            adam_speaking = asyncio.Event()

            async def esp32_sensor_poller() -> None:
                """Poll ESP32 touch sensors."""
                print("📡 ESP32 sensor poller started")
                url = ESP32_SENSORS_URL.format(ESP32_IP=ESP32_IP)
                async with aiohttp.ClientSession() as http_session:
                    try:
                        while not stop.is_set():
                            try:
                                async with http_session.get(url, timeout=ESP32_SENSOR_TIMEOUT_S) as resp:
                                    data = await resp.json()
                                    touch_data = data.get("touch", ESP32_TOUCH_DEFAULT)
                                    if any(t == 1 for t in touch_data):
                                        print("✋ Touch detected!")
                                        if tft_renderer:
                                            tft_renderer.set_emotion(ESP32_TOUCH_EMOTION)
                                        await attention.activate("esp32-touch")
                            except asyncio.TimeoutError:
                                pass  # Timeout is expected
                            except Exception as e:
                                print(f"⚠️  Sensor poll error: {type(e).__name__}: {e}")
                            await asyncio.sleep(ESP32_SENSOR_POLL_INTERVAL_S)
                    except asyncio.CancelledError:
                        print("📡 ESP32 sensor poller cancelled")
                    except Exception as e:
                        print(f"❌ Sensor poller failed: {type(e).__name__}: {e}")

            async def camera() -> None:
                """Capture frames from ESP32 camera and update neck servo."""
                print("📷 Camera task started")
                camera_url = CAMERA_STREAM_URL.format(ESP32_IP=ESP32_IP)
                cap = None
                camera_failed = False
                
                try:
                    try:
                        cap = cv2.VideoCapture(camera_url)
                        print(f"📷 Attempting to connect to camera: {camera_url}")
                    except Exception as e:
                        print(f"❌ Camera initialization failed: {type(e).__name__}: {e}")
                        camera_failed = True
                        cap = None
                    
                    last_sent = 0.0
                    curr_pan = float(NECK_PAN_CENTER)
                    curr_tilt = float(NECK_TILT_CENTER)
                    consecutive_frame_errors = 0
                    MAX_CONSECUTIVE_ERRORS = 5
                    
                    while not stop.is_set():
                        try:
                            # Skip if camera failed
                            if camera_failed or cap is None:
                                if not camera_failed:
                                    print("⚠️  Camera not available - skipping frame capture")
                                    camera_failed = True
                                await asyncio.sleep(1.0)  # Retry every 1 second
                                continue
                            
                            # Try to read frame
                            try:
                                raw = await asyncio.to_thread(
                                    lambda: cap.read()[1] if cap.isOpened() else None
                                )
                            except Exception as e:
                                print(f"⚠️  Camera read error: {type(e).__name__}: {e}")
                                consecutive_frame_errors += 1
                                raw = None
                            
                            # Check for consecutive errors
                            if raw is None:
                                consecutive_frame_errors += 1
                                if consecutive_frame_errors > MAX_CONSECUTIVE_ERRORS:
                                    print(f"❌ Camera: {MAX_CONSECUTIVE_ERRORS} consecutive errors - disabling camera")
                                    camera_failed = True
                                    if cap:
                                        cap.release()
                                        cap = None
                                await asyncio.sleep(CAMERA_OPEN_FAIL_SLEEP_S)
                                continue
                            
                            # Reset error counter on successful read
                            consecutive_frame_errors = 0
                            
                            now = time.time()
                            if now - last_sent >= CAMERA_FPS_INTERVAL:
                                try:
                                    tr = await asyncio.to_thread(
                                        tracker.process_frame, raw, vad.is_active()
                                    )
                                    
                                    # Check for face gaze
                                    try:
                                        if any(f["facing_camera"] for f in tr["faces"]):
                                            await attention.activate("face-gaze-detected")
                                    except Exception as e:
                                        print(f"⚠️  Face detection error: {type(e).__name__}: {e}")
                                    
                                    # Determine neck target angle
                                    try:
                                        if tr.get("faces"):
                                            # Face visible - look at first face
                                            f = tr["faces"][0]
                                            aim_pan = float(
                                                NECK_PAN_MAX - f["cx_norm"] * (NECK_PAN_MAX - NECK_PAN_MIN)
                                            )
                                            aim_tilt = float(
                                                NECK_TILT_MIN + f["cy_norm"] * (NECK_TILT_MAX - NECK_TILT_MIN)
                                            )
                                        elif now - global_audio_active[0] < NECK_AUDIO_SNAP_DELAY_S:
                                            # No face, but sound detected - look at sound direction
                                            aim_pan = float(NECK_PAN_CENTER + global_audio_angle[0])
                                            aim_pan = max(NECK_PAN_MIN, min(NECK_PAN_MAX, aim_pan))
                                            aim_tilt = float(NECK_TILT_CENTER)
                                        else:
                                            # Idle - return to center
                                            aim_pan = float(NECK_PAN_CENTER)
                                            aim_tilt = float(NECK_TILT_CENTER)
                                        
                                        # Smooth movement
                                        curr_pan += (aim_pan - curr_pan) * NECK_SMOOTH_ALPHA
                                        curr_tilt += (aim_tilt - curr_tilt) * NECK_SMOOTH_ALPHA
                                        
                                        # Apply servo commands (with error handling)
                                        try:
                                            await asyncio.to_thread(pan, int(curr_pan))
                                        except Exception as e:
                                            print(f"⚠️  Pan servo error: {type(e).__name__}: {e}")
                                        
                                        try:
                                            tilt(int(curr_tilt))
                                        except Exception as e:
                                            print(f"⚠️  Tilt servo error: {type(e).__name__}: {e}")
                                        
                                    except Exception as e:
                                        print(f"⚠️  Servo control error: {type(e).__name__}: {e}")
                                    
                                    last_sent = now
                                
                                except Exception as e:
                                    print(f"⚠️  Frame processing error: {type(e).__name__}: {e}")
                            
                            await asyncio.sleep(TASK_PROCESS_INTERVAL_MS / 1000.0)
                        
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            print(f"⚠️  Camera loop error: {type(e).__name__}: {e}")
                            await asyncio.sleep(CAMERA_OPEN_FAIL_SLEEP_S)
                
                except asyncio.CancelledError:
                    print("📷 Camera task cancelled")
                except Exception as e:
                    print(f"❌ Camera task failed: {type(e).__name__}: {e}")
                finally:
                    try:
                        if cap:
                            cap.release()
                            print("📷 Camera released")
                    except Exception as e:
                        print(f"⚠️  Camera release error: {type(e).__name__}: {e}")
                    print("📷 Camera task ended")

            async def listen() -> None:
                """Capture stereo audio from microphone."""
                print("🎤 Listen task started")
                stream = None
                microphone_failed = False
                consecutive_mic_errors = 0
                MAX_CONSECUTIVE_MIC_ERRORS = 10
                
                try:
                    # Try to open microphone stream
                    try:
                        if pya is None:
                            raise RuntimeError("PyAudio not initialized")
                        
                        stream = pya.open(
                            format=FORMAT,
                            channels=MIC_CHANNELS,
                            rate=SEND_SAMPLE_RATE,
                            input=True,
                            frames_per_buffer=CHUNK_SIZE
                        )
                        print("✅ Microphone stream opened")
                    except Exception as e:
                        print(f"❌ Microphone initialization failed: {type(e).__name__}: {e}")
                        microphone_failed = True
                        stream = None
                    
                    while not stop.is_set():
                        try:
                            # Skip if microphone failed
                            if microphone_failed or stream is None:
                                if not microphone_failed:
                                    print("⚠️  Microphone not available - skipping audio capture")
                                    microphone_failed = True
                                await asyncio.sleep(1.0)  # Retry every 1 second
                                continue
                            
                            try:
                                raw_data = await asyncio.to_thread(
                                    stream.read, CHUNK_SIZE, exception_on_overflow=False
                                )
                            except Exception as e:
                                print(f"⚠️  Microphone read error: {type(e).__name__}: {e}")
                                consecutive_mic_errors += 1
                                if consecutive_mic_errors > MAX_CONSECUTIVE_MIC_ERRORS:
                                    print(f"❌ Microphone: {MAX_CONSECUTIVE_MIC_ERRORS} consecutive errors - disabling microphone")
                                    microphone_failed = True
                                    if stream:
                                        try:
                                            stream.stop_stream()
                                            stream.close()
                                        except:
                                            pass
                                        stream = None
                                await asyncio.sleep(1.0)
                                continue
                            
                            # Reset error counter on successful read
                            consecutive_mic_errors = 0
                            
                            try:
                                # Process stereo audio
                                stereo_samples = np.frombuffer(raw_data, dtype=np.int16)
                                
                                # Handle different channel counts gracefully
                                if len(stereo_samples) == 0:
                                    continue
                                
                                try:
                                    left_mic = stereo_samples[0::2]
                                    right_mic = stereo_samples[1::2]
                                except Exception as e:
                                    print(f"⚠️  Audio channel extraction error: {type(e).__name__}: {e}")
                                    continue
                                
                                # Send mono (left channel) to voice AI
                                try:
                                    mono_data = left_mic.tobytes()
                                    vad.feed(mono_data)
                                except Exception as e:
                                    print(f"⚠️  Audio processing error: {type(e).__name__}: {e}")
                                    continue
                                
                                # Calculate sound direction if speaking
                                try:
                                    if vad.is_active() and not adam_speaking.is_set():
                                        angle = get_doa_angle(
                                            left_mic.astype(np.float32),
                                            right_mic.astype(np.float32)
                                        )
                                        if abs(angle) > DOA_ANGLE_DEADZONE:
                                            global_audio_angle[0] = angle
                                            global_audio_active[0] = time.time()
                                            await attention.activate("sound-detected")
                                except Exception as e:
                                    print(f"⚠️  DoA processing error: {type(e).__name__}: {e}")
                                
                                # Queue audio data
                                try:
                                    if not mic_q.full():
                                        mic_q.put_nowait(mono_data)
                                except Exception as e:
                                    print(f"⚠️  Queue error: {type(e).__name__}: {e}")
                            
                            except Exception as e:
                                print(f"⚠️  Audio frame processing error: {type(e).__name__}: {e}")
                            
                            await asyncio.sleep(TASK_PROCESS_INTERVAL_MS / 1000.0)
                        
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            print(f"⚠️  Listen loop error: {type(e).__name__}: {e}")
                            await asyncio.sleep(1.0)
                
                except asyncio.CancelledError:
                    print("🎤 Listen task cancelled")
                except Exception as e:
                    print(f"❌ Listen task failed: {type(e).__name__}: {e}")
                finally:
                    try:
                        if stream:
                            stream.stop_stream()
                            stream.close()
                            print("🎤 Microphone stream closed")
                    except Exception as e:
                        print(f"⚠️  Microphone close error: {type(e).__name__}: {e}")
                    print("🎤 Listen task ended")

            async def send() -> None:
                """Send audio to Gemini."""
                print("📤 Send task started")
                consecutive_send_errors = 0
                MAX_CONSECUTIVE_SEND_ERRORS = 10
                
                try:
                    while not stop.is_set():
                        try:
                            try:
                                chunk = await asyncio.wait_for(mic_q.get(), timeout=1.0)
                            except asyncio.TimeoutError:
                                continue
                            
                            # Gate: don't send if ADAM is speaking or not attentive
                            if adam_speaking.is_set() or not attention.is_active():
                                continue
                            
                            try:
                                await session.send_realtime_input(
                                    audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                                )
                                consecutive_send_errors = 0
                            
                            except Exception as e:
                                print(f"⚠️  Send error: {type(e).__name__}: {e}")
                                consecutive_send_errors += 1
                                
                                if consecutive_send_errors > MAX_CONSECUTIVE_SEND_ERRORS:
                                    print(f"❌ Send: {MAX_CONSECUTIVE_SEND_ERRORS} consecutive errors - continuing")
                                    consecutive_send_errors = 0  # Reset to avoid spam
                                
                                await asyncio.sleep(SEND_FAIL_RETRY_SLEEP_S)
                        
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            print(f"⚠️  Send loop error: {type(e).__name__}: {e}")
                            await asyncio.sleep(0.5)
                
                except asyncio.CancelledError:
                    print("📤 Send task cancelled")
                except Exception as e:
                    print(f"❌ Send task failed: {type(e).__name__}: {e}")
                finally:
                    print("📤 Send task ended")

            async def receive() -> None:
                """Receive Gemini responses."""
                print("📥 Receive task started")
                nonlocal latest_handle
                consecutive_receive_errors = 0
                MAX_CONSECUTIVE_RECEIVE_ERRORS = 10
                
                try:
                    while not stop.is_set():
                        try:
                            try:
                                async for msg in session.receive():
                                    if stop.is_set():
                                        break
                                    
                                    try:
                                        # Handle session resumption
                                        try:
                                            if (msg.session_resumption_update and 
                                                msg.session_resumption_update.new_handle):
                                                latest_handle = msg.session_resumption_update.new_handle
                                                print(f"🔄 Session handle updated")
                                        except Exception as e:
                                            print(f"⚠️  Session resumption error: {type(e).__name__}: {e}")
                                        
                                        # Handle server content
                                        try:
                                            sc = msg.server_content
                                            if sc is None:
                                                continue
                                            
                                            # Handle model turn (response)
                                            if sc.model_turn:
                                                try:
                                                    if not adam_speaking.is_set():
                                                        adam_speaking.set()
                                                        try:
                                                            if tft_renderer:
                                                                tft_renderer.set_emotion(TFT_EMOTION_SPEAKING)
                                                        except Exception as e:
                                                            print(f"⚠️  TFT update error: {type(e).__name__}: {e}")
                                                        
                                                        await attention.set_responding(True)
                                                        print("🤖 ADAM speaking")
                                                    
                                                    for part in sc.model_turn.parts:
                                                        try:
                                                            if part.inline_data and part.inline_data.data:
                                                                await out_q.put(part.inline_data.data)
                                                        except Exception as e:
                                                            print(f"⚠️  Audio data queuing error: {type(e).__name__}: {e}")
                                                
                                                except Exception as e:
                                                    print(f"⚠️  Model turn processing error: {type(e).__name__}: {e}")
                                            
                                            # Handle turn complete
                                            if sc.turn_complete:
                                                try:
                                                    await out_q.put(None)
                                                    print("✅ Turn complete")
                                                except Exception as e:
                                                    print(f"⚠️  Turn complete error: {type(e).__name__}: {e}")
                                        
                                        except Exception as e:
                                            print(f"⚠️  Content processing error: {type(e).__name__}: {e}")
                                    
                                    except Exception as e:
                                        print(f"⚠️  Message processing error: {type(e).__name__}: {e}")
                                        consecutive_receive_errors += 1
                                
                                consecutive_receive_errors = 0  # Reset on successful messages
                            
                            except asyncio.CancelledError:
                                raise
                            except Exception as e:
                                print(f"⚠️  Receive stream error: {type(e).__name__}: {e}")
                                consecutive_receive_errors += 1
                                
                                if consecutive_receive_errors > MAX_CONSECUTIVE_RECEIVE_ERRORS:
                                    print(f"❌ Receive: {MAX_CONSECUTIVE_RECEIVE_ERRORS} consecutive errors - reconnecting")
                                    break  # Exit to trigger session reconnection
                                
                                await asyncio.sleep(1.0)
                        
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            print(f"⚠️  Receive loop error: {type(e).__name__}: {e}")
                            await asyncio.sleep(1.0)
                
                except asyncio.CancelledError:
                    print("📥 Receive task cancelled")
                except Exception as e:
                    print(f"❌ Receive task failed: {type(e).__name__}: {e}")
                finally:
                    print("📥 Receive task ended")

            async def speaker() -> None:
                """Output audio from Gemini."""
                print("🔊 Speaker task started")
                stream = None
                speaker_failed = False
                consecutive_speaker_errors = 0
                MAX_CONSECUTIVE_SPEAKER_ERRORS = 5
                
                try:
                    # Try to open speaker stream
                    try:
                        if pya is None:
                            raise RuntimeError("PyAudio not initialized")
                        
                        stream = pya.open(
                            format=FORMAT,
                            channels=OUT_CHANNELS,
                            rate=RECV_SAMPLE_RATE,
                            output=True
                        )
                        print("✅ Speaker stream opened")
                    except Exception as e:
                        print(f"❌ Speaker initialization failed: {type(e).__name__}: {e}")
                        speaker_failed = True
                        stream = None
                    
                    while not stop.is_set():
                        try:
                            # Skip if speaker failed
                            if speaker_failed or stream is None:
                                try:
                                    chunk = await asyncio.wait_for(
                                        out_q.get(),
                                        timeout=0.5
                                    )
                                    if chunk is None:
                                        adam_speaking.clear()
                                        await attention.set_responding(False)
                                        print("🤖 ADAM finished speaking (speaker unavailable)")
                                except asyncio.TimeoutError:
                                    pass
                                except Exception as e:
                                    print(f"⚠️  Queue drain error: {type(e).__name__}: {e}")
                                
                                if not speaker_failed:
                                    print("⚠️  Speaker not available - skipping audio output")
                                    speaker_failed = True
                                await asyncio.sleep(1.0)
                                continue
                            
                            try:
                                chunk = await asyncio.wait_for(
                                    out_q.get(),
                                    timeout=SPEAKER_TIMEOUT_S
                                )
                                
                                if chunk is None:
                                    # Turn complete sentinel
                                    try:
                                        if tft_renderer:
                                            tft_renderer.set_emotion(current_tft_emotion)
                                    except Exception as e:
                                        print(f"⚠️  TFT emotion error: {type(e).__name__}: {e}")
                                    
                                    adam_speaking.clear()
                                    await attention.set_responding(False)
                                    print("🤖 ADAM finished speaking")
                                    consecutive_speaker_errors = 0
                                else:
                                    # Write audio chunk
                                    try:
                                        await asyncio.to_thread(stream.write, chunk)
                                        consecutive_speaker_errors = 0
                                    except Exception as e:
                                        print(f"⚠️  Speaker write error: {type(e).__name__}: {e}")
                                        consecutive_speaker_errors += 1
                                        if consecutive_speaker_errors > MAX_CONSECUTIVE_SPEAKER_ERRORS:
                                            print(f"❌ Speaker: {MAX_CONSECUTIVE_SPEAKER_ERRORS} consecutive errors - disabling speaker")
                                            speaker_failed = True
                                            if stream:
                                                try:
                                                    stream.stop_stream()
                                                    stream.close()
                                                except:
                                                    pass
                                                stream = None
                            
                            except asyncio.TimeoutError:
                                if adam_speaking.is_set():
                                    adam_speaking.clear()
                                    print("⚠️  Speaker timeout - cleared speaking flag")
                            
                            except asyncio.CancelledError:
                                raise
                            except Exception as e:
                                print(f"⚠️  Speaker operation error: {type(e).__name__}: {e}")
                                consecutive_speaker_errors += 1
                                await asyncio.sleep(0.1)
                        
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            print(f"❌ Speaker loop error: {type(e).__name__}: {e}")
                            await asyncio.sleep(1.0)
                
                except asyncio.CancelledError:
                    print("🔊 Speaker task cancelled")
                except Exception as e:
                    print(f"❌ Speaker task failed: {type(e).__name__}: {e}")
                finally:
                    try:
                        if stream:
                            stream.stop_stream()
                            stream.close()
                            print("🔊 Speaker stream closed")
                    except Exception as e:
                        print(f"⚠️  Speaker close error: {type(e).__name__}: {e}")
                    print("🔊 Speaker task ended")

            # Start all tasks
            print("🚀 Starting all I/O tasks...")
            tasks = [
                asyncio.create_task(camera(), name="camera"),
                asyncio.create_task(listen(), name="listen"),
                asyncio.create_task(send(), name="send"),
                asyncio.create_task(receive(), name="receive"),
                asyncio.create_task(speaker(), name="speaker"),
                asyncio.create_task(esp32_sensor_poller(), name="sensors"),
            ]
            print(f"✅ Started {len(tasks)} tasks")
            
            # Wait for first task to complete or stop signal
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            print("🛑 First task completed - cancelling all...")
            
            # Cancel remaining tasks
            for t in tasks:
                if not t.done():
                    t.cancel()
            
            # Wait for cancellation to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            print("✅ All tasks cancelled")
    
    except asyncio.CancelledError:
        print("🌐 Session task cancelled")
    except Exception as e:
        print(f"❌ Session error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"🌐 Session closed. Latest handle: {latest_handle}")
    
    return latest_handle

async def main() -> None:
    """Main entry point."""
    print("\n" + "="*80)
    print("🤖 ADAM v30 — Autonomous Desktop AI Module (Pi Wi-Fi Integration)")
    print("="*80)
    
    try:
        # Initialize TFT renderer
        if tft_renderer:
            try:
                print("📺 Starting TFT renderer...")
                tft_renderer.start()
                print("✅ TFT renderer started")
            except Exception as e:
                print(f"⚠️  TFT start failed: {type(e).__name__}: {e}")
        
        # Initialize managers
        print("🔔 Initializing attention manager...")
        attention = AttentionManager()
        
        print("🎤 Initializing voice detector...")
        vad = VoiceActivityDetector()
        
        print("📷 Initializing person tracker...")
        tracker = PersonTracker()
        
        # Create Gemini client
        print("🌐 Initializing Gemini API client...")
        try:
            client = genai.Client(api_key=API_KEY)
            print("✅ Gemini client ready")
        except Exception as e:
            print(f"❌ Gemini client init failed: {type(e).__name__}: {e}")
            raise
        
        # Initialize control objects
        stop = asyncio.Event()
        out_q: asyncio.Queue = asyncio.Queue(maxsize=OUT_QUEUE_MAXSIZE)
        resume_handle = None
        
        # Print component status
        print("\n" + "📊 COMPONENT STATUS ".center(80, "─"))
        print(f"  Camera (USB):        {'✅ READY' if COMPONENTS_STATUS['camera'] else '❌ UNAVAILABLE'}")
        print(f"  Microphone (Stereo): {'✅ READY' if COMPONENTS_STATUS['microphone'] else '❌ UNAVAILABLE'}")
        print(f"  Speaker (Amp):       {'✅ READY' if COMPONENTS_STATUS['microphone'] else '❌ UNAVAILABLE'}")
        print(f"  Servo (Pan/Tilt):    {'✅ READY' if COMPONENTS_STATUS['servo'] else '❌ UNAVAILABLE'}")
        print(f"  TFT Display:         {'✅ READY' if COMPONENTS_STATUS['tft'] else '❌ UNAVAILABLE'}")
        print("─" * 80)
        
        print("\n" + "🚀 "*20)
        print("✅ ADAM ONLINE - Entering main loop...")
        print("🚀 "*20 + "\n")
        
        # Main reconnect loop
        reconnect_count = 0
        while not stop.is_set():
            try:
                print(f"\n🌐 Session #{reconnect_count + 1} starting...")
                resume_handle = await run_session(
                    client, 
                    resume_handle, 
                    stop, 
                    out_q, 
                    attention, 
                    vad, 
                    tracker
                )
                
                if resume_handle is None:
                    print("⚠️  Session ended without handle - stopping")
                    break
                
                reconnect_count += 1
                print(f"🔄 Reconnecting... (session #{reconnect_count + 1})")
                await asyncio.sleep(1.0)  # Brief delay before reconnect
            
            except asyncio.CancelledError:
                print("⚠️  Main loop cancelled")
                break
            except Exception as e:
                print(f"❌ Session failed: {type(e).__name__}: {e}")
                reconnect_count += 1
                await asyncio.sleep(2.0)  # Longer delay after error
    
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received")
    except Exception as e:
        print(f"❌ Main error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n" + "="*80)
        print("🛑 SHUTDOWN SEQUENCE")
        print("="*80)
        
        try:
            stop.set()
            print("✅ Stop signal set")
        except Exception as e:
            print(f"⚠️  Stop set failed: {e}")
        
        try:
            if pya:
                pya.terminate()
                print("✅ PyAudio terminated")
        except Exception as e:
            print(f"⚠️  PyAudio cleanup failed: {e}")
        
        try:
            if tft_renderer:
                tft_renderer.stop()
                print("✅ TFT renderer stopped")
        except Exception as e:
            print(f"⚠️  TFT stop failed: {e}")
        
        try:
            close_neck()
            print("✅ Neck closed")
        except Exception as e:
            print(f"⚠️  Neck cleanup failed: {e}")
        
        print("="*80)
        print("✅ ADAM SHUTDOWN COMPLETE")
        print("="*80 + "\n")

if __name__ == "__main__":
    try:
        print("🤖 Starting ADAM...\n")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if tft_renderer:
                tft_renderer.stop()
            close_neck()
            if pya:
                pya.terminate()
        except Exception:
            pass