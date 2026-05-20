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
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

LIVE_MODEL          = "gemini-2.5-flash-preview-09-2025"
GEN_MODEL_CASCADE   = ["gemini-2.5-flash-preview-09-2025"]
GEN_RETRIES         = 2

POST_SPEECH_MUTE_S  = 0.4
VOICE               = "Charon"

CAMERA_FPS_INTERVAL = 1.0          

ENABLE_IDLE         = True
IDLE_TIMEOUT_S      = 90           
ATTENTION_TIMEOUT_S = 30       

# Physical neck tracking
NECK_PAN_CENTER       = 90
NECK_TILT_CENTER      = 85
NECK_PAN_MIN          = 30
NECK_PAN_MAX          = 150
NECK_TILT_MIN         = 50
NECK_TILT_MAX         = 120
NECK_PAN_DEADZONE     = 5         
NECK_TILT_DEADZONE    = 6          
NECK_TRACK_INTERVAL   = 0.12       
NECK_MAX_STEP         = 3          
NECK_RECENTER_HOLD_S  = 1.5        
NECK_FACE_TRACK_ALPHA_PAN = 0.32   
NECK_FACE_TRACK_ALPHA_TILT = 0.18  
NECK_IDLE_START_AFTER_S = 4.0      

# Stereo Mic Sound Localization (DoA) Config
MIC_DISTANCE_M        = 0.065      # Distance between the two INMP441 mics (65mm)

DETECT_W                    = 320  
DETECT_H                    = 240  
FACE_DETECT_MIN_NEIGHBORS   = 4
VAD_FRAME_MS                = 20   
VAD_WINDOW                  = 10   
GAZE_AWAY_DEACTIVATE_S      = 2.0  

WAKE_WORDS      = ["adam", "hey adam", "ok adam", "okay adam"]
VOSK_MODEL_PATH = "vosk-model-small-en-in-0.4"

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE      = Path(BASE_DIR) / "adam_memory.json"
FACE_MEMORY_FILE = Path(BASE_DIR) / "adam_faces.json"
CONV_MEMORY_FILE = Path(BASE_DIR) / "adam_conversations.json"

CONV_MAX_TURNS    = 40   
CONV_PROMPT_TURNS = 20   

SEARCH_CACHE_TTL_S  = 1800   
SEARCH_MIN_GAP_S    = 5.0    

_ddg_cache:   dict[str, tuple[str, float]] = {}
_last_ddg_t:  float                        = 0.0

global_audio_angle = [0.0]
global_audio_active = [0.0]

# ═════════════════════════════════════════════════════════════════════════════
# HARDWARE ABSTRACTION
# ═════════════════════════════════════════════════════════════════════════════

tft_renderer = TFTEmotionRenderer()
current_tft_emotion = "idle"

try:
    pan_servo = AngularServo(12, min_angle=-90, max_angle=90, min_pulse_width=0.0005, max_pulse_width=0.0025)
    print("✅ Pan Servo initialized on GPIO 12")
except Exception as e:
    pan_servo = None
    print(f"⚠️ Pan Servo init failed: {e}")

def neck_is_ready():
    return True

def pan(angle, speed=None):
    if pan_servo:
        try:
            mapped_angle = max(-90, min(90, int(angle) - 90))
            pan_servo.angle = mapped_angle
        except Exception as e:
            print(f"Pan error: {e}")

async def set_tilt_wifi(angle):
    url = f"http://{ESP32_IP}/tilt?angle={angle}"
    async with aiohttp.ClientSession() as session:
        try:
            await session.get(url, timeout=1.0)
        except Exception:
            pass 

def tilt(angle, speed=None):
    asyncio.create_task(set_tilt_wifi(angle))

def reset_neck():
    pan(NECK_PAN_CENTER)
    tilt(NECK_TILT_CENTER)

def close_neck():
    reset_neck()

# ═════════════════════════════════════════════════════════════════════════════
# SOUND LOCALIZATION (Direction of Arrival)
# ═════════════════════════════════════════════════════════════════════════════

def get_doa_angle(left_channel, right_channel, sample_rate=16000, mic_dist=MIC_DISTANCE_M):
    """
    Calculates the angle of the sound source using Generalized Cross-Correlation (GCC-PHAT).
    Returns angle in degrees. Negative = Left, Positive = Right.
    """
    N = 1024
    L = np.fft.rfft(left_channel, n=N)
    R = np.fft.rfft(right_channel, n=N)
    
    R_cross = L * np.conj(R)
    cc = np.fft.irfft(R_cross / (np.abs(R_cross) + 1e-15))
    
    # Calculate max possible sample shift based on physical mic distance
    max_shift = int(sample_rate * mic_dist / 343.0) + 1
    
    # Center the correlation
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(cc) - max_shift
    
    # Convert shift directly to angle
    val = (shift / sample_rate) * 343.0 / mic_dist
    val = np.clip(val, -1.0, 1.0)
    angle_deg = np.degrees(np.arcsin(val))
    
    return angle_deg

# ═════════════════════════════════════════════════════════════════════════════
# VISION TRACKER
# ═════════════════════════════════════════════════════════════════════════════

class PersonTracker:
    def __init__(self) -> None:
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade   = cv2.CascadeClassifier(cascade)
        self._available = not self._cascade.empty()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._adam_speaking:  bool      = False

    def process_frame(self, frame: np.ndarray, vad_active: bool = True) -> dict:
        result: dict = {"faces": [], "active_speaker_idx": None, "face_count": 0}
        if not self._available or frame is None: return result

        h_f, w_f = frame.shape[:2]
        gray_small = self._clahe.apply(cv2.cvtColor(cv2.resize(frame, (DETECT_W, DETECT_H)), cv2.COLOR_BGR2GRAY))

        raw_det = self._cascade.detectMultiScale(
            gray_small, scaleFactor=1.2, minNeighbors=FACE_DETECT_MIN_NEIGHBORS, minSize=(20, 20)
        )

        scale_x = w_f / DETECT_W
        scale_y = h_f / DETECT_H

        faces = [(int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)) for (x,y,w,h) in raw_det]
        result["face_count"] = len(faces)
        
        for idx, (fx, fy, fw, fh) in enumerate(faces):
            cx, cy = (fx + fw / 2) / w_f, (fy + fh / 2) / h_f
            facing_camera = (fw / max(fh, 1) > 0.65) 
            result["faces"].append({
                "id": idx, "cx_norm": cx, "cy_norm": cy, "facing_camera": facing_camera
            })

        return result

    def build_context(self, tr: dict) -> str:
        count = tr["face_count"]
        if count == 0: return "[CAMERA: No faces in frame.]"
        parts = []
        for i, f in enumerate(tr["faces"]):
            pos = "left" if f["cx_norm"] < 0.40 else "right" if f["cx_norm"] > 0.60 else "centre"
            parts.append(f"P{i+1}({pos},{ '→CAM' if f['facing_camera'] else '→AWAY' })")
        return f"[CAMERA: {count} people — {', '.join(parts)}.]"

    def set_adam_speaking(self, flag: bool) -> None:
        self._adam_speaking = flag

# ═════════════════════════════════════════════════════════════════════════════
# AUDIO CORE
# ═════════════════════════════════════════════════════════════════════════════

class VoiceActivityDetector:
    def __init__(self) -> None:
        self._ready, self._vad = False, None
        self._buf = b""
        self._votes: deque = deque(maxlen=VAD_WINDOW)
        self._frame_bytes = int(SEND_SAMPLE_RATE * (VAD_FRAME_MS / 1000.0)) * 2
        if WEBRTCVAD_AVAILABLE:
            try: self._vad = _webrtcvad.Vad(2); self._ready = True
            except Exception: pass
    def feed(self, pcm_chunk: bytes) -> None:
        if not self._ready: return
        self._buf += pcm_chunk
        while len(self._buf) >= self._frame_bytes:
            frame = self._buf[: self._frame_bytes]
            self._buf = self._buf[self._frame_bytes :]
            try: self._votes.append(1 if self._vad.is_speech(frame, SEND_SAMPLE_RATE) else 0)
            except Exception: pass
    def is_active(self) -> bool:
        if not self._ready or not self._votes: return True
        return sum(self._votes) > len(self._votes) // 2

class AttentionState:
    PASSIVE, ATTENTIVE, RESPONDING = "passive", "attentive", "responding"

class AttentionManager:
    def __init__(self) -> None:
        self._state, self._last_t, self._lock = AttentionState.PASSIVE, 0.0, asyncio.Lock()
    def is_active(self) -> bool:
        if self._state == AttentionState.ATTENTIVE:
            if time.time() - self._last_t > ATTENTION_TIMEOUT_S:
                self._state = AttentionState.PASSIVE
                return False
            return True
        return False
    async def activate(self, reason: str = "") -> None:
        async with self._lock:
            if self._state == AttentionState.RESPONDING: return
            self._state, self._last_t = AttentionState.ATTENTIVE, time.time()
    async def set_responding(self, on: bool) -> None:
        async with self._lock:
            self._state = AttentionState.RESPONDING if on else AttentionState.ATTENTIVE
            if not on: self._last_t = time.time()
    def touch(self) -> None:
        if self._state in (AttentionState.ATTENTIVE, AttentionState.RESPONDING): self._last_t = time.time()

# ═════════════════════════════════════════════════════════════════════════════
# I2S AUDIO SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
FORMAT           = pyaudio.paInt16
MIC_CHANNELS     = 2  # CRITICAL: 2 Mics = Stereo input
OUT_CHANNELS     = 1  # Amp is mono
SEND_SAMPLE_RATE = 16000
RECV_SAMPLE_RATE = 24000
CHUNK_SIZE       = 512
pya = pyaudio.PyAudio()

# ═════════════════════════════════════════════════════════════════════════════
# RUN SESSION
# ═════════════════════════════════════════════════════════════════════════════

async def run_session(client, resume_handle, stop, out_q, attention, vad, tracker) -> str | None:
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"], 
        system_instruction="You are ADAM. Keep answers short and natural.", 
        session_resumption=types.SessionResumptionConfig(handle=resume_handle),
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)))
    )

    latest_handle: str | None = resume_handle

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            mic_q = asyncio.Queue(maxsize=120)
            adam_speaking = asyncio.Event()

            async def esp32_sensor_poller():
                url = f"http://{ESP32_IP}/sensors"
                async with aiohttp.ClientSession() as http_session:
                    while not stop.is_set():
                        try:
                            async with http_session.get(url, timeout=0.8) as resp:
                                data = await resp.json()
                                if any(t == 1 for t in data.get("touch", [0,0,0,0])):
                                    tft_renderer.set_emotion("happy")
                                    await attention.activate("esp32-touch")
                        except Exception: pass 
                        await asyncio.sleep(0.2) 

            async def camera():
                cap = cv2.VideoCapture(f"http://{ESP32_IP}:80/stream")
                last_sent = 0.0
                curr_pan, curr_tilt = float(NECK_PAN_CENTER), float(NECK_TILT_CENTER)
                try:
                    while not stop.is_set():
                        raw = await asyncio.to_thread(lambda: cap.read()[1] if cap.isOpened() else None)
                        if raw is None: 
                            await asyncio.sleep(0.1)
                            continue
                        
                        now = time.time()
                        if now - last_sent >= CAMERA_FPS_INTERVAL:
                            tr = await asyncio.to_thread(tracker.process_frame, raw, vad.is_active())
                            if any(f["facing_camera"] for f in tr["faces"]):
                                await attention.activate("face-gaze-detected")
                            
                            # Auto Neck Pan - Vision vs Audio DoA
                            if tr.get("faces"):
                                # If face is visible, look at the face
                                f = tr["faces"][0]
                                aim_pan = float(NECK_PAN_MAX - f["cx_norm"] * (NECK_PAN_MAX - NECK_PAN_MIN))
                                aim_tilt = float(NECK_TILT_MIN + f["cy_norm"] * (NECK_TILT_MAX - NECK_TILT_MIN))
                            elif now - global_audio_active[0] < 2.0:
                                # If no face, but someone is talking, snap to the Sound Direction
                                aim_pan = float(NECK_PAN_CENTER + global_audio_angle[0])
                                aim_pan = max(NECK_PAN_MIN, min(NECK_PAN_MAX, aim_pan))
                                aim_tilt = float(NECK_TILT_CENTER)
                            else:
                                aim_pan = float(NECK_PAN_CENTER)
                                aim_tilt = float(NECK_TILT_CENTER)
                                
                            # Smooth Movement
                            curr_pan += (aim_pan - curr_pan) * 0.3
                            curr_tilt += (aim_tilt - curr_tilt) * 0.3
                            
                            await asyncio.to_thread(pan, int(curr_pan))
                            tilt(int(curr_tilt)) 
                                
                            last_sent = now
                        await asyncio.sleep(0.01)
                except asyncio.CancelledError: pass
                finally:
                    if cap: cap.release()

            async def listen():
                # Record in Stereo
                stream = pya.open(format=FORMAT, channels=MIC_CHANNELS, rate=SEND_SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)
                try:
                    while not stop.is_set():
                        raw_data = await asyncio.to_thread(stream.read, CHUNK_SIZE, exception_on_overflow=False)
                        
                        # Process Stereo audio
                        stereo_samples = np.frombuffer(raw_data, dtype=np.int16)
                        left_mic = stereo_samples[0::2]
                        right_mic = stereo_samples[1::2]
                        
                        # Send mono (Left channel) to Voice AI
                        mono_data = left_mic.tobytes()
                        vad.feed(mono_data)
                        
                        # Calculate Sound Direction if speaking
                        if vad.is_active() and not adam_speaking.is_set():
                            angle = get_doa_angle(left_mic.astype(np.float32), right_mic.astype(np.float32))
                            if abs(angle) > 10: # Deadzone filter
                                global_audio_angle[0] = angle
                                global_audio_active[0] = time.time()
                                await attention.activate("sound-detected")

                        if not mic_q.full(): mic_q.put_nowait(mono_data)
                except asyncio.CancelledError: pass
                finally:
                    stream.stop_stream(); stream.close()

            async def send():
                try:
                    while not stop.is_set():
                        chunk = await mic_q.get() 
                        if adam_speaking.is_set() or not attention.is_active(): continue
                        try:
                            await session.send_realtime_input(audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000"))
                        except Exception: await asyncio.sleep(0.01)
                except asyncio.CancelledError: pass

            async def receive():
                nonlocal latest_handle
                try:
                    while not stop.is_set():
                        async for msg in session.receive():
                            if stop.is_set(): break
                            if msg.session_resumption_update and msg.session_resumption_update.new_handle:
                                latest_handle = msg.session_resumption_update.new_handle
                            
                            sc = msg.server_content
                            if sc is None: continue

                            if sc.model_turn:
                                if not adam_speaking.is_set():
                                    adam_speaking.set()
                                    tft_renderer.set_emotion("speaking") 
                                    await attention.set_responding(True)
                                
                                for part in sc.model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        await out_q.put(part.inline_data.data)

                            if sc.turn_complete:
                                await out_q.put(None)
                except asyncio.CancelledError: pass

            async def speaker():
                # Output in Mono
                stream = pya.open(format=FORMAT, channels=OUT_CHANNELS, rate=RECV_SAMPLE_RATE, output=True)
                try:
                    while not stop.is_set():
                        chunk = await asyncio.wait_for(out_q.get(), timeout=0.3)
                        if chunk is None:
                            tft_renderer.set_emotion(current_tft_emotion) 
                            adam_speaking.clear()
                            await attention.set_responding(False)
                        else:
                            await asyncio.to_thread(stream.write, chunk)
                except asyncio.TimeoutError:
                    if adam_speaking.is_set(): adam_speaking.clear()
                except asyncio.CancelledError: pass
                finally:
                    stream.stop_stream(); stream.close()

            tasks = [
                asyncio.create_task(camera()), asyncio.create_task(listen()),
                asyncio.create_task(send()), asyncio.create_task(receive()),
                asyncio.create_task(speaker()), asyncio.create_task(esp32_sensor_poller())
            ]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in tasks: t.cancel()
    except Exception as e:
        print(f"Session error: {e}")
    return latest_handle

async def main() -> None:
    tft_renderer.start()
    attention = AttentionManager()
    vad = VoiceActivityDetector()
    tracker = PersonTracker()
    client = genai.Client(api_key=API_KEY)
    stop = asyncio.Event()
    out_q = asyncio.Queue(maxsize=200)
    resume_handle = None

    while not stop.is_set():
        resume_handle = await run_session(client, resume_handle, stop, out_q, attention, vad, tracker)
        if resume_handle is None: break

    stop.set()
    pya.terminate()
    tft_renderer.stop()
    close_neck()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
    finally:
        tft_renderer.stop()
        close_neck()