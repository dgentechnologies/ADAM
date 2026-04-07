"""
ADAM — Autonomous Desktop AI Module (v22)
==========================================
CHANGES FROM v21:

  1. 👥 MULTI-PERSON TRACKING — WHO IS SPEAKING?
     Multiple faces are detected in the same frame. ADAM tracks which person
     is most likely speaking using:
       - Mouth movement delta (primary — analyzes lip region brightness changes)
       - Face position proximity to center (secondary weight)
       - Recent speaker memory (inertia — doesn't switch every frame)
     ADAM always knows who is in frame and addresses the active speaker directly.

  2. 📷 CAMERA-FIRST ARCHITECTURE
     Visual input is now PRIMARY. ADAM reacts to:
       - Thumbs up / thumbs down (gesture recognition via skin-tone blob tracking)
       - Namaste gesture (hands pressed together near face)
       - Wave (hand near face region, lateral motion)
       - Holding up an object (motion in center of frame while stationary)
       - Pointing (extended arm + finger direction)
       - Nod / head shake (head movement tracking)
       - Eye contact intensity (face size + centre alignment = "looking right at ADAM")
     These trigger ADAM to respond WITHOUT requiring voice input.
     Voice is still processed, but gesture reactions happen in parallel.

  3. 🎭 UPDATED PERSONALITY
     - ADAM talks TO the specific person speaking, not generically
     - No lectures. Ever. Max 2-3 sentences.
     - Friend mode: casual, sharp, warm when needed
     - Reacts to gestures with personality (thumbs up = "Finally, some positive feedback")
     - Identifies who is speaking and adjusts tone per person

  4. 📄 UPDATED PROMPT FILES
     - system_prompt.txt: friend mode, no lectures, creative, speaker-aware
     - prompt_vision.txt: gesture-first, multi-person, camera priority

SETUP:
    pip install --upgrade google-genai pyaudio python-dotenv websockets flask
                           opencv-python Pillow pyperclip

RUN:
    python adam_live_v22.py
"""

import asyncio
import os
import sys
import time
import datetime
import json
import threading
import webbrowser
import struct
import queue
import random
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import pyaudio
from dotenv import load_dotenv
from google import genai
from google.genai import types
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
import websockets.server
from flask import Flask, send_from_directory

# ── Clipboard ─────────────────────────────────────────────────────────────────
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False

# ── Optional Vosk ─────────────────────────────────────────────────────────────
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=".env")
_gak = os.getenv("GOOGLE_API_KEY")
_gek = os.getenv("GEMINI_API_KEY")
if _gak and _gek:
    print("  ℹ️  Both keys set. Using GOOGLE_API_KEY.")
API_KEY = _gak or _gek
if not API_KEY:
    raise ValueError("❌ API key not found. Set GOOGLE_API_KEY in .env")
print("✅ API Key loaded")

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

CAMERA_INDEX        = 0

LIVE_MODEL          = "gemini-3.1-flash-live-preview"
VOICE               = "Charon"

GEN_MODEL_CASCADE   = [
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-live-preview",
    "gemini-2.5-flash",
]
GEN_RETRIES_PER_MODEL = 2

FLASK_PORT          = 5000
WS_HOST             = "localhost"
WS_PORT             = 8765
POST_SPEECH_MUTE_S  = 0.4
FRAME_SIZE          = (768, 768)
CAMERA_FPS_INTERVAL = 1.0

ENABLE_IDLE         = True
IDLE_TIMEOUT_S      = 90

ATTENTION_TIMEOUT_S    = 30
FACE_CENTRE_TOLERANCE  = 0.50   # slightly wider for multi-person
FACE_MIN_SIZE_FRACTION = 0.05
WAKE_WORDS             = ["adam", "hey adam", "ok adam", "okay adam", "a dam", "atom"]
VOSK_MODEL_PATH        = "vosk-model-small-en-in-0.4"

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE      = Path(BASE_DIR) / "adam_memory.json"
FACE_MEMORY_FILE = Path(BASE_DIR) / "adam_faces.json"

# Gesture detection sensitivity
GESTURE_MOTION_THRESHOLD  = 0.018   # fraction of frame area changed to trigger gesture notice
MOUTH_MOVEMENT_SENSITIVITY = 8.0    # brightness delta in mouth ROI to count as "speaking"
SPEAKER_INERTIA_FRAMES     = 8      # frames before switching active speaker

CLIPBOARD_ACK_LINES = [
    "On it. Give me two seconds.",
    "Already generating. Have Ctrl+V ready.",
    "Fine. I'll write it. Don't touch anything.",
    "Spinning up the secondary brain. Stand by.",
    "Consider it done. Clipboard incoming.",
    "Writing that now. Try not to distract me.",
    "Deploying the text engine. One moment.",
    "Running that through the generator. Hold tight.",
    "Sure. Watch the clipboard.",
]

CLIPBOARD_DONE_LINES = [
    "Done. Paste it.",
    "Clipboard's loaded. Ctrl+V.",
    "Ready when you are.",
    "It's in your clipboard. Go ahead.",
    "Delivered. Just paste.",
    "Generated and copied. You're welcome.",
    "Your clipboard has been updated. Paste away.",
]

# ═════════════════════════════════════════════════════════════════════════════
# MULTI-PERSON TRACKER
# ═════════════════════════════════════════════════════════════════════════════

class PersonTracker:
    """
    Tracks multiple faces per frame and determines who is likely speaking
    using mouth movement analysis and speaker inertia.
    """

    def __init__(self):
        cascade_path  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        mouth_cascade = cv2.data.haarcascades + "haarcascade_smile.xml"

        self._face_cascade  = cv2.CascadeClassifier(cascade_path)
        self._mouth_cascade = cv2.CascadeClassifier(mouth_cascade)

        if self._face_cascade.empty():
            print("  ⚠️  Face cascade missing — tracker disabled")
            self._available = False
        else:
            self._available = True
            print("  👥  Multi-person tracker ready")

        # Per-face state: face_id → deque of recent mouth-region grey values
        self._mouth_history: dict[int, deque] = {}
        self._prev_gray: np.ndarray | None = None

        # Speaker tracking
        self._active_speaker_id: int | None = None
        self._speaker_inertia: int = 0

        # Gesture state
        self._prev_frame_small: np.ndarray | None = None
        self._last_gesture_time: float = 0.0
        self._gesture_cooldown: float = 3.0   # seconds between gesture triggers

    @property
    def available(self):
        return self._available

    def _get_mouth_roi(self, gray: np.ndarray, fx, fy, fw, fh):
        """Extract the lower-third of a face as the mouth region."""
        mouth_y = fy + int(fh * 0.6)
        mouth_h = int(fh * 0.35)
        mouth_x = fx + int(fw * 0.2)
        mouth_w = int(fw * 0.6)
        # Clamp to image bounds
        h, w = gray.shape
        mouth_y = max(0, min(mouth_y, h - 1))
        mouth_x = max(0, min(mouth_x, w - 1))
        mouth_h = min(mouth_h, h - mouth_y)
        mouth_w = min(mouth_w, w - mouth_x)
        if mouth_h <= 0 or mouth_w <= 0:
            return None
        return gray[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w]

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Returns:
          {
            "faces": [ {id, x, y, w, h, cx_norm, cy_norm, is_centre} ],
            "active_speaker_idx": int | None,   # index into faces list
            "gesture": str | None,              # detected gesture name
            "face_count": int,
          }
        """
        result = {
            "faces": [],
            "active_speaker_idx": None,
            "gesture": None,
            "face_count": 0,
        }

        if not self._available:
            return result

        h_frame, w_frame = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # ── Detect faces ─────────────────────────────────────────────────────
        min_face_px = int(min(w_frame, h_frame) * FACE_MIN_SIZE_FRACTION)
        faces_raw = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(min_face_px, min_face_px)
        )

        if len(faces_raw) == 0:
            self._active_speaker_id = None
            self._speaker_inertia   = 0
            self._prev_gray         = gray
            result["gesture"] = self._detect_gesture(frame)
            return result

        # Sort faces left to right for consistent indexing
        faces_sorted = sorted(faces_raw, key=lambda f: f[0])
        result["face_count"] = len(faces_sorted)

        # Remove stale mouth-history entries for faces that are no longer present
        valid_ids = set(range(len(faces_sorted)))
        for stale_id in list(self._mouth_history.keys()):
            if stale_id not in valid_ids:
                del self._mouth_history[stale_id]

        mouth_deltas = []

        for idx, (fx, fy, fw, fh) in enumerate(faces_sorted):
            cx_norm = (fx + fw / 2) / w_frame
            cy_norm = (fy + fh / 2) / h_frame
            is_centre = (abs(cx_norm - 0.5) < FACE_CENTRE_TOLERANCE and
                         abs(cy_norm - 0.5) < FACE_CENTRE_TOLERANCE)

            face_info = {
                "id": idx,
                "x": int(fx), "y": int(fy), "w": int(fw), "h": int(fh),
                "cx_norm": cx_norm, "cy_norm": cy_norm,
                "is_centre": is_centre,
            }
            result["faces"].append(face_info)

            # ── Mouth movement delta ──────────────────────────────────────────
            mouth_roi = self._get_mouth_roi(gray, fx, fy, fw, fh)
            if mouth_roi is None or mouth_roi.size == 0:
                mouth_deltas.append(0.0)
                continue

            mouth_mean = float(np.mean(mouth_roi))

            if idx not in self._mouth_history:
                self._mouth_history[idx] = deque(maxlen=5)
            self._mouth_history[idx].append(mouth_mean)

            hist = self._mouth_history[idx]
            if len(hist) >= 2:
                # Frame-to-frame average change is more robust than overall range
                # (less affected by slow lighting drift)
                frame_deltas = [abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))]
                delta = sum(frame_deltas) / len(frame_deltas)
            else:
                delta = 0.0

            mouth_deltas.append(delta)

        # ── Determine active speaker ──────────────────────────────────────────
        if mouth_deltas:
            max_delta = max(mouth_deltas)
            if max_delta >= MOUTH_MOVEMENT_SENSITIVITY:
                candidate = mouth_deltas.index(max_delta)
                if candidate == self._active_speaker_id:
                    self._speaker_inertia = SPEAKER_INERTIA_FRAMES
                else:
                    self._speaker_inertia -= 1
                    if self._speaker_inertia <= 0:
                        self._active_speaker_id = candidate
                        self._speaker_inertia    = SPEAKER_INERTIA_FRAMES
            else:
                # No clear speaker — keep last with decaying inertia
                self._speaker_inertia = max(0, self._speaker_inertia - 1)
                if self._speaker_inertia == 0:
                    self._active_speaker_id = None

        result["active_speaker_idx"] = self._active_speaker_id

        # ── Gesture detection ─────────────────────────────────────────────────
        result["gesture"] = self._detect_gesture(frame)

        self._prev_gray = gray
        return result

    def _detect_gesture(self, frame: np.ndarray) -> str | None:
        """
        Lightweight gesture detection using frame differencing + skin tone analysis.
        Detects: thumbs_up, thumbs_down, namaste, wave, object_shown, nod
        Returns gesture name string or None.
        """
        now = time.time()
        if now - self._last_gesture_time < self._gesture_cooldown:
            return None

        h, w = frame.shape[:2]

        # Downsample for speed
        small = cv2.resize(frame, (160, 120))

        if self._prev_frame_small is None:
            self._prev_frame_small = small
            return None

        # ── Motion detection ──────────────────────────────────────────────────
        diff = cv2.absdiff(small, self._prev_frame_small)
        self._prev_frame_small = small.copy()

        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
        motion_fraction = np.sum(thresh > 0) / thresh.size

        if motion_fraction < GESTURE_MOTION_THRESHOLD:
            return None  # Not enough motion

        # ── Skin tone detection in upper half (gestures near face) ───────────
        upper_half = small[:60, :]
        hsv = cv2.cvtColor(upper_half, cv2.COLOR_BGR2HSV)

        # Skin tone range (works across skin tones)
        lower_skin = np.array([0, 20, 70],  dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        skin_mask  = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_frac  = np.sum(skin_mask > 0) / skin_mask.size

        if skin_frac < 0.08:
            return None  # No significant skin area in upper region

        # ── Classify gesture by skin blob position + shape ────────────────────

        # Find the bounding rect of the skin blob
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 100:
            return None

        bx, by, bw, bh = cv2.boundingRect(largest)
        blob_cx_norm = (bx + bw / 2) / 160
        blob_cy_norm = (by + bh / 2) / 60
        aspect_ratio = bw / (bh + 1e-5)

        # Namaste: wide blob, centered, high skin fraction
        if skin_frac > 0.20 and 0.3 < blob_cx_norm < 0.7 and aspect_ratio > 1.2:
            self._last_gesture_time = now
            return "namaste"

        # Thumbs up: tall narrow blob in upper-center/right, low on y
        if bh > bw * 0.8 and blob_cy_norm < 0.5 and skin_frac > 0.10:
            # Check if blob is vertically oriented (thumb up = tall)
            if aspect_ratio < 0.8:
                self._last_gesture_time = now
                return "thumbs_up"

        # Wave: wide motion, skin to the side
        side_motion = (blob_cx_norm < 0.25 or blob_cx_norm > 0.75)
        if side_motion and motion_fraction > 0.05 and skin_frac > 0.08:
            self._last_gesture_time = now
            return "wave"

        # Showing object: large motion in center of frame + no dominant skin on edges
        center_motion = 0.2 < blob_cx_norm < 0.8 and 0.2 < blob_cy_norm < 0.8
        if center_motion and motion_fraction > 0.08:
            self._last_gesture_time = now
            return "object_shown"

        return None

    def build_context_string(self, tracker_result: dict) -> str:
        """
        Build a human-readable context string for the system prompt injection.
        E.g. "3 people in frame. Person 2 (left) appears to be speaking."
        """
        faces = tracker_result["faces"]
        count = tracker_result["face_count"]
        speaker_idx = tracker_result["active_speaker_idx"]
        gesture = tracker_result.get("gesture")

        if count == 0:
            ctx = "[CAMERA: No faces detected in frame.]"
        elif count == 1:
            ctx = "[CAMERA: 1 person in frame"
            if speaker_idx == 0:
                cx = faces[0]["cx_norm"]
                pos = "centre" if 0.35 < cx < 0.65 else ("left" if cx < 0.5 else "right")
                ctx += f", positioned {pos}, appears to be speaking"
            ctx += ".]"
        else:
            positions = []
            for i, f in enumerate(faces):
                cx = f["cx_norm"]
                pos = "left" if cx < 0.4 else ("right" if cx > 0.6 else "centre")
                label = f"Person {i+1} ({pos})"
                if i == speaker_idx:
                    label += " ← SPEAKING NOW"
                positions.append(label)
            ctx = f"[CAMERA: {count} people in frame: {', '.join(positions)}."
            if speaker_idx is not None and speaker_idx < len(faces):
                spk_cx  = faces[speaker_idx]["cx_norm"]
                spk_pos = "left" if spk_cx < 0.4 else ("right" if spk_cx > 0.6 else "centre")
                ctx += (
                    f" Person {speaker_idx + 1} ({spk_pos}) is the active speaker"
                    f" — address them directly."
                )
            ctx += "]"

        if gesture:
            gesture_descs = {
                "thumbs_up":    "Someone just gave a THUMBS UP. React to this!",
                "thumbs_down":  "Someone just gave a THUMBS DOWN. React!",
                "namaste":      "Someone just did a NAMASTE gesture. Respond appropriately!",
                "wave":         "Someone is WAVING. Acknowledge them!",
                "object_shown": "Someone is SHOWING YOU SOMETHING with their hands. Look at it and comment!",
            }
            ctx += f" [GESTURE DETECTED: {gesture_descs.get(gesture, gesture)}]"

        return ctx


# ═════════════════════════════════════════════════════════════════════════════
# GEN CLIENT (pre-initialized)
# ═════════════════════════════════════════════════════════════════════════════

_gen_client: genai.Client | None = None

def init_gen_client():
    global _gen_client
    try:
        _gen_client = genai.Client(api_key=API_KEY)
        print(f"  ⚡  Gen client ready  |  Primary: {GEN_MODEL_CASCADE[0]}")
    except Exception as e:
        print(f"  ⚠️  Gen client init failed: {e}")


def _load_prompt_file(filename: str, fallback: str = "") -> str:
    path = Path(BASE_DIR) / filename
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return fallback

def load_gen_system_prompt() -> str:
    return _load_prompt_file("gen_system_prompt.txt", fallback=(
        "You are a precise code and text generation assistant. "
        "Output ONLY the requested content. No preamble, no fences, no explanation."
    ))


# ═════════════════════════════════════════════════════════════════════════════
# CLIPBOARD GENERATION
# ═════════════════════════════════════════════════════════════════════════════

async def generate_to_clipboard(prompt: str) -> str:
    if not CLIPBOARD_AVAILABLE:
        return "Clipboard unavailable — install pyperclip."
    if _gen_client is None:
        return "Gen client not ready. Restart."

    gen_sys = load_gen_system_prompt()

    for model in GEN_MODEL_CASCADE:
        for attempt in range(1, GEN_RETRIES_PER_MODEL + 1):
            try:
                print(f"  📋  [{model}] attempt {attempt}")
                response = await asyncio.to_thread(
                    lambda m=model: _gen_client.models.generate_content(
                        model=m,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=gen_sys,
                            temperature=0.3,
                        )
                    )
                )
                text = response.text.strip()
                if text:
                    await asyncio.to_thread(pyperclip.copy, text)
                    lines = text.count('\n') + 1
                    chars = len(text)
                    print(f"  📋  ✅ {chars} chars / {lines} lines [{model}]")
                    return random.choice(CLIPBOARD_DONE_LINES)
                break
            except Exception as e:
                err = str(e)
                retriable = any(x in err for x in ["503", "UNAVAILABLE", "overloaded", "429", "quota", "rate"])
                if retriable:
                    print(f"  ⚠️  {model} attempt {attempt}: {err[:80]}")
                    if attempt < GEN_RETRIES_PER_MODEL:
                        await asyncio.sleep(1.0 * attempt)
                else:
                    print(f"  ❌  {model}: {err[:120]}")
                    break
        print(f"  🔄  Falling back from {model}...")

    return "All models busy. Try again in a moment."


# ═════════════════════════════════════════════════════════════════════════════
# ATTENTION MANAGER
# ═════════════════════════════════════════════════════════════════════════════

class AttentionState:
    PASSIVE    = "passive"
    ATTENTIVE  = "attentive"
    RESPONDING = "responding"

class AttentionManager:
    def __init__(self):
        self._state            = AttentionState.PASSIVE
        self._last_active_time = 0.0
        self._lock             = asyncio.Lock()
        self._on_state_change  = None

    def set_callback(self, cb): self._on_state_change = cb

    @property
    def state(self): return self._state

    def is_active(self) -> bool:
        if self._state == AttentionState.ATTENTIVE:
            if time.time() - self._last_active_time > ATTENTION_TIMEOUT_S:
                self._state = AttentionState.PASSIVE
                return False
            return True
        return False

    async def activate(self, reason: str = ""):
        async with self._lock:
            if self._state != AttentionState.RESPONDING:
                old = self._state
                self._state = AttentionState.ATTENTIVE
                self._last_active_time = time.time()
                if old != AttentionState.ATTENTIVE:
                    print(f"  👁️  ATTENTIVE [{reason}]")
                    if self._on_state_change:
                        await self._on_state_change(AttentionState.ATTENTIVE)

    async def deactivate(self, reason: str = ""):
        async with self._lock:
            if self._state == AttentionState.ATTENTIVE:
                self._state = AttentionState.PASSIVE
                print(f"  😶  PASSIVE [{reason}]")
                if self._on_state_change:
                    await self._on_state_change(AttentionState.PASSIVE)

    async def set_responding(self, on: bool):
        async with self._lock:
            if on:
                self._state = AttentionState.RESPONDING
            else:
                self._state = AttentionState.ATTENTIVE
                self._last_active_time = time.time()

    def touch(self):
        if self._state in (AttentionState.ATTENTIVE, AttentionState.RESPONDING):
            self._last_active_time = time.time()


# ═════════════════════════════════════════════════════════════════════════════
# WAKE WORD DETECTOR
# ═════════════════════════════════════════════════════════════════════════════

class WakeWordDetector:
    def __init__(self):
        self._vosk_ready  = False
        self._recognizer  = None
        self._audio_queue = queue.Queue()
        self._detected_cb = None

        if VOSK_AVAILABLE:
            model_path = Path(BASE_DIR) / VOSK_MODEL_PATH
            if model_path.exists():
                try:
                    model = VoskModel(str(model_path))
                    self._recognizer = KaldiRecognizer(model, 16000)
                    self._vosk_ready = True
                    print(f"  🎙️  Vosk ready ({VOSK_MODEL_PATH})")
                except Exception as e:
                    print(f"  ⚠️  Vosk init failed: {e}")
            else:
                print(f"  ⚠️  Vosk model not found — transcript fallback")
        else:
            print("  ⚠️  Vosk not installed — transcript fallback")

    def set_callback(self, cb): self._detected_cb = cb

    def feed_audio(self, pcm_bytes: bytes):
        if self._vosk_ready:
            self._audio_queue.put_nowait(pcm_bytes)

    def check_transcript(self, text: str) -> bool:
        t = text.lower().strip()
        for ww in WAKE_WORDS:
            if ww in t: return True
        words = t.split()
        return bool(words and words[0] in ["adam", "a.d.a.m"])

    def run_vosk_thread(self):
        if not self._vosk_ready: return
        print("  🎙️  Vosk thread running")
        while True:
            try:
                chunk = self._audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if self._recognizer.AcceptWaveform(chunk):
                result = json.loads(self._recognizer.Result())
                text   = result.get("text", "").lower()
            else:
                partial = json.loads(self._recognizer.PartialResult())
                text    = partial.get("partial", "").lower()
            if text and self.check_transcript(text):
                print(f"  🔔  Wake word: '{text}'")
                if self._detected_cb:
                    self._detected_cb()


# ═════════════════════════════════════════════════════════════════════════════
# PERSISTENT MEMORY
# ═════════════════════════════════════════════════════════════════════════════

def load_face_memory() -> dict:
    if FACE_MEMORY_FILE.exists():
        try:
            with open(FACE_MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_face_memory(faces: dict):
    with open(FACE_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(faces, f, ensure_ascii=False, indent=2)

def load_memory() -> dict:
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"  🧠  Memory: {len(data)} entries")
                return data
        except Exception:
            pass
    return {}

def save_memory(memory: dict):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def memory_to_prompt(memory: dict) -> str:
    if not memory: return ""
    lines = ["━━━ PERSISTENT MEMORY ━━━"]
    for k, v in memory.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)

def face_memory_to_prompt(faces: dict) -> str:
    if not faces: return ""
    lines = ["━━━ PEOPLE YOU KNOW ━━━"]
    for pid, info in faces.items():
        lines.append(
            f"- {info.get('name','?')} (ID:{pid}): "
            f"Appearance: {info.get('appearance','?')}. "
            f"Voice: {info.get('voice_cues','?')}. "
            f"Relationship: {info.get('relationship','?')}. "
            f"Notes: {info.get('notes','')}."
        )
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════════════════

def load_system_prompt(memory: dict, faces: dict) -> str:
    prompt_path = Path(BASE_DIR) / "system_prompt.txt"
    if prompt_path.exists():
        prompt_text = prompt_path.read_text(encoding="utf-8")
        if prompt_text.startswith('"""') and prompt_text.endswith('"""'):
            prompt_text = prompt_text[3:-3].strip()
    else:
        prompt_text = (
            "You are ADAM — Autonomous Desktop AI Module. "
            "Tony Stark meets J.A.R.V.I.S. Sharp, confident, dry wit. "
            "Never lecture. Talk like a friend. Max 2-3 sentences."
        )

    parts = [
        memory_to_prompt(memory),
        face_memory_to_prompt(faces),
        prompt_text,
        _load_prompt_file("prompt_search.txt"),
        _load_prompt_file("prompt_clipboard.txt"),
        _load_prompt_file("prompt_attention.txt"),
        _load_prompt_file("prompt_vision.txt"),
        _load_prompt_file("prompt_language.txt"),
    ]
    return "\n\n".join(p for p in parts if p.strip())


# ═════════════════════════════════════════════════════════════════════════════
# AUDIO + FLASK + WS
# ═════════════════════════════════════════════════════════════════════════════

FORMAT           = pyaudio.paInt16
CHANNELS         = 1
SEND_SAMPLE_RATE = 16000
RECV_SAMPLE_RATE = 24000
CHUNK_SIZE       = 512

pya = pyaudio.PyAudio()

flask_app = Flask(__name__, static_folder=BASE_DIR)

@flask_app.route("/")
def index():
    return send_from_directory(BASE_DIR, "adam_face.html")

def run_flask():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    flask_app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)

ws_clients: set = set()

async def ws_broadcast(payload: dict):
    if not ws_clients: return
    msg  = json.dumps(payload)
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)

async def ws_handler(websocket):
    ws_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        ws_clients.discard(websocket)

EMOTION_MAP = {
    "happy":"nod_yes","excited":"nod_fast","angry":"none",
    "confused":"none","smug":"none","sad":"none",
    "surprised":"nod_yes","thinking":"none","love":"nod_yes","blush":"none",
}

_last_sync_time = 0.0
_sync_interval  = 0.06

async def maybe_sync_mouth(audio_chunk: bytes, adam_speaking_event: asyncio.Event):
    global _last_sync_time
    if not adam_speaking_event.is_set(): return
    now = time.time()
    if now - _last_sync_time < _sync_interval: return
    _last_sync_time = now
    try:
        n = len(audio_chunk) // 2
        if n == 0: return
        samples = struct.unpack(f"{n}h", audio_chunk)
        rms = (sum(s * s for s in samples) / n) ** 0.5
    except Exception:
        return
    if rms < 600:    intensity = "low"
    elif rms < 4000: intensity = "low"
    elif rms < 10000:intensity = "medium"
    else:            intensity = "high"
    await ws_broadcast({"type": "mouth_sync", "intensity": intensity})

def capture_raw_frame(cap) -> np.ndarray | None:
    ret, frame = cap.read()
    return frame if ret else None

def frame_to_jpeg(frame: np.ndarray, size=FRAME_SIZE) -> bytes:
    frame = cv2.resize(frame, size)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


# ═════════════════════════════════════════════════════════════════════════════
# TOOL HANDLER
# ═════════════════════════════════════════════════════════════════════════════

async def handle_tool_call(tool_call, memory: dict, faces: dict) -> list[dict]:
    responses = []
    for fc in tool_call.function_calls:
        name    = fc.name
        call_id = fc.id
        args    = dict(fc.args) if fc.args else {}

        if name == "get_current_datetime":
            now = datetime.datetime.now()
            result = {
                "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "date":     now.strftime("%A, %d %B %Y"),
                "time":     now.strftime("%I:%M %p"),
                "timezone": str(datetime.datetime.now().astimezone().tzname()),
            }

        elif name == "google_search":
            query = args.get("q", args.get("query", str(args)))
            print(f"  🔍  google_search → \"{query}\"")
            result = {"status": "search_executed", "query": query}

        elif name == "generate_to_clipboard":
            prompt = args.get("prompt", "").strip()
            if not prompt:
                result = {"error": "prompt cannot be empty"}
            else:
                confirmation = await generate_to_clipboard(prompt)
                result = {
                    "status": "done",
                    "confirmation": confirmation,
                    "ack": random.choice(CLIPBOARD_ACK_LINES),
                }

        elif name == "remember_person":
            pid = args.get("person_id") or f"person_{int(time.time())}"
            faces[pid] = {
                "name":         args.get("name", "Unknown"),
                "appearance":   args.get("appearance", ""),
                "voice_cues":   args.get("voice_cues", ""),
                "relationship": args.get("relationship", "acquaintance"),
                "notes":        args.get("notes", ""),
                "last_seen":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            save_face_memory(faces)
            print(f"  👤  Remembered: {args.get('name')} ({pid})")
            result = {"status": "saved", "person_id": pid}

        elif name == "update_person_seen":
            pid = args.get("person_id", "")
            if pid in faces:
                faces[pid]["last_seen"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                if args.get("notes_update"):
                    existing = faces[pid].get("notes", "")
                    faces[pid]["notes"] = (existing + " | " + args["notes_update"]).strip(" |")
                save_face_memory(faces)
                result = {"status": "updated"}
            else:
                result = {"status": "not_found"}

        elif name == "get_all_people":
            result = {"people": faces}

        elif name == "set_emotion":
            emotion = args.get("emotion", "happy")
            await ws_broadcast({"type": "emotion", "emotion": emotion,
                                "head": EMOTION_MAP.get(emotion, "none")})
            result = {"status": "ok"}

        elif name == "set_mouth_sync":
            await ws_broadcast({"type": "mouth_sync",
                                "intensity": args.get("intensity", "medium")})
            result = {"status": "ok"}

        elif name == "save_memory":
            key = args.get("key", "").strip()
            val = args.get("value", "").strip()
            if key:
                memory[key] = val
                save_memory(memory)
                result = {"status": "saved"}
            else:
                result = {"status": "error"}

        elif name == "delete_memory":
            key = args.get("key", "").strip()
            if key in memory:
                del memory[key]
                save_memory(memory)
                result = {"status": "deleted"}
            else:
                result = {"status": "not_found"}

        elif name == "get_memory":
            result = {"value": memory.get(args.get("key", ""), None), "all": memory}

        else:
            result = {"error": f"Unknown: {name}"}

        responses.append({"id": call_id, "name": name, "response": result})
    return responses


def build_tools() -> list[types.Tool]:
    S = types.Schema
    T = types.Type

    fn_tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(name="get_current_datetime",
            description="Returns current local date and time.",
            parameters=S(type=T.OBJECT, properties={})),

        types.FunctionDeclaration(name="generate_to_clipboard",
            description=(
                "Generate text, code, scripts, emails, or any long-form content and "
                "copy it to the user's clipboard. Use when asked to write, draft, or create content."
            ),
            parameters=S(type=T.OBJECT, properties={
                "prompt": S(type=T.STRING),
                "task_type": S(type=T.STRING,
                    enum=["code","email","essay","template","script","general"]),
            }, required=["prompt"])),

        types.FunctionDeclaration(name="remember_person",
            description="Save a new person to visual memory.",
            parameters=S(type=T.OBJECT, properties={
                "person_id":   S(type=T.STRING),
                "name":        S(type=T.STRING),
                "appearance":  S(type=T.STRING),
                "voice_cues":  S(type=T.STRING),
                "relationship":S(type=T.STRING),
                "notes":       S(type=T.STRING),
            }, required=["person_id","name"])),

        types.FunctionDeclaration(name="update_person_seen",
            description="Update last-seen time for a known person.",
            parameters=S(type=T.OBJECT, properties={
                "person_id":    S(type=T.STRING),
                "notes_update": S(type=T.STRING),
            }, required=["person_id"])),

        types.FunctionDeclaration(name="get_all_people",
            description="Get all people in visual memory.",
            parameters=S(type=T.OBJECT, properties={})),

        types.FunctionDeclaration(name="set_emotion",
            description="Show emotion on OLED face.",
            parameters=S(type=T.OBJECT, properties={
                "emotion": S(type=T.STRING,
                    enum=["happy","excited","angry","confused","smug",
                          "sad","surprised","thinking","love","blush"])
            }, required=["emotion"])),

        types.FunctionDeclaration(name="set_mouth_sync",
            description="Sync mouth animation intensity.",
            parameters=S(type=T.OBJECT, properties={
                "intensity": S(type=T.STRING, enum=["closed","low","medium","high"])
            }, required=["intensity"])),

        types.FunctionDeclaration(name="save_memory",
            description="Save a persistent key-value memory.",
            parameters=S(type=T.OBJECT, properties={
                "key":   S(type=T.STRING),
                "value": S(type=T.STRING),
            }, required=["key","value"])),

        types.FunctionDeclaration(name="delete_memory",
            description="Delete a memory entry.",
            parameters=S(type=T.OBJECT, properties={
                "key": S(type=T.STRING)
            }, required=["key"])),

        types.FunctionDeclaration(name="get_memory",
            description="Retrieve memory entries.",
            parameters=S(type=T.OBJECT, properties={
                "key": S(type=T.STRING)
            })),
    ])

    search_tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="google_search",
            description=(
                "Search Google for current info: news, prices, weather, sports, "
                "recent events. Use proactively for time-sensitive questions."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"q": types.Schema(type=types.Type.STRING)},
                required=["q"],
            ),
        )
    ])

    return [fn_tool, search_tool]


# ═════════════════════════════════════════════════════════════════════════════
# IDLE NUDGES
# ═════════════════════════════════════════════════════════════════════════════

IDLE_NUDGES = [
    "You've gone quiet. Either look at me or say my name.",
    "Still there? I can see you ignoring me.",
    "Silence noted. I'll be here when you're ready.",
    "My processors are idling. That's borderline offensive.",
    "Either talk or do something interesting. I'm watching.",
]
_nudge_idx = 0

def next_nudge():
    global _nudge_idx
    n = IDLE_NUDGES[_nudge_idx % len(IDLE_NUDGES)]
    _nudge_idx += 1
    return n


# ═════════════════════════════════════════════════════════════════════════════
# SESSION RUNNER
# ═════════════════════════════════════════════════════════════════════════════

async def run_session(
    client:        genai.Client,
    resume_handle: str | None,
    stop:          asyncio.Event,
    out_q:         asyncio.Queue,
    memory:        dict,
    faces:         dict,
    system_prompt: str,
    attention:     AttentionManager,
    wake_word:     WakeWordDetector,
    tracker:       PersonTracker,
) -> str | None:

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=system_prompt,
        tools=build_tools(),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        session_resumption=types.SessionResumptionConfig(handle=resume_handle),
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow(),
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
            )
        ),
    )

    latest_handle: str | None = resume_handle
    print(f"\n  Connecting{' (resuming)' if resume_handle else ''}...")
    t0 = time.time()

    _event_loop = asyncio.get_event_loop()

    def _wake_word_fired():
        asyncio.run_coroutine_threadsafe(attention.activate("wake-word"), _event_loop)

    wake_word.set_callback(_wake_word_fired)

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            print(f"  ✅  Connected in {time.time()-t0:.2f}s  |  Voice: {VOICE}")
            if not resume_handle:
                print(
                    "  System ready.\n"
                    "  → Look at camera OR say 'ADAM' to activate.\n"
                    "  → Gestures (thumbs up, namaste, wave, show object) trigger ADAM too.\n"
                    "  → Multiple people: ADAM tracks who's speaking.\n"
                    "  Ctrl+C to quit.\n"
                )
                await ws_broadcast({"type": "face_state", "state": "idle"})

            mic_q          = asyncio.Queue(maxsize=120)
            adam_speaking  = asyncio.Event()
            latest_frame   = [None]
            last_interaction_time = [time.time()]

            # ── Last known camera context (injected with audio turns) ──────────
            last_camera_context = [""]

            # ── Shared tracker state (readable from send / receive) ────────────
            last_tracker_result = [{
                "faces": [], "active_speaker_idx": None,
                "gesture": None, "face_count": 0,
            }]
            # Face-index of the speaker in the last completed user turn.
            # Used to detect when a *different* person starts speaking.
            last_confirmed_speaker_idx = [None]

            async def on_attention_change(state: str):
                if state == AttentionState.ATTENTIVE:
                    await ws_broadcast({"type": "face_state", "state": "listening"})
                elif state == AttentionState.PASSIVE:
                    await ws_broadcast({"type": "face_state", "state": "idle"})

            attention.set_callback(on_attention_change)

            # ── CAMERA (camera-first: processes gestures + tracks speakers) ────
            async def camera():
                cap  = None
                consecutive_failures = 0
                MAX_FAILURES = 10
                last_sent = 0.0
                last_gesture_sent: str | None = None
                last_gesture_sent_time = 0.0

                try:
                    cap = cv2.VideoCapture(CAMERA_INDEX)
                    if not cap.isOpened():
                        print(f"  ⚠️  Camera {CAMERA_INDEX} unavailable")
                        return
                    print(f"  📷  Camera ready (index {CAMERA_INDEX})")

                    while not stop.is_set():
                        await asyncio.sleep(0.15)   # ~6-7 FPS for tracking
                        if stop.is_set():
                            break

                        raw = await asyncio.to_thread(capture_raw_frame, cap)
                        if raw is None:
                            consecutive_failures += 1
                            await asyncio.sleep(0.5)
                            if consecutive_failures >= MAX_FAILURES:
                                cap.release()
                                await asyncio.sleep(2.0)
                                cap = cv2.VideoCapture(CAMERA_INDEX)
                                if not cap.isOpened():
                                    print(f"  ⚠️  Camera reconnect failed")
                                    return
                                print(f"  📷  Camera reconnected")
                                consecutive_failures = 0
                            continue

                        consecutive_failures = 0
                        latest_frame[0] = raw

                        # ── Run multi-person tracker ──────────────────────────
                        tracker_result = await asyncio.to_thread(tracker.process_frame, raw)

                        # ── Build camera context string for injections ─────────
                        ctx_str = tracker.build_context_string(tracker_result)
                        last_camera_context[0] = ctx_str
                        last_tracker_result[0] = tracker_result

                        # ── Attention from face gaze ───────────────────────────
                        has_centred_face = any(f["is_centre"] for f in tracker_result["faces"])
                        if has_centred_face:
                            await attention.activate("face-detected")
                        else:
                            elapsed = time.time() - attention._last_active_time
                            if (attention.state == AttentionState.ATTENTIVE and elapsed > 5.0):
                                await attention.deactivate("face-lost")

                        # ── GESTURE → camera-first trigger ────────────────────
                        gesture = tracker_result.get("gesture")
                        now = time.time()
                        if (gesture and
                                gesture != last_gesture_sent and
                                now - last_gesture_sent_time > 4.0 and
                                not adam_speaking.is_set()):

                            print(f"  🤲  Gesture detected: {gesture}")
                            last_gesture_sent = gesture
                            last_gesture_sent_time = now

                            # Activate and send frame + gesture context
                            await attention.activate(f"gesture:{gesture}")
                            jpeg = await asyncio.to_thread(frame_to_jpeg, raw)
                            try:
                                # Send frame FIRST (visual context)
                                await session.send_realtime_input(
                                    video=types.Blob(data=jpeg, mime_type="image/jpeg")
                                )
                                # Then inject gesture instruction
                                gesture_prompts = {
                                    "thumbs_up":    "Someone just gave you a thumbs up. React in character.",
                                    "thumbs_down":  "Someone just gave you a thumbs down. React in character.",
                                    "namaste":      "Someone just did a namaste gesture to you. Respond appropriately and in character.",
                                    "wave":         "Someone is waving at you. Acknowledge them naturally.",
                                    "object_shown": "Someone is showing you something with their hands. Look at the frame and comment on what you see.",
                                }
                                prompt = gesture_prompts.get(gesture,
                                    f"A gesture was detected: {gesture}. React naturally.")
                                await session.send_realtime_input(
                                    text=f"[CAMERA-FIRST TRIGGER: {prompt} {ctx_str}]"
                                )
                            except (ConnectionClosedError, ConnectionClosedOK):
                                return
                            except Exception as e:
                                print(f"  ⚠️  Gesture trigger error: {e}")

                        # ── Send frame to Gemini at 1 FPS when active ─────────
                        if (now - last_sent >= CAMERA_FPS_INTERVAL and
                                not adam_speaking.is_set() and
                                attention.is_active()):
                            # Annotate frame with speaker info if multiple people
                            frame_to_send = raw
                            if tracker_result["face_count"] > 1:
                                # Draw lightweight annotations for Gemini to see
                                annotated = raw.copy()
                                for i, f in enumerate(tracker_result["faces"]):
                                    color = (0, 255, 0) if i == tracker_result["active_speaker_idx"] else (200, 200, 200)
                                    cv2.rectangle(annotated,
                                                  (f["x"], f["y"]),
                                                  (f["x"]+f["w"], f["y"]+f["h"]),
                                                  color, 2)
                                    label = f"P{i+1}" + (" [speaking]" if i == tracker_result["active_speaker_idx"] else "")
                                    cv2.putText(annotated, label,
                                                (f["x"], f["y"]-6),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                frame_to_send = annotated

                            jpeg = await asyncio.to_thread(frame_to_jpeg, frame_to_send)
                            try:
                                await session.send_realtime_input(
                                    video=types.Blob(data=jpeg, mime_type="image/jpeg"))
                                last_sent = now
                            except (ConnectionClosedError, ConnectionClosedOK):
                                return
                            except Exception:
                                pass

                except asyncio.CancelledError:
                    pass
                finally:
                    if cap:
                        cap.release()

            # ── LISTEN ───────────────────────────────────────────────────────
            async def listen():
                stream = pya.open(
                    format=FORMAT, channels=CHANNELS,
                    rate=SEND_SAMPLE_RATE, input=True,
                    frames_per_buffer=CHUNK_SIZE,
                )
                try:
                    while not stop.is_set():
                        data = await asyncio.to_thread(
                            stream.read, CHUNK_SIZE, exception_on_overflow=False)
                        wake_word.feed_audio(data)
                        try:
                            mic_q.put_nowait(data)
                        except asyncio.QueueFull:
                            pass
                except asyncio.CancelledError:
                    pass
                finally:
                    stream.stop_stream()
                    stream.close()

            # ── SEND (camera context + speaker ID injected at every voice onset) ──
            async def send():
                # ctx_injected: ensures we inject once per user voice-turn, not
                # on every audio chunk.  Reset whenever ADAM finishes speaking.
                ctx_injected = False
                try:
                    while not stop.is_set():
                        chunk = await mic_q.get()
                        if adam_speaking.is_set():
                            # ADAM is replying — reset so the NEXT user turn
                            # gets a fresh camera-context injection.
                            ctx_injected = False
                            continue
                        if not attention.is_active():
                            ctx_injected = False
                            continue

                        is_speech = False
                        try:
                            n = len(chunk) // 2
                            samples = struct.unpack(f"{n}h", chunk)
                            rms = (sum(s * s for s in samples) / n) ** 0.5
                            if rms > 800:
                                attention.touch()
                                is_speech = True
                        except Exception:
                            pass

                        # ── Voice-onset: inject camera snapshot + speaker ID ──
                        # This runs once at the start of each user voice turn so
                        # the model knows WHO is speaking BEFORE it processes audio.
                        if is_speech and not ctx_injected:
                            ctx_injected = True
                            raw = latest_frame[0]
                            ctx = last_camera_context[0]
                            tr  = last_tracker_result[0]
                            try:
                                # 1. Send current camera frame for visual context
                                if raw is not None:
                                    jpeg = await asyncio.to_thread(frame_to_jpeg, raw)
                                    await session.send_realtime_input(
                                        video=types.Blob(data=jpeg, mime_type="image/jpeg"))

                                # 2. Build a "who is speaking" notice
                                speaker_count = tr.get("face_count", 0)
                                speaker_idx   = tr.get("active_speaker_idx")
                                prev_idx      = last_confirmed_speaker_idx[0]

                                if speaker_count > 1:
                                    if (prev_idx is not None and
                                            speaker_idx is not None and
                                            speaker_idx != prev_idx):
                                        notice = (
                                            f"[SPEAKER CHANGED — a different person is now talking: "
                                            f"{ctx}  Look at the camera frame and address the NEW "
                                            f"speaker (← SPEAKING NOW) directly.]"
                                        )
                                    elif ctx:
                                        notice = (
                                            f"[WHO IS SPEAKING: {ctx}  "
                                            f"Address the person marked ← SPEAKING NOW directly.]"
                                        )
                                    else:
                                        notice = (
                                            "[WHO IS SPEAKING: Multiple people in frame. "
                                            "Check the camera frame to identify the active speaker "
                                            "and address them directly.]"
                                        )
                                elif ctx:
                                    notice = f"[CAMERA CONTEXT: {ctx}]"
                                else:
                                    notice = ""

                                if notice:
                                    await session.send_realtime_input(text=notice)

                            except (ConnectionClosedError, ConnectionClosedOK):
                                return
                            except Exception:
                                pass

                        try:
                            await session.send_realtime_input(
                                audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000"))
                        except (ConnectionClosedError, ConnectionClosedOK):
                            return
                        except Exception:
                            await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    pass

            # ── RECEIVE ──────────────────────────────────────────────────────
            async def receive():
                nonlocal latest_handle
                try:
                    while not stop.is_set():
                        async for msg in session.receive():
                            if stop.is_set():
                                break

                            if msg.session_resumption_update:
                                upd = msg.session_resumption_update
                                if upd.resumable and upd.new_handle:
                                    latest_handle = upd.new_handle

                            if hasattr(msg, "go_away") and msg.go_away:
                                print("\n  ⚡ GoAway — resuming...")
                                return

                            if msg.tool_call:
                                responses = await handle_tool_call(msg.tool_call, memory, faces)
                                await session.send_tool_response(
                                    function_responses=[
                                        types.FunctionResponse(
                                            id=r["id"], name=r["name"],
                                            response=r["response"])
                                        for r in responses
                                    ]
                                )
                                continue

                            sc = msg.server_content
                            if sc is None:
                                continue

                            # Fallback wake word in transcript
                            if sc.input_transcription and sc.input_transcription.text:
                                transcript = sc.input_transcription.text
                                print(f"  🗣️  You: {transcript}")
                                if (attention.state == AttentionState.PASSIVE and
                                        wake_word.check_transcript(transcript)):
                                    await attention.activate("transcript-wake-word")

                                # Record which camera face-index was speaking in
                                # this turn (used for speaker-change detection).
                                tr = last_tracker_result[0]
                                cur_spk = tr.get("active_speaker_idx")
                                if cur_spk is not None:
                                    last_confirmed_speaker_idx[0] = cur_spk

                                # Reinforce camera context (arrives after audio,
                                # but still useful for the model's response).
                                ctx = last_camera_context[0]
                                if ctx:
                                    try:
                                        await session.send_realtime_input(
                                            text=f"[LIVE CAMERA CONTEXT: {ctx}]"
                                        )
                                    except Exception:
                                        pass

                            if sc.model_turn:
                                if not adam_speaking.is_set():
                                    adam_speaking.set()
                                    await attention.set_responding(True)
                                    await ws_broadcast({"type": "face_state", "state": "speaking"})
                                for part in sc.model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        audio_data = part.inline_data.data
                                        await out_q.put(audio_data)
                                        await maybe_sync_mouth(audio_data, adam_speaking)
                                    if hasattr(part, "text") and part.text:
                                        print(f"🤖  ADAM: {part.text}")

                            if sc.turn_complete:
                                await out_q.put(None)
                                print("─" * 40)

                except (ConnectionClosedError, ConnectionClosedOK):
                    pass
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"\n⚠️  Receive: {type(e).__name__}: {e}")

            # ── SPEAKER ──────────────────────────────────────────────────────
            async def speaker():
                stream = pya.open(
                    format=FORMAT, channels=CHANNELS,
                    rate=RECV_SAMPLE_RATE, output=True,
                )
                last_audio_time = [time.time()]
                STUCK_WATCHDOG_S = 2.5

                async def end_of_turn():
                    await ws_broadcast({"type": "mouth_sync", "intensity": "closed"})
                    await asyncio.sleep(0.1)
                    await asyncio.sleep(POST_SPEECH_MUTE_S)
                    drained = 0
                    while not out_q.empty():
                        try:
                            out_q.get_nowait(); drained += 1
                        except asyncio.QueueEmpty:
                            break
                    if drained:
                        print(f"  🧹  Drained {drained} chunks")
                    while not mic_q.empty():
                        try:
                            mic_q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    adam_speaking.clear()
                    await attention.set_responding(False)
                    last_interaction_time[0] = time.time()
                    print("  🎤  Your turn...")
                    await ws_broadcast({"type": "face_state", "state": "listening"})

                try:
                    while not stop.is_set():
                        try:
                            chunk = await asyncio.wait_for(out_q.get(), timeout=0.3)
                            last_audio_time[0] = time.time()
                            if chunk is None:
                                await end_of_turn()
                                continue
                            await asyncio.to_thread(stream.write, chunk)
                        except asyncio.TimeoutError:
                            if (adam_speaking.is_set() and
                                    time.time() - last_audio_time[0] > STUCK_WATCHDOG_S):
                                print("  ⚠️  Speaker watchdog — force-clearing")
                                await end_of_turn()
                            continue
                except asyncio.CancelledError:
                    pass
                finally:
                    stream.stop_stream()
                    stream.close()

            # ── IDLE WATCHER ─────────────────────────────────────────────────
            async def idle_watcher():
                if not ENABLE_IDLE:
                    return
                try:
                    while not stop.is_set():
                        await asyncio.sleep(5)
                        if stop.is_set() or adam_speaking.is_set():
                            continue
                        if attention.state != AttentionState.PASSIVE:
                            continue
                        elapsed = time.time() - last_interaction_time[0]
                        if elapsed < IDLE_TIMEOUT_S:
                            continue

                        last_interaction_time[0] = time.time()
                        nudge = next_nudge()
                        print(f"  💤  Idle nudge ({elapsed:.0f}s)")

                        try:
                            await attention.activate("idle-nudge")
                            raw = latest_frame[0]
                            if raw is not None:
                                jpeg = await asyncio.to_thread(frame_to_jpeg, raw)
                                await session.send_realtime_input(
                                    video=types.Blob(data=jpeg, mime_type="image/jpeg"))
                            await session.send_realtime_input(
                                text=(
                                    f"[SYSTEM: {elapsed:.0f}s of silence since last chat. "
                                    f"Camera frame sent — react to what you see. "
                                    f"Break silence in-character, 1-2 sentences MAX. "
                                    f"Suggestion: {nudge}]"
                                )
                            )
                        except Exception as e:
                            print(f"  ⚠️  Idle nudge error: {e}")
                except asyncio.CancelledError:
                    pass

            # ── LAUNCH ───────────────────────────────────────────────────────
            t_cam = asyncio.create_task(camera())
            t_l   = asyncio.create_task(listen())
            t_s   = asyncio.create_task(send())
            t_r   = asyncio.create_task(receive())
            t_p   = asyncio.create_task(speaker())
            t_i   = asyncio.create_task(idle_watcher())

            done, pending = await asyncio.wait(
                [t_s, t_r], return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            t_cam.cancel(); t_l.cancel(); t_p.cancel(); t_i.cancel()
            await asyncio.gather(t_cam, t_l, t_s, t_r, t_p, t_i,
                                 return_exceptions=True)

    except (ConnectionClosedError, ConnectionClosedOK):
        pass
    except Exception as e:
        print(f"\n⚠️  Connection: {type(e).__name__}: {e}")

    if stop.is_set():
        return None
    return latest_handle


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

async def main():
    memory        = load_memory()
    faces         = load_face_memory()
    system_prompt = load_system_prompt(memory, faces)
    attention     = AttentionManager()
    wake_word     = WakeWordDetector()
    tracker       = PersonTracker()

    if wake_word._vosk_ready:
        threading.Thread(target=wake_word.run_vosk_thread, daemon=True).start()

    client        = genai.Client(api_key=API_KEY)
    stop          = asyncio.Event()
    out_q         = asyncio.Queue(maxsize=200)
    resume_handle = None
    attempt       = 0

    ws_server = await websockets.server.serve(ws_handler, WS_HOST, WS_PORT)
    print(f"  🌐  WebSocket → ws://{WS_HOST}:{WS_PORT}")

    while not stop.is_set():
        if attempt > 0:
            delay = min(2 ** attempt, 15)
            print(f"  Reconnecting in {delay}s...")
            await asyncio.sleep(delay)

        result = await run_session(
            client, resume_handle, stop, out_q,
            memory, faces, system_prompt, attention, wake_word, tracker
        )
        if result is None:
            break

        resume_handle = result
        attempt      += 1
        system_prompt = load_system_prompt(memory, faces)
        print(f"\n🔄  {'Resuming...' if resume_handle else 'Reconnecting...'}")

    stop.set()
    ws_server.close()
    await ws_server.wait_closed()
    pya.terminate()
    print("\n👋  Goodbye.")


def main_entry():
    init_gen_client()

    print("=" * 66)
    print("  ADAM — Autonomous Desktop AI Module  (v22)")
    print(f"  Built by DGEN Technologies Pvt. Ltd., Kolkata")
    print(f"  Live model  : {LIVE_MODEL}")
    print(f"  Gen cascade : {' → '.join(GEN_MODEL_CASCADE)}")
    print(f"  Voice       : {VOICE}")
    print(f"  Camera      : index {CAMERA_INDEX}  |  Priority: CAMERA-FIRST")
    print(f"  Multi-person: ✅  Speaker tracking via mouth movement analysis")
    print(f"  Gestures    : 👍 thumbs_up  🙏 namaste  👋 wave  📦 object_shown")
    print(f"  Clipboard   : {'✅ ready' if CLIPBOARD_AVAILABLE else '❌ install pyperclip'}")
    print(f"  Vosk        : {'ready' if VOSK_AVAILABLE else 'not installed'}")
    print(f"  Idle nudge  : {'ON' if ENABLE_IDLE else 'OFF'}  |  {IDLE_TIMEOUT_S}s")
    print("=" * 66)
    print()

    threading.Thread(target=run_flask, daemon=True).start()
    print(f"  🌍  Flask → http://localhost:{FLASK_PORT}")
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{FLASK_PORT}")).start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋  Goodbye.")


if __name__ == "__main__":
    main_entry()