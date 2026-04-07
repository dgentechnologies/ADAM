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
FACES_DIR        = Path(BASE_DIR) / "faces"          # saved face crop photos

# Gesture detection sensitivity
GESTURE_MOTION_THRESHOLD  = 0.025   # fraction of frame area changed to trigger gesture notice
MOUTH_MOVEMENT_SENSITIVITY = 6.0    # brightness delta in mouth ROI to count as "speaking"
SPEAKER_INERTIA_FRAMES     = 10     # frames before switching active speaker
WAVE_MIN_LATERAL_MOVEMENT  = 0.22   # min horizontal blob travel (fraction of frame) for wave
WAVE_MOTION_THRESHOLD      = 0.07   # motion fraction required specifically for wave
WAVE_SKIN_THRESHOLD        = 0.10   # skin fraction in upper half required for wave
FACE_DETECT_MIN_NEIGHBORS  = 4      # lower = fewer missed faces (was 5)

# Camera preview window
SHOW_PREVIEW     = True                   # Set False to run headless
PREVIEW_WIN_NAME = "ADAM — Camera Preview"
PREVIEW_SIZE     = (640, 480)             # (width, height) of the preview window

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

# Queue used to ship preview data from the camera coroutine to the display
# thread.  maxsize=1 ensures the thread always shows the freshest frame.
_preview_queue: queue.Queue = queue.Queue(maxsize=1)
# Set by main_entry() to stop the preview thread cleanly.
_preview_stop  = threading.Event()

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
        # Wave tracking: persistent blob-cx history for lateral movement detection
        self._blob_cx_history: deque = deque(maxlen=8)
        # ADAM-speaking flag — suppresses speaker-id updates while ADAM is talking
        self._adam_speaking: bool = False

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
            gray, scaleFactor=1.1, minNeighbors=FACE_DETECT_MIN_NEIGHBORS,
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
        # Frozen while ADAM is talking to prevent false speaker switches.
        if not self._adam_speaking and mouth_deltas:
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
        Gesture detection using frame differencing + skin tone analysis.
        Detects: thumbs_up, namaste, wave, object_shown.

        Wave fix: requires sustained lateral hand movement (blob cx range >=
        WAVE_MIN_LATERAL_MOVEMENT) so hair-fixing / head-scratching motions
        that stay near the centre are NOT mis-classified as waves.
        """
        now = time.time()
        h, w = frame.shape[:2]

        # Downsample for speed
        small = cv2.resize(frame, (160, 120))

        # Define skin range once (used in both tracking and classification)
        lower_skin = np.array([0, 20, 70],  dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)

        # ── Always track blob-cx for wave detection (even during cooldown) ────
        upper_track = small[:60, :]
        skin_track  = cv2.inRange(
            cv2.cvtColor(upper_track, cv2.COLOR_BGR2HSV), lower_skin, upper_skin)
        skin_frac_t = np.sum(skin_track > 0) / skin_track.size

        if skin_frac_t >= 0.06:
            ctrs_t, _ = cv2.findContours(skin_track, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            if ctrs_t:
                lg_t = max(ctrs_t, key=cv2.contourArea)
                if cv2.contourArea(lg_t) >= 60:
                    bxt, _, bwt, _ = cv2.boundingRect(lg_t)
                    self._blob_cx_history.append((bxt + bwt / 2) / 160)
        else:
            # No skin visible — clear stale history
            if self._blob_cx_history:
                self._blob_cx_history.clear()

        if self._prev_frame_small is None:
            self._prev_frame_small = small
            return None

        # ── Gesture cooldown ──────────────────────────────────────────────────
        if now - self._last_gesture_time < self._gesture_cooldown:
            self._prev_frame_small = small.copy()
            return None

        # ── Motion detection ──────────────────────────────────────────────────
        diff = cv2.absdiff(small, self._prev_frame_small)
        self._prev_frame_small = small.copy()

        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
        motion_fraction = np.sum(thresh > 0) / thresh.size

        # ── Wave: check with its OWN stricter threshold + lateral-movement proof ──
        # Hair-fixing stays near the centre and produces a small cx range.
        # A real wave travels from one side (cx<0.28 or cx>0.72) across the frame.
        if (len(self._blob_cx_history) >= 4 and
                motion_fraction > WAVE_MOTION_THRESHOLD and
                skin_frac_t > WAVE_SKIN_THRESHOLD):
            cx_range    = max(self._blob_cx_history) - min(self._blob_cx_history)
            has_side    = any(cx < 0.28 or cx > 0.72 for cx in self._blob_cx_history)
            if cx_range >= WAVE_MIN_LATERAL_MOVEMENT and has_side:
                self._last_gesture_time = now
                self._blob_cx_history.clear()
                return "wave"

        if motion_fraction < GESTURE_MOTION_THRESHOLD:
            return None  # Not enough global motion for other gestures

        # ── Skin tone check for remaining gestures ────────────────────────────
        upper_half = small[:60, :]
        hsv        = cv2.cvtColor(upper_half, cv2.COLOR_BGR2HSV)
        skin_mask  = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_frac  = np.sum(skin_mask > 0) / skin_mask.size

        if skin_frac < 0.08:
            return None

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
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

        # Thumbs up: tall narrow blob in upper half
        if bh > bw * 0.8 and blob_cy_norm < 0.5 and skin_frac > 0.10:
            if aspect_ratio < 0.8:
                self._last_gesture_time = now
                return "thumbs_up"

        # Showing object: large motion in centre of frame
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

    # ── Speaker state management ──────────────────────────────────────────────

    def set_adam_speaking(self, flag: bool):
        """
        Call with True when ADAM starts speaking, False when ADAM finishes.
        Freezes active_speaker detection so the tracker does NOT flip the
        current speaker based on mouth movement during ADAM's playback.
        When switching back to False the inertia is reset so the next human
        speaker is detected cleanly.
        """
        if flag == self._adam_speaking:
            return
        self._adam_speaking = flag
        if not flag:
            # ADAM just finished — clear stale speaker so the next person
            # who moves their mouth becomes the new detected speaker.
            self._speaker_inertia   = 0
            self._active_speaker_id = None
            self._mouth_history.clear()

    def reset_for_new_turn(self):
        """
        Reset all mouth and speaker tracking at the end of a conversation turn.
        Called after ADAM's turn_complete so the next voice onset gets a fresh start.
        """
        self._speaker_inertia   = 0
        self._active_speaker_id = None
        self._mouth_history.clear()


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
        photo_note = f" [Photo: {info['photo_path']}]" if info.get('photo_path') else ""
        lines.append(
            f"- {info.get('name','?')} (ID:{pid}){photo_note}: "
            f"Appearance: {info.get('appearance','?')}. "
            f"Voice: {info.get('voice_cues','?')}. "
            f"Relationship: {info.get('relationship','?')}. "
            f"Notes: {info.get('notes','')}."
        )
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Memory-saving behaviour injected into every session prompt.
# ─────────────────────────────────────────────────────────────────────────────
_MEMORY_SAVE_GUIDE = """\
━━━ AUTOMATIC MEMORY RULES ━━━
- When a user tells you a story or shares a memorable event → IMMEDIATELY call
  save_story(topic="<short topic>", content="<full story>") to record it.
- When user says “remember [X]” / “don’t forget [X]” / “keep in mind” →
  call save_memory(key="<topic>", value="<what to remember>") right away and
  confirm: “Got it, I’ll remember that.”
- When user gives you a commitment / task (“you will do X”, “make sure you X”) →
  save_memory(key="commitment_<topic>", value="<commitment>").
- When meeting / recognising someone: use remember_person() for appearance and
  voice cues; optionally call save_person_photo() while you can see their face.
- Save ANYTHING that the user explicitly asks you to remember — no exceptions.
"""


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
        _MEMORY_SAVE_GUIDE,
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
# CAMERA PREVIEW — live debug window
# ═════════════════════════════════════════════════════════════════════════════

def draw_preview_frame(data: dict) -> np.ndarray:
    """
    Render face-tracking annotations onto a copy of the camera frame.

    Draws:
      • Face bounding boxes  (green + thick for active speaker, grey for others)
      • Person label P1 / P2 … with '◉ SPEAKING' badge on the active speaker
      • Mouth-ROI rectangle per face (cyan/blue outline)
      • Semi-transparent status bar at the bottom (ADAM state + face count)
      • Gesture badge top-right corner — fades after 3 seconds
      • Header label 'ADAM  Camera Monitor'
    """
    raw           = data["frame"]
    tr            = data["tracker_result"]
    attn_state    = data["attention_state"]
    adam_speaking = data["adam_speaking"]
    gesture       = data.get("gesture")
    gesture_ts    = data.get("gesture_ts", 0.0)

    pw, ph = PREVIEW_SIZE
    frame  = cv2.resize(raw, (pw, ph))

    raw_h, raw_w = raw.shape[:2]
    sx = pw / raw_w
    sy = ph / raw_h

    speaker_idx = tr.get("active_speaker_idx")

    # ── Per-face annotations ──────────────────────────────────────────────────
    for i, f in enumerate(tr.get("faces", [])):
        is_spk = (i == speaker_idx)

        # Scale bounding box to preview size
        fx = int(f["x"] * sx)
        fy = int(f["y"] * sy)
        fw = int(f["w"] * sx)
        fh = int(f["h"] * sy)

        # Active speaker → bright green / thick; others → cool grey / thin
        box_color = (0, 230, 80)  if is_spk else (160, 160, 160)
        box_thick = 3             if is_spk else 1
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_color, box_thick)

        # Person label
        label   = f"P{i + 1}"
        if is_spk:
            label += " ◉ SPEAKING"
        lbl_y = fy - 8 if fy > 20 else fy + fh + 18
        cv2.putText(frame, label, (fx + 4, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, box_color, 2, cv2.LINE_AA)

        # Mouth ROI — lower 35 % of face, inner 60 % width
        my = fy + int(fh * 0.6)
        mh = int(fh * 0.35)
        mx = fx + int(fw * 0.2)
        mw = int(fw * 0.6)
        my = max(0, min(my, ph - 1))
        mx = max(0, min(mx, pw - 1))
        mh = min(mh, ph - my)
        mw = min(mw, pw - mx)
        if mh > 0 and mw > 0:
            mouth_color = (0, 200, 255) if is_spk else (100, 100, 200)
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), mouth_color, 1)
            cv2.putText(frame, "mouth", (mx, my - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, mouth_color, 1, cv2.LINE_AA)

    # ── Semi-transparent status bar at bottom ────────────────────────────────
    bar_h   = 36
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, ph - bar_h), (pw, ph), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    if adam_speaking:
        state_txt   = "ADAM SPEAKING"
        state_color = (0, 220, 255)
    elif attn_state == AttentionState.RESPONDING:
        state_txt   = "RESPONDING"
        state_color = (0, 180, 255)
    elif attn_state == AttentionState.ATTENTIVE:
        state_txt   = "ATTENTIVE"
        state_color = (0, 230, 80)
    else:
        state_txt   = "PASSIVE"
        state_color = (120, 120, 120)

    cv2.putText(frame, state_txt, (10, ph - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, state_color, 2, cv2.LINE_AA)

    face_txt = f"Faces: {tr.get('face_count', 0)}"
    cv2.putText(frame, face_txt, (pw - 110, ph - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Gesture badge — top-right, fades out over 3 s ────────────────────────
    age = time.time() - gesture_ts
    if gesture and age < 3.0:
        badge_map = {
            "thumbs_up":    "THUMBS UP",
            "thumbs_down":  "THUMBS DOWN",
            "namaste":      "NAMASTE",
            "wave":         "WAVE",
            "object_shown": "OBJECT SHOWN",
        }
        badge_txt = badge_map.get(gesture, gesture.upper())
        alpha     = max(0.0, 1.0 - age / 3.0)
        b_color   = (int(50 * alpha), int(255 * alpha), int(120 * alpha))
        (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        bx = pw - tw - 14
        by = 14
        cv2.rectangle(frame, (bx - 6, by - 4), (bx + tw + 6, by + th + 6),
                      (20, 20, 20), -1)
        cv2.putText(frame, badge_txt, (bx, by + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, b_color, 2, cv2.LINE_AA)

    # ── Header label ──────────────────────────────────────────────────────────
    cv2.putText(frame, "ADAM  Camera Monitor  [Q = close]", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


def run_preview_thread():
    """
    Dedicated daemon thread for cv2.imshow().
    Reads packed data dicts from _preview_queue and renders the preview window.
    Press 'Q' inside the window to close the preview (stops ADAM too).
    """
    if not SHOW_PREVIEW:
        return

    window_open = False
    while not _preview_stop.is_set():
        try:
            data = _preview_queue.get(timeout=0.3)
        except queue.Empty:
            if window_open:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    _preview_stop.set()
                    break
            continue

        annotated = draw_preview_frame(data)

        if not window_open:
            cv2.namedWindow(PREVIEW_WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(PREVIEW_WIN_NAME, *PREVIEW_SIZE)
            window_open = True

        cv2.imshow(PREVIEW_WIN_NAME, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            _preview_stop.set()
            break

    if window_open:
        try:
            cv2.destroyWindow(PREVIEW_WIN_NAME)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# TOOL HANDLER
# ═════════════════════════════════════════════════════════════════════════════

async def handle_tool_call(tool_call, memory: dict, faces: dict,
                           current_frame=None) -> list[dict]:
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

        elif name == "save_story":
            topic   = args.get("topic", "untitled").strip().replace(" ", "_")
            content = args.get("content", "").strip()
            if content:
                date_key = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                key      = f"story_{topic}_{date_key}"
                memory[key] = content
                save_memory(memory)
                print(f"  📖  Story saved: '{key}' ({len(content)} chars)")
                result = {"status": "saved", "key": key}
            else:
                result = {"status": "error", "reason": "content cannot be empty"}

        elif name == "save_person_photo":
            pid = args.get("person_id", "").strip()
            if not pid:
                result = {"status": "error", "reason": "person_id required"}
            elif current_frame is None:
                result = {"status": "error", "reason": "no camera frame available"}
            else:
                try:
                    FACES_DIR.mkdir(exist_ok=True)
                    # Try to find and crop the face from the frame
                    gray_c = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                    gray_c = cv2.equalizeHist(gray_c)
                    face_cas = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                    detected = face_cas.detectMultiScale(
                        gray_c, 1.1, FACE_DETECT_MIN_NEIGHBORS, minSize=(60, 60))
                    if len(detected) > 0:
                        fx, fy, fw, fh = max(detected, key=lambda f: f[2] * f[3])
                        pad = int(fw * 0.3)
                        ch, cw = current_frame.shape[:2]
                        x1 = max(0, fx - pad); y1 = max(0, fy - pad)
                        x2 = min(cw, fx + fw + pad); y2 = min(ch, fy + fh + pad)
                        crop = current_frame[y1:y2, x1:x2]
                        photo_path = str(FACES_DIR / f"{pid}.jpg")
                        cv2.imwrite(photo_path, crop)
                        msg = "face crop saved"
                    else:
                        # No face detected — save full frame as fallback
                        photo_path = str(FACES_DIR / f"{pid}_full.jpg")
                        cv2.imwrite(photo_path, current_frame)
                        msg = "full frame saved (no face detected)"
                    # Update face memory record
                    if pid in faces:
                        faces[pid]["photo_path"] = photo_path
                        save_face_memory(faces)
                    print(f"  📸  Photo [{msg}]: {photo_path}")
                    result = {"status": "saved", "path": photo_path, "note": msg}
                except Exception as e:
                    result = {"status": "error", "reason": str(e)}

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
            description=(
                "Save a persistent key-value memory. Use this when the user says "
                "'remember [X]', 'don't forget [X]', or gives you a commitment/task. "
                "Also use for any important fact you learn about the user."
            ),
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

        types.FunctionDeclaration(name="save_story",
            description=(
                "Save a story, anecdote, or memorable event told by the user. "
                "Call this IMMEDIATELY whenever a user shares a story or narrative. "
                "Stores it with a timestamped key so it is never forgotten."
            ),
            parameters=S(type=T.OBJECT, properties={
                "topic":   S(type=T.STRING,
                    description="Short slug for the story topic, e.g. 'trip_to_goa'"),
                "content": S(type=T.STRING,
                    description="Full story or event description to remember"),
            }, required=["topic","content"])),

        types.FunctionDeclaration(name="save_person_photo",
            description=(
                "Capture and save a face-crop photo of a person from the current camera frame. "
                "Call this after remember_person() to associate a visual reference. "
                "Requires person_id matching the one used in remember_person()."
            ),
            parameters=S(type=T.OBJECT, properties={
                "person_id": S(type=T.STRING,
                    description="Same person_id used in remember_person()"),
            }, required=["person_id"])),
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
        # temperature=1.2 — higher than default so ADAM doesn't fall into
        # repetitive response patterns; still coherent but more varied/natural.
        generation_config=types.GenerationConfig(temperature=1.2),
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
                        # Tell tracker whether ADAM is currently speaking so it
                        # suppresses speaker-id updates during ADAM's response.
                        tracker.set_adam_speaking(adam_speaking.is_set())
                        tracker_result = await asyncio.to_thread(tracker.process_frame, raw)

                        # ── Build camera context string for injections ─────────
                        ctx_str = tracker.build_context_string(tracker_result)
                        last_camera_context[0] = ctx_str
                        last_tracker_result[0] = tracker_result

                        # ── Attention from face gaze ───────────────────────────
                        # ANY visible face (not just centred) keeps ADAM attentive.
                        has_face = tracker_result["face_count"] > 0
                        if has_face:
                            await attention.activate("face-detected")
                        else:
                            elapsed = time.time() - attention._last_active_time
                            if (attention.state == AttentionState.ATTENTIVE and elapsed > 8.0):
                                await attention.deactivate("no-face-in-frame")

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

                        # ── Push to live preview window ───────────────────────
                        if SHOW_PREVIEW:
                            preview_data = {
                                "frame":           raw,
                                "tracker_result":  tracker_result,
                                "attention_state": attention.state,
                                "adam_speaking":   adam_speaking.is_set(),
                                "gesture":         last_gesture_sent,
                                "gesture_ts":      last_gesture_sent_time,
                            }
                            # Drop the stale frame (if any) then put the fresh one.
                            # With maxsize=1 this guarantees the display thread
                            # always sees the most recent data.
                            try:
                                _preview_queue.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                _preview_queue.put_nowait(preview_data)
                            except queue.Full:
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
                                responses = await handle_tool_call(
                                    msg.tool_call, memory, faces, latest_frame[0])
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
                                # Cleanly reset speaker tracking so the next voice
                                # turn detects the active speaker from scratch.
                                tracker.reset_for_new_turn()
                                last_confirmed_speaker_idx[0] = None
                                print("─" * 40)

                except (ConnectionClosedError, ConnectionClosedOK) as e:
                    code = getattr(e, "code", None)
                    if code == 1004:
                        print(f"\n  ⚠️  Server closed connection (code 1004) — will resume")
                    # Return cleanly so the outer loop can reconnect/resume.
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
                STUCK_WATCHDOG_S = 1.5   # was 2.5 — quicker unstick on error-1004

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
                    tracker.set_adam_speaking(False)   # unfreeze speaker detection
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
            # Flush stale audio from previous session before reconnecting
            while not out_q.empty():
                try:
                    out_q.get_nowait()
                except asyncio.QueueEmpty:
                    break

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

    if SHOW_PREVIEW:
        threading.Thread(target=run_preview_thread, daemon=True, name="preview").start()
        print(f"  📺  Preview → window '{PREVIEW_WIN_NAME}'  (press Q to close)")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋  Goodbye.")
    finally:
        _preview_stop.set()


if __name__ == "__main__":
    main_entry()