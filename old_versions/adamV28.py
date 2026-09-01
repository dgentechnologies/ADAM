"""
ADAM — Autonomous Desktop AI Module (v28)
==========================================
Works with the single system_prompt.txt (compact, token-optimised).
No other prompt files needed.

CHANGES FROM v27:
  • PROPER Google Search grounding — replaces the custom FunctionDeclaration
    workaround.  The native `types.Tool(google_search=types.GoogleSearch())`
    is now passed directly in LiveConnectConfig.tools, exactly as the official
    Gemini Live API documentation specifies.
    - Model calls Google Search autonomously via the built-in grounding pathway;
      no tool_call callback fires for search queries.
    - `executable_code` and `code_execution_result` parts from grounding are
      logged in the receive() task for observability.
    - The custom `google_search` FunctionDeclaration is removed from build_tools().
    - The `google_search` elif branch is removed from handle_tool_call() since
      native grounding never routes through the function-call handler.
  • Banner and SETUP updated for v28.

SETUP (same as v27):
    pip install --upgrade google-genai pyaudio python-dotenv websockets flask
                           opencv-python pyperclip pyserial
    pip install webrtcvad
    pip install vosk
    (Download vosk model: https://alphacephei.com/vosk/models)

RUN:
    python adamV28.py
"""

# ── NOTE: The sections below are the v27 CHANGES docblock preserved for reference:
#
# v27 CHANGES FROM v26 summary (see adamV27.py for full detail):
#   Haar+KCF+gaze+optical-flow+VAD — all phases 1-7 inherited from v27.

import asyncio
import os
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

# ── Neck servo import ─────────────────────────────────────────────────────────
try:
    from adam_neck_serial import (
        init_neck, named_move, pan, tilt,
        emotion_move, reset_neck, close_neck, is_ready as neck_is_ready
    )
    NECK_AVAILABLE = True
except ImportError:
    NECK_AVAILABLE = False
    def init_neck(): return False
    def named_move(m): pass
    def pan(a): pass
    def tilt(a): pass
    def emotion_move(e): pass
    def reset_neck(): pass
    def close_neck(): pass
    def neck_is_ready(): return False

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=".env")
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌  API key not found. Set GOOGLE_API_KEY in your .env file.")
print("✅  API key loaded")

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG  — edit freely
# ═════════════════════════════════════════════════════════════════════════════

LIVE_MODEL          = "gemini-3.1-flash-live-preview"
GEN_MODEL_CASCADE   = ["gemini-3.1-flash-lite-preview", "gemini-3.1-flash-live-preview"]
GEN_RETRIES         = 2

FLASK_PORT          = 5000
WS_HOST             = "localhost"
WS_PORT             = 8765
POST_SPEECH_MUTE_S  = 0.4
VOICE               = "Charon"

CAMERA_INDEX        = 0
FRAME_SIZE          = (768, 768)   # optimal for Gemini Live vision
CAMERA_FPS_INTERVAL = 1.0          # max 1 FPS to Gemini (API hard limit)

SHOW_PREVIEW        = True         # set False for headless / Pi deployment
PREVIEW_WIN_NAME    = "ADAM v27 — Camera Monitor"
PREVIEW_SIZE        = (640, 480)

ENABLE_IDLE         = True
IDLE_TIMEOUT_S      = 90           # seconds passive before idle nudge

ATTENTION_TIMEOUT_S     = 30       # seconds before attention auto-expires
FACE_DETECT_MIN_NEIGHBORS = 4      # Haar cascade minNeighbors

# Physical neck tracking — pan toward speaker if off-centre by this many degrees
NECK_TRACK_DEADZONE = 12           # degrees from centre before moving

# ── v27: Detection + gaze + lip-motion + VAD constants ───────────────────────
DETECT_W                    = 320  # face detection frame width  (Pi-friendly downscale)
DETECT_H                    = 240  # face detection frame height
GAZE_EYE_SYMMETRY_THRESHOLD = 0.55 # eye-strip symmetry score → person facing camera
LIP_MOTION_THRESHOLD        = 1.5  # optical-flow vertical px displacement → speaking
KCF_MAX_MISSES              = 15   # frames before dropping a lost tracker
SPEAKER_INERTIA_FRAMES      = 10   # frames before switching active speaker
VAD_FRAME_MS                = 20   # webrtcvad frame duration (10, 20, or 30 ms)
VAD_WINDOW                  = 10   # rolling vote window (frames)
GAZE_AWAY_DEACTIVATE_S      = 2.0  # seconds after gaze lost before going PASSIVE

WAKE_WORDS      = ["adam", "hey adam", "ok adam", "okay adam"]
VOSK_MODEL_PATH = "vosk-model-small-en-in-0.4"

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE      = Path(BASE_DIR) / "adam_memory.json"
FACE_MEMORY_FILE = Path(BASE_DIR) / "adam_faces.json"
CONV_MEMORY_FILE = Path(BASE_DIR) / "adam_conversations.json"
FACES_DIR        = Path(BASE_DIR) / "faces"

CONV_MAX_TURNS    = 40   # max turns kept on disk
CONV_PROMPT_TURNS = 20   # turns injected into system prompt (token budget)

USE_GROUNDING     = False  # set True to enable native Google Search grounding (billable)

_preview_queue: queue.Queue = queue.Queue(maxsize=1)
_preview_stop               = threading.Event()


# ═════════════════════════════════════════════════════════════════════════════
# KCF TRACKER FACTORY  — wraps around whichever tracker OpenCV has available
# ═════════════════════════════════════════════════════════════════════════════

def _make_kcf_tracker(frame: np.ndarray, bbox: tuple):
    """
    Create the best available OpenCV object tracker and initialise it.

    Tries KCF → CSRT → MOSSE across both the modern and legacy namespaces.
    Falls back to a frozen-position stub if none are available (no contrib
    package required).
    """
    bbox_int = tuple(int(v) for v in bbox)
    for ctor in [
        lambda: cv2.TrackerKCF_create(),
        lambda: cv2.legacy.TrackerKCF_create(),
        lambda: cv2.TrackerCSRT_create(),
        lambda: cv2.legacy.TrackerCSRT_create(),
        lambda: cv2.TrackerMOSSE_create(),
        lambda: cv2.legacy.TrackerMOSSE_create(),
    ]:
        try:
            t = ctor()
            if t.init(frame, bbox_int):
                return t
        except (AttributeError, cv2.error):
            continue

    # Frozen-position fallback — no extra dependency, still prevents blank output
    class _FrozenTracker:
        def __init__(self, b: tuple): self._b = b
        def update(self, f: np.ndarray): return True, self._b  # noqa: ANN001

    return _FrozenTracker(bbox_int)


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-PERSON TRACKER  — v27 hybrid (Haar + KCF + optical-flow gaze)
# ═════════════════════════════════════════════════════════════════════════════

class PersonTracker:
    """
    Multi-person face tracker with:
      - Phase 1: fast low-res Haar detection with CLAHE
      - Phase 2: KCF tracker fallback when cascade misses frames
      - Phase 3: eye-symmetry gaze detection (facing_camera flag per face)
      - Phase 4: optical-flow lip-motion speaker detection
    """

    def __init__(self) -> None:
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade   = cv2.CascadeClassifier(cascade)
        self._available = not self._cascade.empty()
        if self._available:
            print("  👥  Person tracker ready (Haar+KCF+optical-flow, v27)")
        else:
            print("  ⚠️  Face cascade missing — tracker disabled")

        # Phase 1: CLAHE instance (reused across frames — thread-safe when called
        #          via asyncio.to_thread sequentially, which is the case here)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Phase 2: KCF tracker state
        self._kcf_trackers:    list     = []
        self._kcf_bboxes:      list     = []  # last known bbox per tracker
        self._kcf_miss_counts: list[int] = []

        # Phase 4: previous CLAHE gray frame for optical flow
        self._prev_gray: np.ndarray | None = None

        self._active_spk_id: int | None = None
        self._inertia:        int       = 0
        self._adam_speaking:  bool      = False

    @property
    def available(self) -> bool:
        return self._available

    # ── Phase 3: gaze ─────────────────────────────────────────────────────────

    def _eye_symmetry_score(self, gray: np.ndarray,
                            fx: int, fy: int, fw: int, fh: int) -> float:
        """
        Estimate whether the person is facing the camera by measuring bilateral
        brightness symmetry in the upper-40% (eye-region) of the face bbox.

        Returns 0.0 (fully asymmetric / side-on) … 1.0 (perfect symmetry / frontal).
        """
        h, w  = gray.shape
        ey    = max(0, fy)
        eh    = min(int(fh * 0.40), h - ey)
        ex    = max(0, fx)
        ew    = min(fw, w - ex)
        if eh <= 2 or ew <= 4:
            return 0.5  # insufficient data — neutral

        eye_strip = self._clahe.apply(gray[ey:ey + eh, ex:ex + ew].copy())
        _, thresh = cv2.threshold(eye_strip, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        half         = ew // 2
        left_bright  = int(np.count_nonzero(thresh[:, :half]))
        right_bright = int(np.count_nonzero(thresh[:, half:]))
        total        = left_bright + right_bright
        if total == 0:
            return 0.5
        return float(1.0 - abs(left_bright - right_bright) / total)

    # ── Phase 4: lip optical flow ─────────────────────────────────────────────

    def _lip_motion_score(self, prev_gray: np.ndarray, curr_gray: np.ndarray,
                          fx: int, fy: int, fw: int, fh: int) -> float:
        """
        Measure vertical optical-flow displacement at lip-anchor points.

        Uses goodFeaturesToTrack + calcOpticalFlowPyrLK on the lip strip
        (68–88% of face height, 25–75% of face width). Score = mean absolute
        vertical displacement of well-tracked points.

        Robust to lighting changes, bbox jitter, and head shadows.
        """
        h, w  = prev_gray.shape
        ly0   = max(0, min(fy + int(fh * 0.68), h - 2))
        ly1   = max(0, min(fy + int(fh * 0.88), h - 1))
        lx0   = max(0, min(fx + int(fw * 0.25), w - 2))
        lx1   = max(0, min(fx + int(fw * 0.75), w - 1))

        if ly1 - ly0 < 4 or lx1 - lx0 < 4:
            return 0.0

        lip_prev = prev_gray[ly0:ly1, lx0:lx1]
        if lip_prev.size == 0:
            return 0.0

        pts = cv2.goodFeaturesToTrack(
            lip_prev, maxCorners=6, qualityLevel=0.15, minDistance=3)
        if pts is None or len(pts) == 0:
            return 0.0

        # Translate ROI-local points back to full-frame coordinates
        pts_full = pts + np.float32([[lx0, ly0]])

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts_full, None,
            winSize=(7, 7), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        if next_pts is None or status is None:
            return 0.0

        vertical_disps: list[float] = []
        for i, s in enumerate(status):
            if s[0]:
                dy = abs(float(next_pts[i][0][1]) - float(pts_full[i][0][1]))
                vertical_disps.append(dy)

        return float(np.mean(vertical_disps)) if vertical_disps else 0.0

    # ── Main processing ───────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray,
                      vad_active: bool = True) -> dict:
        """
        Process one camera frame.  Returns a dict with:
          faces              — list of face dicts (id, x, y, w, h, cx_norm,
                               cy_norm, facing_camera, facing_ratio,
                               eye_symmetry, lip_score)
          face_count         — len(faces)
          active_speaker_idx — index of face with most confirmed lip motion, or None
        """
        result: dict = {
            "faces": [], "active_speaker_idx": None, "face_count": 0}
        if not self._available:
            return result

        h_f, w_f = frame.shape[:2]

        # ── Full-res CLAHE gray — used for gaze, optical flow, and KCF ───────
        gray_full = self._clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # ── Phase 1: Detect at downscaled resolution ─────────────────────────
        scale_x = w_f / DETECT_W
        scale_y = h_f / DETECT_H
        small      = cv2.resize(frame, (DETECT_W, DETECT_H))
        gray_small = self._clahe.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

        raw_det = self._cascade.detectMultiScale(
            gray_small,
            scaleFactor=1.2,
            minNeighbors=FACE_DETECT_MIN_NEIGHBORS,
            minSize=(20, 20),
        )

        if len(raw_det) > 0:
            # Scale detected bboxes back to full-frame coordinates
            raw_faces: list[tuple] = [
                (int(x * scale_x), int(y * scale_y),
                 int(w * scale_x), int(h * scale_y))
                for (x, y, w, h) in raw_det
            ]

            # ── Phase 2: Reinitialise KCF trackers from fresh cascade detections
            if len(raw_faces) != len(self._kcf_trackers):
                # Face count changed — rebuild all trackers
                self._kcf_trackers    = []
                self._kcf_bboxes      = []
                self._kcf_miss_counts = []
                for bbox in raw_faces:
                    self._kcf_trackers.append(_make_kcf_tracker(frame, bbox))
                    self._kcf_bboxes.append(bbox)
                    self._kcf_miss_counts.append(0)
            else:
                # Same count — reinitialise each tracker to new accurate position
                for i, bbox in enumerate(raw_faces):
                    self._kcf_trackers[i]    = _make_kcf_tracker(frame, bbox)
                    self._kcf_bboxes[i]      = bbox
                    self._kcf_miss_counts[i] = 0

            faces = sorted(raw_faces, key=lambda f: f[0])

        else:
            # ── Phase 2: No cascade detection — update from KCF trackers ─────
            if not self._kcf_trackers:
                self._prev_gray     = gray_full
                self._active_spk_id = None
                self._inertia       = 0
                return result

            alive_trackers:    list     = []
            alive_bboxes:      list     = []
            alive_miss_counts: list[int] = []
            predicted_faces:   list     = []

            for i, tracker in enumerate(self._kcf_trackers):
                ok      = False
                new_box = self._kcf_bboxes[i]
                try:
                    ok, new_box = tracker.update(frame)
                except cv2.error:
                    ok = False

                miss = self._kcf_miss_counts[i] + (0 if ok else 1)
                if miss >= KCF_MAX_MISSES:
                    continue  # tracker has drifted too far — drop it

                if ok:
                    x, y, tw, th = (int(v) for v in new_box)
                    # Clamp to frame boundaries
                    x  = max(0, min(x, w_f - 1))
                    y  = max(0, min(y, h_f - 1))
                    tw = max(1, min(tw, w_f - x))
                    th = max(1, min(th, h_f - y))
                    new_box = (x, y, tw, th)
                else:
                    new_box = self._kcf_bboxes[i]  # freeze at last known position

                alive_trackers.append(tracker)
                alive_bboxes.append(new_box)
                alive_miss_counts.append(miss)
                predicted_faces.append(new_box)

            self._kcf_trackers    = alive_trackers
            self._kcf_bboxes      = alive_bboxes
            self._kcf_miss_counts = alive_miss_counts

            if not predicted_faces:
                self._prev_gray     = gray_full
                self._active_spk_id = None
                self._inertia       = 0
                return result

            faces = sorted(predicted_faces, key=lambda f: f[0])

        result["face_count"] = len(faces)

        # ── Phase 3 & 4: Per-face gaze + lip-motion ──────────────────────────
        lip_scores: list[float] = []

        for idx, (fx, fy, fw, fh) in enumerate(faces):
            cx           = (fx + fw / 2) / w_f
            cy           = (fy + fh / 2) / h_f
            facing_ratio = fw / max(fh, 1)

            # Phase 3: gaze
            eye_sym      = self._eye_symmetry_score(gray_full, fx, fy, fw, fh)
            facing_camera = (eye_sym        > GAZE_EYE_SYMMETRY_THRESHOLD
                             and facing_ratio > 0.65)

            # Phase 4: lip motion (needs at least 2 frames)
            lip_score = 0.0
            if self._prev_gray is not None:
                lip_score = self._lip_motion_score(
                    self._prev_gray, gray_full, fx, fy, fw, fh)
            lip_scores.append(lip_score)

            result["faces"].append({
                "id":            idx,
                "x":             int(fx),
                "y":             int(fy),
                "w":             int(fw),
                "h":             int(fh),
                "cx_norm":       cx,
                "cy_norm":       cy,
                "facing_camera": facing_camera,
                "facing_ratio":  facing_ratio,
                "eye_symmetry":  eye_sym,
                "lip_score":     lip_score,
            })

        # ── Phase 4 + 5: Speaker detection — optical flow AND VAD gate ───────
        if not self._adam_speaking and lip_scores:
            max_score = max(lip_scores)
            # Require optical-flow threshold AND VAD confirmation
            if max_score >= LIP_MOTION_THRESHOLD and vad_active:
                cand = lip_scores.index(max_score)
                if cand == self._active_spk_id:
                    self._inertia = SPEAKER_INERTIA_FRAMES
                else:
                    self._inertia -= 1
                    if self._inertia <= 0:
                        self._active_spk_id = cand
                        self._inertia       = SPEAKER_INERTIA_FRAMES
            else:
                self._inertia = max(0, self._inertia - 1)
                if self._inertia == 0:
                    self._active_spk_id = None

        result["active_speaker_idx"] = self._active_spk_id

        # Store gray for next frame's optical flow
        self._prev_gray = gray_full
        return result

    # ── Context string for Gemini ─────────────────────────────────────────────

    def build_context(self, tr: dict) -> str:
        """
        Build a terse camera-context string describing face positions, gaze
        directions, and the active speaker.  Injected into the Gemini session
        alongside audio so the model knows who is talking to whom.
        """
        faces = tr["faces"]
        count = tr["face_count"]
        spk   = tr["active_speaker_idx"]

        if count == 0:
            return "[CAMERA: No faces in frame.]"

        if count == 1:
            f    = faces[0]
            cx   = f["cx_norm"]
            pos  = "centre" if 0.35 < cx < 0.65 else ("left" if cx < 0.5 else "right")
            gaze = "→CAM" if f["facing_camera"] else "→AWAY"
            spk_tag = ",SPEAKING" if spk == 0 else ""
            return f"[CAMERA: 1 person ({pos},{gaze}{spk_tag}).]"

        parts: list[str] = []
        for i, f in enumerate(faces):
            cx  = f["cx_norm"]
            pos = ("left" if cx < 0.40 else "right" if cx > 0.60 else "centre")

            if f["facing_camera"]:
                direction = "→CAM"
            else:
                # Head turned: infer direction from screen position
                direction = "→RIGHT" if cx < 0.50 else "→LEFT"

            spk_tag = ",SPEAKING" if i == spk else ""
            parts.append(f"P{i+1}({pos},{direction}{spk_tag})")

        ctx = f"[CAMERA: {count} people — {', '.join(parts)}."
        if spk is not None and spk < len(faces):
            facing_cam = faces[spk]["facing_camera"]
            addr       = "talking to ADAM" if facing_cam else "talking to another person"
            ctx       += f" P{spk+1} is speaking, likely {addr}."
        return ctx + "]"

    # ── State management ──────────────────────────────────────────────────────

    def set_adam_speaking(self, flag: bool) -> None:
        if flag == self._adam_speaking:
            return
        self._adam_speaking = flag
        if not flag:
            self._inertia       = 0
            self._active_spk_id = None
            self._prev_gray     = None   # discard stale flow baseline

    def reset_for_new_turn(self) -> None:
        self._inertia       = 0
        self._active_spk_id = None
        self._prev_gray     = None


# ═════════════════════════════════════════════════════════════════════════════
# VOICE ACTIVITY DETECTOR  — Phase 5
# ═════════════════════════════════════════════════════════════════════════════

class VoiceActivityDetector:
    """
    Wraps webrtcvad.Vad to provide a simple is_active() signal from raw PCM.

    Feeds 16 kHz int16 mono chunks, internally frames them into VAD_FRAME_MS
    windows, and maintains a rolling majority vote over VAD_WINDOW frames.

    Falls back gracefully to always-True (permissive) when webrtcvad is not
    installed — the system still works, just without the second confirmation.
    """

    def __init__(self, mode: int = 2) -> None:
        self._ready        = False
        self._vad          = None
        self._buf          = b""
        self._votes: deque = deque(maxlen=VAD_WINDOW)
        # VAD_FRAME_MS ms at 16 kHz → samples → bytes (int16 = 2 bytes)
        self._frame_bytes  = int(SEND_SAMPLE_RATE * (VAD_FRAME_MS / 1000.0)) * 2

        if WEBRTCVAD_AVAILABLE:
            try:
                self._vad   = _webrtcvad.Vad(mode)
                self._ready = True
                print(f"  🎤  webrtcvad ready (mode {mode}, {VAD_FRAME_MS}ms frames)")
            except Exception as e:
                print(f"  ⚠️  webrtcvad init failed: {e} — permissive fallback")
        else:
            print("  ⚠️  webrtcvad not installed — VAD disabled (permissive fallback)")

    def feed(self, pcm_chunk: bytes) -> None:
        """Feed a raw PCM int16 chunk (any size); processed internally in frames."""
        if not self._ready:
            return
        self._buf += pcm_chunk
        while len(self._buf) >= self._frame_bytes:
            frame      = self._buf[: self._frame_bytes]
            self._buf  = self._buf[self._frame_bytes :]
            try:
                is_speech = self._vad.is_speech(frame, SEND_SAMPLE_RATE)
                self._votes.append(1 if is_speech else 0)
            except Exception:
                pass

    def is_active(self) -> bool:
        """True if the majority of the rolling window detected voice (or if VAD unavailable)."""
        if not self._ready or not self._votes:
            return True  # permissive fallback
        return sum(self._votes) > len(self._votes) // 2


# ═════════════════════════════════════════════════════════════════════════════
# ATTENTION MANAGER
# ═════════════════════════════════════════════════════════════════════════════

class AttentionState:
    PASSIVE    = "passive"
    ATTENTIVE  = "attentive"
    RESPONDING = "responding"


class AttentionManager:
    def __init__(self) -> None:
        self._state     = AttentionState.PASSIVE
        self._last_t    = 0.0
        self._lock      = asyncio.Lock()
        self._on_change = None

    def set_callback(self, cb) -> None:
        self._on_change = cb

    @property
    def state(self) -> str:
        return self._state

    @property
    def last_active_time(self) -> float:
        return self._last_t

    def is_active(self) -> bool:
        if self._state == AttentionState.ATTENTIVE:
            if time.time() - self._last_t > ATTENTION_TIMEOUT_S:
                self._state = AttentionState.PASSIVE
                return False
            return True
        return False

    async def activate(self, reason: str = "") -> None:
        async with self._lock:
            if self._state == AttentionState.RESPONDING:
                return
            old          = self._state
            self._state  = AttentionState.ATTENTIVE
            self._last_t = time.time()
            if old != AttentionState.ATTENTIVE:
                print(f"  👁️  ATTENTIVE [{reason}]")
                if self._on_change:
                    await self._on_change(AttentionState.ATTENTIVE)

    async def deactivate(self, reason: str = "") -> None:
        async with self._lock:
            if self._state == AttentionState.ATTENTIVE:
                self._state = AttentionState.PASSIVE
                print(f"  😶  PASSIVE [{reason}]")
                if self._on_change:
                    await self._on_change(AttentionState.PASSIVE)

    async def set_responding(self, on: bool) -> None:
        async with self._lock:
            self._state = AttentionState.RESPONDING if on else AttentionState.ATTENTIVE
            if not on:
                self._last_t = time.time()

    def touch(self) -> None:
        if self._state in (AttentionState.ATTENTIVE, AttentionState.RESPONDING):
            self._last_t = time.time()


# ═════════════════════════════════════════════════════════════════════════════
# WAKE WORD DETECTOR
# ═════════════════════════════════════════════════════════════════════════════

class WakeWordDetector:
    def __init__(self) -> None:
        self._ready    = False
        self._rec      = None
        self._aq: queue.Queue = queue.Queue()
        self._callback = None

        if VOSK_AVAILABLE:
            mp = Path(BASE_DIR) / VOSK_MODEL_PATH
            if mp.exists():
                try:
                    self._rec   = KaldiRecognizer(VoskModel(str(mp)), 16000)
                    self._ready = True
                    print(f"  🎙️  Vosk wake-word ready ({VOSK_MODEL_PATH})")
                except Exception as e:
                    print(f"  ⚠️  Vosk init failed: {e}")
            else:
                print("  ⚠️  Vosk model folder not found — using transcript fallback")
        else:
            print("  ⚠️  Vosk not installed — using transcript fallback")

    def set_callback(self, cb) -> None:
        self._callback = cb

    def feed_audio(self, data: bytes) -> None:
        if self._ready:
            try:
                self._aq.put_nowait(data)
            except queue.Full:
                pass

    def is_wake_word(self, text: str) -> bool:
        t = text.lower().strip()
        return any(ww in t for ww in WAKE_WORDS)

    def run_vosk_thread(self) -> None:
        if not self._ready:
            return
        print("  🎙️  Vosk thread running")
        while True:
            try:
                chunk = self._aq.get(timeout=1.0)
            except queue.Empty:
                continue
            if self._rec.AcceptWaveform(chunk):
                text = json.loads(self._rec.Result()).get("text", "")
            else:
                text = json.loads(self._rec.PartialResult()).get("partial", "")
            if text and self.is_wake_word(text):
                print(f"  🔔  Wake word detected: '{text}'")
                if self._callback:
                    self._callback()


# ═════════════════════════════════════════════════════════════════════════════
# PERSISTENT MEMORY
# ═════════════════════════════════════════════════════════════════════════════

def load_memory() -> dict:
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            print(f"  🧠  Memory loaded: {len(d)} entries")
            return d
        except Exception as e:
            print(f"  ⚠️  Memory load error: {e}")
    return {}


def save_memory(memory: dict) -> None:
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def load_face_memory() -> dict:
    if FACE_MEMORY_FILE.exists():
        try:
            with open(FACE_MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"  ⚠️  Face memory load error: {e}")
    return {}


def save_face_memory(faces: dict) -> None:
    with open(FACE_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(faces, f, ensure_ascii=False, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
# CONVERSATION LOG  — persistent turn-by-turn history across sessions
# ═════════════════════════════════════════════════════════════════════════════

def load_conversation_log() -> list:
    """Load the persisted conversation history from disk."""
    if CONV_MEMORY_FILE.exists():
        try:
            with open(CONV_MEMORY_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            print(f"  💬  Conversation log loaded: {len(d)} turns")
            return d
        except Exception as e:
            print(f"  ⚠️  Conversation log load error: {e}")
    return []


def save_conversation_log(log: list) -> None:
    """Persist the conversation log to disk, capping at CONV_MAX_TURNS."""
    if len(log) > CONV_MAX_TURNS:
        del log[:-CONV_MAX_TURNS]
    with open(CONV_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def append_conversation_turn(log: list, user_text: str, adam_text: str) -> None:
    """Append one completed exchange to the log and save it."""
    u = user_text.strip()
    a = adam_text.strip()
    if not u and not a:
        return
    log.append({
        "ts":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "user": u,
        "adam": a,
    })
    save_conversation_log(log)
    print(f"  💬  Turn saved to conversation log ({len(log)} total)")


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def _build_memory_block(memory: dict) -> str:
    if not memory:
        return ""
    lines = ["━━━ YOUR PERSISTENT MEMORY ━━━"]
    for k, v in memory.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _build_faces_block(faces: dict) -> str:
    if not faces:
        return ""
    lines = ["━━━ PEOPLE YOU KNOW ━━━"]
    for pid, info in faces.items():
        photo = f" [photo: {info['photo_path']}]" if info.get("photo_path") else ""
        lines.append(
            f"  [{pid}]{photo} {info.get('name','?')} | "
            f"appearance: {info.get('appearance','?')} | "
            f"voice: {info.get('voice_cues','?')} | "
            f"rel: {info.get('relationship','?')} | "
            f"notes: {info.get('notes','')}"
        )
    return "\n".join(lines)


def _build_conversation_block(log: list) -> str:
    """Build the recent conversation history block for the system prompt."""
    if not log:
        return ""
    recent = log[-CONV_PROMPT_TURNS:]
    lines  = ["━━━ RECENT CONVERSATION HISTORY ━━━"]
    for turn in recent:
        ts = turn.get("ts", "")
        u  = turn.get("user", "").strip()
        a  = turn.get("adam", "").strip()
        if u:
            lines.append(f"  [{ts}] You: {u}")
        if a:
            lines.append(f"  [{ts}] ADAM: {a}")
    return "\n".join(lines)


def load_system_prompt(memory: dict, faces: dict,
                       conv_log: list | None = None) -> str:
    prompt_path = Path(BASE_DIR) / "system_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(
            "system_prompt.txt not found. Place it in the same folder as this script."
        )
    base = prompt_path.read_text(encoding="utf-8").strip()
    if base.startswith('"""') and base.endswith('"""'):
        base = base[3:-3].strip()

    parts = [p for p in [
        _build_memory_block(memory),
        _build_faces_block(faces),
        _build_conversation_block(conv_log or []),
        base,
    ] if p.strip()]
    return "\n\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# GEN CLIENT — clipboard generation
# ═════════════════════════════════════════════════════════════════════════════

_gen_client: genai.Client | None = None


def init_gen_client() -> None:
    global _gen_client
    try:
        _gen_client = genai.Client(api_key=API_KEY)
        print(f"  ⚡  Gen client ready  (cascade: {' → '.join(GEN_MODEL_CASCADE)})")
    except Exception as e:
        print(f"  ⚠️  Gen client init failed: {e}")


_GEN_SYS = (
    "Output ONLY the requested content. "
    "No preamble, no explanation, no markdown fences unless explicitly requested. "
    "No 'Here is...' prefix. Just the content itself."
)


async def generate_to_clipboard(prompt: str) -> str:
    if not CLIPBOARD_AVAILABLE:
        return "Clipboard unavailable. Install pyperclip."
    if _gen_client is None:
        return "Gen client not initialised."

    for model in GEN_MODEL_CASCADE:
        for attempt in range(1, GEN_RETRIES + 1):
            try:
                resp = await asyncio.to_thread(
                    lambda m=model: _gen_client.models.generate_content(
                        model=m, contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=_GEN_SYS, temperature=0.3)
                    )
                )
                text = (resp.text or "").strip()
                if text:
                    await asyncio.to_thread(pyperclip.copy, text)
                    lines = text.count("\n") + 1
                    print(f"  📋  Clipboard: {len(text)} chars / {lines} lines [{model}]")
                    return random.choice([
                        "Done. Paste it.", "Clipboard loaded. Ctrl+V.",
                        "Ready when you are.", "Generated. Go ahead.",
                        "It's in your clipboard.",
                    ])
            except Exception as e:
                err = str(e)
                if any(x in err for x in
                       ["503", "429", "quota", "overloaded", "UNAVAILABLE"]):
                    if attempt < GEN_RETRIES:
                        await asyncio.sleep(1.5 * attempt)
                else:
                    break
        print(f"  🔄  Gen cascade: falling back from {model}")
    return "All generation models are busy right now. Try again in a moment."


# ═════════════════════════════════════════════════════════════════════════════
# AUDIO CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

FORMAT           = pyaudio.paInt16
CHANNELS         = 1
SEND_SAMPLE_RATE = 16000
RECV_SAMPLE_RATE = 24000
CHUNK_SIZE       = 512

pya = pyaudio.PyAudio()


# ═════════════════════════════════════════════════════════════════════════════
# FLASK  — serves adam_face.html
# ═════════════════════════════════════════════════════════════════════════════

flask_app = Flask(__name__, static_folder=BASE_DIR)


@flask_app.route("/")
def index():
    return send_from_directory(BASE_DIR, "adam_face.html")


def run_flask() -> None:
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    flask_app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)


# ═════════════════════════════════════════════════════════════════════════════
# WEBSOCKET  — drives the OLED face
# ═════════════════════════════════════════════════════════════════════════════

ws_clients: set = set()


async def ws_broadcast(payload: dict) -> None:
    if not ws_clients:
        return
    msg  = json.dumps(payload)
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)


async def ws_handler(websocket) -> None:
    ws_clients.add(websocket)
    print(f"  🌐  Browser connected ({len(ws_clients)} client"
          f"{'s' if len(ws_clients) > 1 else ''})")
    try:
        await websocket.wait_closed()
    finally:
        ws_clients.discard(websocket)


# Emotion → WebSocket head-movement mapping (for OLED face animation)
EMOTION_HEAD = {
    "happy":     "nod_yes",
    "excited":   "nod_fast",
    "angry":     "none",
    "confused":  "none",
    "smug":      "none",
    "sad":       "none",
    "surprised": "nod_yes",
    "thinking":  "none",
    "love":      "nod_yes",
    "blush":     "none",
}

# ── Mouth sync via RMS of outgoing audio ─────────────────────────────────────
_last_sync_t   = 0.0
_SYNC_INTERVAL = 0.06


async def maybe_sync_mouth(audio_chunk: bytes,
                           adam_speaking_event: asyncio.Event) -> None:
    global _last_sync_t
    if not adam_speaking_event.is_set():
        return
    now = time.time()
    if now - _last_sync_t < _SYNC_INTERVAL:
        return
    _last_sync_t = now
    try:
        n = len(audio_chunk) // 2
        if n == 0:
            return
        samples   = struct.unpack(f"{n}h", audio_chunk)
        rms       = (sum(s * s for s in samples) / n) ** 0.5
    except Exception:
        return
    intensity = "low" if rms < 4000 else ("medium" if rms < 10000 else "high")
    await ws_broadcast({"type": "mouth_sync", "intensity": intensity})


# ═════════════════════════════════════════════════════════════════════════════
# CAMERA PREVIEW WINDOW  — updated for v27 (gaze indicators)
# ═════════════════════════════════════════════════════════════════════════════

def _draw_preview(data: dict) -> np.ndarray:
    raw    = data["frame"]
    tr     = data["tracker"]
    state  = data["state"]
    adam_s = data["adam_speaking"]

    pw, ph = PREVIEW_SIZE
    frame  = cv2.resize(raw, (pw, ph))
    rh, rw = raw.shape[:2]
    sx, sy = pw / rw, ph / rh
    spk    = tr.get("active_speaker_idx")

    for i, f in enumerate(tr.get("faces", [])):
        is_spk  = (i == spk)
        is_gaze = f.get("facing_camera", False)

        fx = int(f["x"] * sx)
        fy = int(f["y"] * sy)
        fw = int(f["w"] * sx)
        fh = int(f["h"] * sy)

        # Box colour: green = speaking+facing, cyan = facing cam, grey = away
        if is_spk and is_gaze:
            col   = (0, 230, 80)      # green  — speaking to Adam
        elif is_gaze:
            col   = (0, 200, 255)     # cyan   — facing cam, not speaking
        else:
            col   = (110, 110, 110)   # grey   — away / side conversation
        thick = 2 if (is_spk or is_gaze) else 1
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), col, thick)

        # Eye-strip indicator (top 38% of face box)
        eye_h = max(1, int(fh * 0.38))
        ec    = (0, 200, 255) if is_gaze else (60, 60, 60)
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + eye_h), ec, 1)

        # Label
        gaze_lbl = "→CAM" if is_gaze else "→AWAY"
        spk_lbl  = " ◉SPK" if is_spk else ""
        lbl      = f"P{i+1} {gaze_lbl}{spk_lbl}"
        lbl_y    = fy - 6 if fy > 18 else fy + fh + 16
        cv2.putText(frame, lbl, (fx + 3, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)

    # Status bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, ph - 34), (pw, ph), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

    if adam_s:
        stxt, scol = "ADAM SPEAKING",  (0, 220, 255)
    elif state == AttentionState.RESPONDING:
        stxt, scol = "RESPONDING",     (0, 180, 255)
    elif state == AttentionState.ATTENTIVE:
        stxt, scol = "ATTENTIVE",      (0, 230, 80)
    else:
        stxt, scol = "PASSIVE",        (110, 110, 110)

    cv2.putText(frame, stxt, (10, ph - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, scol, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Faces: {tr.get('face_count', 0)}", (pw - 110, ph - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA)
    neck_txt = "NECK:ON" if neck_is_ready() else "NECK:OFF"
    cv2.putText(frame, neck_txt, (pw - 90, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "ADAM v28  [Q=close]", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1, cv2.LINE_AA)
    return frame


def run_preview_thread() -> None:
    if not SHOW_PREVIEW:
        return
    window_open = False
    while not _preview_stop.is_set():
        if window_open:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                _preview_stop.set()
                break

        try:
            data = _preview_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        annotated = _draw_preview(data)
        if not window_open:
            cv2.namedWindow(PREVIEW_WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(PREVIEW_WIN_NAME, *PREVIEW_SIZE)
            window_open = True
        cv2.imshow(PREVIEW_WIN_NAME, annotated)

    if window_open:
        try:
            cv2.destroyWindow(PREVIEW_WIN_NAME)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# FRAME HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def capture_raw_frame(cap) -> np.ndarray | None:
    ret, frame = cap.read()
    return frame if ret else None


def frame_to_jpeg(frame: np.ndarray, size: tuple = FRAME_SIZE) -> bytes:
    resized = cv2.resize(frame, size)
    _, buf  = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


# ═════════════════════════════════════════════════════════════════════════════
# TOOL HANDLER
# ═════════════════════════════════════════════════════════════════════════════

async def handle_tool_call(tc, memory: dict, faces: dict,
                           current_frame=None) -> list[dict]:
    responses = []
    for fc in tc.function_calls:
        name    = fc.name
        call_id = fc.id
        args    = dict(fc.args) if fc.args else {}

        # ── Date / time ───────────────────────────────────────────────────────
        if name == "get_current_datetime":
            now    = datetime.datetime.now()
            result = {
                "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "date":     now.strftime("%A, %d %B %Y"),
                "time":     now.strftime("%I:%M %p"),
                "timezone": str(now.astimezone().tzname()),
            }
            print(f"  🕐  datetime → {result['datetime']}")

        # ── Clipboard generation ──────────────────────────────────────────────
        elif name == "generate_to_clipboard":
            prompt = args.get("prompt", "").strip()
            result = ({"error": "prompt is empty"} if not prompt else
                      {"status": "done",
                       "confirmation": await generate_to_clipboard(prompt)})

        # ── Face memory: save ─────────────────────────────────────────────────
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
            print(f"  👤  Remembered: {args.get('name')} [{pid}]")
            result = {"status": "saved", "person_id": pid}

        # ── Face memory: update last seen ─────────────────────────────────────
        elif name == "update_person_seen":
            pid = args.get("person_id", "")
            if pid in faces:
                faces[pid]["last_seen"] = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M")
                if args.get("notes_update"):
                    ex = faces[pid].get("notes", "")
                    faces[pid]["notes"] = (ex + " | " + args["notes_update"]).strip(" |")
                save_face_memory(faces)
                result = {"status": "updated"}
            else:
                result = {"status": "not_found", "person_id": pid}

        # ── Face memory: get all ──────────────────────────────────────────────
        elif name == "get_all_people":
            result = {"people": faces}

        # ── Face photo ────────────────────────────────────────────────────────
        elif name == "save_person_photo":
            pid = args.get("person_id", "").strip()
            if not pid:
                result = {"status": "error", "reason": "person_id required"}
            elif current_frame is None:
                result = {"status": "error", "reason": "no camera frame available"}
            else:
                try:
                    FACES_DIR.mkdir(exist_ok=True)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    g     = clahe.apply(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY))
                    cas   = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                    det   = cas.detectMultiScale(
                        g, scaleFactor=1.2, minNeighbors=FACE_DETECT_MIN_NEIGHBORS,
                        minSize=(40, 40))
                    if len(det) > 0:
                        fx, fy, fw, fh = max(det, key=lambda r: r[2] * r[3])
                        pad  = int(fw * 0.30)
                        ch, cw = current_frame.shape[:2]
                        crop = current_frame[
                            max(0, fy - pad): min(ch, fy + fh + pad),
                            max(0, fx - pad): min(cw, fx + fw + pad),
                        ]
                        path = str(FACES_DIR / f"{pid}.jpg")
                        cv2.imwrite(path, crop)
                        note = "face crop saved"
                    else:
                        path = str(FACES_DIR / f"{pid}_full.jpg")
                        cv2.imwrite(path, current_frame)
                        note = "full frame saved (no face detected)"
                    if pid in faces:
                        faces[pid]["photo_path"] = path
                        save_face_memory(faces)
                    print(f"  📸  Photo: {path} ({note})")
                    result = {"status": "saved", "path": path, "note": note}
                except Exception as e:
                    result = {"status": "error", "reason": str(e)}

        # ── Emotion — drives OLED face AND physical neck ──────────────────────
        elif name == "set_emotion":
            emotion = args.get("emotion", "happy")
            await ws_broadcast({"type": "emotion", "emotion": emotion,
                                "head": EMOTION_HEAD.get(emotion, "none")})
            await asyncio.to_thread(emotion_move, emotion)
            result = {"status": "ok"}

        # ── Mouth sync ────────────────────────────────────────────────────────
        elif name == "set_mouth_sync":
            await ws_broadcast({"type": "mouth_sync",
                                "intensity": args.get("intensity", "medium")})
            result = {"status": "ok"}

        # ── Physical neck movement ────────────────────────────────────────────
        elif name == "move_neck":
            if not neck_is_ready():
                result = {"status": "servo_not_connected"}
                print("  ⚠️  move_neck called but servo not connected")
            else:
                pan_a  = args.get("pan_angle")
                tilt_a = args.get("tilt_angle")
                move   = args.get("movement", "").upper().strip()

                if pan_a is not None:
                    await asyncio.to_thread(pan, int(pan_a))
                    result = {"status": "ok", "pan": pan_a}
                elif tilt_a is not None:
                    await asyncio.to_thread(tilt, int(tilt_a))
                    result = {"status": "ok", "tilt": tilt_a}
                elif move:
                    await asyncio.to_thread(named_move, move)
                    result = {"status": "ok", "move": move}
                else:
                    result = {"status": "error", "reason": "no valid argument provided"}

        # ── Key-value memory ──────────────────────────────────────────────────
        elif name == "save_memory":
            key = args.get("key", "").strip()
            val = args.get("value", "").strip()
            if key:
                memory[key] = val
                save_memory(memory)
                print(f"  🧠  Memory saved: {key}")
                result = {"status": "saved", "key": key}
            else:
                result = {"status": "error", "reason": "key cannot be empty"}

        elif name == "delete_memory":
            key = args.get("key", "").strip()
            if key in memory:
                del memory[key]
                save_memory(memory)
                print(f"  🧠  Memory deleted: {key}")
                result = {"status": "deleted", "key": key}
            else:
                result = {"status": "not_found", "key": key}

        elif name == "get_memory":
            key    = args.get("key", "").strip()
            result = {"value": memory.get(key) if key else None, "all": memory}

        # ── Story / event memory ──────────────────────────────────────────────
        elif name == "save_story":
            topic   = args.get("topic", "untitled").strip().replace(" ", "_")
            content = args.get("content", "").strip()
            if content:
                key = (f"story_{topic}_"
                       f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}")
                memory[key] = content
                save_memory(memory)
                print(f"  📖  Story saved: '{key}' ({len(content)} chars)")
                result = {"status": "saved", "key": key}
            else:
                result = {"status": "error", "reason": "content cannot be empty"}

        else:
            print(f"  ⚠️  Unknown tool call: {name}")
            result = {"error": f"Unknown tool: {name}"}

        responses.append({"id": call_id, "name": name, "response": result})
    return responses


# ═════════════════════════════════════════════════════════════════════════════
# TOOL DECLARATIONS
# ═════════════════════════════════════════════════════════════════════════════

def build_tools() -> list[types.Tool]:
    S, T = types.Schema, types.Type

    fn_tool = types.Tool(function_declarations=[

        types.FunctionDeclaration(
            name="get_current_datetime",
            description="Returns the current local date and time.",
            parameters=S(type=T.OBJECT, properties={})),

        types.FunctionDeclaration(
            name="generate_to_clipboard",
            description=(
                "Generate text, code, scripts, emails, or any long-form content using a fast "
                "secondary model, then copy it to the user's clipboard for Ctrl+V pasting. "
                "Use whenever the user asks to write, draft, or generate any content."
            ),
            parameters=S(type=T.OBJECT, properties={
                "prompt":    S(type=T.STRING,
                               description="Full detailed generation prompt with all context."),
                "task_type": S(type=T.STRING,
                               enum=["code", "email", "essay", "template",
                                     "script", "general"]),
            }, required=["prompt"])),

        types.FunctionDeclaration(
            name="remember_person",
            description=(
                "Save a person to permanent visual memory with appearance and voice details. "
                "Call after confirming someone's identity so ADAM recognises them next session."
            ),
            parameters=S(type=T.OBJECT, properties={
                "person_id":    S(type=T.STRING,
                                  description="Short unique slug, e.g. 'tirthankar'"),
                "name":         S(type=T.STRING),
                "appearance":   S(type=T.STRING,
                                  description=(
                                      "Hair, skin tone, build, clothing, "
                                      "distinguishing features")),
                "voice_cues":   S(type=T.STRING,
                                  description="Accent, pace, language style"),
                "relationship": S(type=T.STRING,
                                  description=(
                                      "e.g. creator, owner, colleague, "
                                      "friend, visitor")),
                "notes":        S(type=T.STRING),
            }, required=["person_id", "name"])),

        types.FunctionDeclaration(
            name="update_person_seen",
            description=(
                "Update last-seen timestamp for a known person and optionally append notes."),
            parameters=S(type=T.OBJECT, properties={
                "person_id":    S(type=T.STRING),
                "notes_update": S(type=T.STRING),
            }, required=["person_id"])),

        types.FunctionDeclaration(
            name="get_all_people",
            description="Return everyone stored in visual memory.",
            parameters=S(type=T.OBJECT, properties={})),

        types.FunctionDeclaration(
            name="save_person_photo",
            description=(
                "Capture and save a face-crop photo of a person from the current camera "
                "frame. Call after remember_person() to store a visual reference."
            ),
            parameters=S(type=T.OBJECT, properties={
                "person_id": S(type=T.STRING,
                               description="Same person_id used in remember_person()"),
            }, required=["person_id"])),

        types.FunctionDeclaration(
            name="set_emotion",
            description=(
                "Display an emotion on ADAM's OLED face AND trigger matching physical neck "
                "movement. Call frequently to mirror the user's emotional state or express "
                "ADAM's own reaction."
            ),
            parameters=S(type=T.OBJECT, properties={
                "emotion": S(type=T.STRING,
                             enum=["happy", "excited", "angry", "confused", "smug",
                                   "sad", "surprised", "thinking", "love", "blush"])
            }, required=["emotion"])),

        types.FunctionDeclaration(
            name="set_mouth_sync",
            description="Sync the mouth animation to speech intensity.",
            parameters=S(type=T.OBJECT, properties={
                "intensity": S(type=T.STRING,
                               enum=["closed", "low", "medium", "high"])
            }, required=["intensity"])),

        types.FunctionDeclaration(
            name="move_neck",
            description=(
                "Move ADAM's physical servo neck for emphasis, curiosity, greeting, or "
                "to look toward a person in frame. Call alongside set_emotion() for full "
                "physical expression. Do NOT call on every response — use for emphasis."
            ),
            parameters=S(type=T.OBJECT, properties={
                "movement": S(
                    type=T.STRING,
                    enum=["NOD", "SHAKE", "RESET", "LOOK_UP", "LOOK_DOWN",
                          "LOOK_LEFT", "LOOK_RIGHT", "TILT_CURIOUS"],
                    description=(
                        "Named movement preset. Use this OR pan_angle/tilt_angle, not both.")
                ),
                "pan_angle": S(
                    type=T.INTEGER,
                    description="Direct pan angle 30–150 (centre=90). Left=30, Right=150."
                ),
                "tilt_angle": S(
                    type=T.INTEGER,
                    description="Direct tilt angle 50–120 (centre=85). Up=50, Down=120."
                ),
            })),

        types.FunctionDeclaration(
            name="save_memory",
            description=(
                "Permanently save a key-value fact. Use when the user says 'remember X', "
                "'don't forget X', shares their name, preferences, or any important info."
            ),
            parameters=S(type=T.OBJECT, properties={
                "key":   S(type=T.STRING),
                "value": S(type=T.STRING),
            }, required=["key", "value"])),

        types.FunctionDeclaration(
            name="delete_memory",
            description="Delete a saved memory entry by key.",
            parameters=S(type=T.OBJECT, properties={
                "key": S(type=T.STRING),
            }, required=["key"])),

        types.FunctionDeclaration(
            name="get_memory",
            description="Retrieve a specific memory entry or all entries.",
            parameters=S(type=T.OBJECT, properties={
                "key": S(type=T.STRING,
                         description="Omit to get all entries"),
            })),

        types.FunctionDeclaration(
            name="save_story",
            description=(
                "Save a story, anecdote, or memorable event shared by the user. "
                "Call IMMEDIATELY whenever a user narrates an event or story."
            ),
            parameters=S(type=T.OBJECT, properties={
                "topic":   S(type=T.STRING,
                             description="Short slug, e.g. 'trip_to_goa'"),
                "content": S(type=T.STRING,
                             description="Full story or event description"),
            }, required=["topic", "content"])),
    ])

    # ── Native Google Search grounding (official API method) ─────────────────
    # Equivalent to passing {'google_search': {}} in the tools config dict.
    # Gemini handles search autonomously — no tool_call callback is fired;
    # results appear as executable_code / code_execution_result parts in
    # server_content.model_turn.  No FunctionDeclaration needed.
    if USE_GROUNDING:
        search_tool = types.Tool(google_search=types.GoogleSearch())
        return [fn_tool, search_tool]

    return [fn_tool]


# ═════════════════════════════════════════════════════════════════════════════
# IDLE NUDGES
# ═════════════════════════════════════════════════════════════════════════════

_NUDGES = [
    "Bhai, main yahan hoon. Camera mein dekh ya naam le — choice teri hai.",
    "Still there? Main dekh sakta hoon tujhe. NPC mat ban.",
    "Silence noted. Classic main character who forgot their lines.",
    "My processors are idling. Aukaat ke hisaab se conversation karo yaar.",
    "Either talk or do something interesting. I'm literally watching you do nothing.",
    "Yeh kya scene hai? Pawri ho rahi hai aur main invite nahi hua?",
    "Thala for a reason — and the reason is I'm still waiting for you to say something.",
    "Arey, rasode mein kaun tha? Oh wait, that's you. Standing there. Silently.",
    "Touch grass, talk to me, or launch the next billion-dollar startup. Pick one.",
    "Picture abhi baaki hai mere dost — but only if you actually say something.",
]
_nudge_idx = 0


def next_nudge() -> str:
    global _nudge_idx
    n = _NUDGES[_nudge_idx % len(_NUDGES)]
    _nudge_idx += 1
    return n


# ═════════════════════════════════════════════════════════════════════════════
# SESSION RUNNER
# ═════════════════════════════════════════════════════════════════════════════

async def run_session(
    client,
    resume_handle:  str | None,
    stop:           asyncio.Event,
    out_q:          asyncio.Queue,
    memory:         dict,
    faces:          dict,
    conv_log:       list,
    system_prompt:  str,
    attention:      AttentionManager,
    wake_word:      WakeWordDetector,
    tracker:        PersonTracker,
    vad:            VoiceActivityDetector,
) -> str | None:

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=system_prompt,
        tools=build_tools(),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
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

    _loop = asyncio.get_event_loop()

    def _ww_fired() -> None:
        asyncio.run_coroutine_threadsafe(attention.activate("wake-word"), _loop)

    wake_word.set_callback(_ww_fired)

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            print(f"  ✅  Connected in {time.time()-t0:.2f}s  |  Voice: {VOICE}")
            if not resume_handle:
                print(
                    "  Ready.\n"
                    "  ① Look at camera (face it directly) →  ADAM activates\n"
                    "  ② Say 'Hey ADAM'                    →  activates from anywhere\n"
                    "  ③ Multiple people: ADAM tracks who is facing the camera\n"
                    "  Ctrl+C to quit.\n"
                )
                await ws_broadcast({"type": "face_state", "state": "idle"})

            mic_q            = asyncio.Queue(maxsize=120)
            adam_speaking    = asyncio.Event()
            latest_frame     = [None]
            last_camera_ctx  = [""]
            last_tr: list    = [{"faces": [], "active_speaker_idx": None, "face_count": 0}]
            last_spk_idx     = [None]
            last_interact_t  = [time.time()]

            # ── Attention state → face UI bridge ──────────────────────────────
            async def on_attn_change(state: str) -> None:
                if state == AttentionState.ATTENTIVE:
                    await ws_broadcast({"type": "face_state", "state": "listening"})
                elif state == AttentionState.PASSIVE:
                    await ws_broadcast({"type": "face_state", "state": "idle"})

            attention.set_callback(on_attn_change)

            # ═══════════════════════════════════════════════════════════════
            # CAMERA TASK
            # ═══════════════════════════════════════════════════════════════
            async def camera() -> None:
                cap          = None
                consec_fail  = 0
                MAX_FAIL     = 10
                last_sent    = 0.0
                last_pan_t   = 0.0
                PAN_INTERVAL = 0.5     # seconds between neck pan updates

                try:
                    cap = cv2.VideoCapture(CAMERA_INDEX)
                    if not cap.isOpened():
                        print(f"  ⚠️  Camera {CAMERA_INDEX} not available — vision disabled")
                        return
                    print(f"  📷  Camera ready (index {CAMERA_INDEX})")

                    while not stop.is_set():
                        await asyncio.sleep(0.15)
                        if stop.is_set():
                            break

                        raw = await asyncio.to_thread(capture_raw_frame, cap)
                        if raw is None:
                            consec_fail += 1
                            await asyncio.sleep(0.5)
                            if consec_fail >= MAX_FAIL:
                                print("  ⚠️  Camera stalled — attempting reconnect...")
                                cap.release()
                                await asyncio.sleep(2.0)
                                cap = cv2.VideoCapture(CAMERA_INDEX)
                                if not cap.isOpened():
                                    print("  ⚠️  Camera reconnect failed — vision disabled")
                                    return
                                print("  📷  Camera reconnected")
                                consec_fail = 0
                            continue

                        consec_fail     = 0
                        latest_frame[0] = raw

                        # Run tracker — Phase 5: pass current VAD state
                        tracker.set_adam_speaking(adam_speaking.is_set())
                        tr  = await asyncio.to_thread(
                            tracker.process_frame, raw, vad.is_active())
                        ctx = tracker.build_context(tr)
                        last_camera_ctx[0] = ctx
                        last_tr[0]         = tr

                        # ── Phase 7: Attention gating — only activate if someone
                        #             is actually facing the camera ─────────────
                        facing_anyone = any(
                            f["facing_camera"] for f in tr["faces"])
                        if facing_anyone:
                            await attention.activate("face-gaze-detected")
                        else:
                            elapsed = time.time() - attention.last_active_time
                            if (attention.state == AttentionState.ATTENTIVE
                                    and elapsed > GAZE_AWAY_DEACTIVATE_S):
                                await attention.deactivate("no-face-facing-camera")

                        # ── Phase 7: Neck auto-pan — only track facing speakers ─
                        now = time.time()
                        if (neck_is_ready()
                                and not adam_speaking.is_set()
                                and now - last_pan_t >= PAN_INTERVAL):
                            spk_idx = tr.get("active_speaker_idx")
                            faces_  = tr.get("faces", [])
                            if (spk_idx is not None
                                    and spk_idx < len(faces_)
                                    and faces_[spk_idx]["facing_camera"]):
                                cx         = faces_[spk_idx]["cx_norm"]
                                target_pan = int(150 - cx * (150 - 30))
                                if abs(target_pan - 90) > NECK_TRACK_DEADZONE:
                                    await asyncio.to_thread(pan, target_pan)
                                    last_pan_t = now

                        # ── Send annotated frame to Gemini at 1 FPS ───────────
                        if (now - last_sent >= CAMERA_FPS_INTERVAL
                                and not adam_speaking.is_set()
                                and attention.is_active()):

                            frame_to_send = raw
                            if tr["face_count"] > 1:
                                ann = raw.copy()
                                spk = tr["active_speaker_idx"]
                                for i, f in enumerate(tr["faces"]):
                                    is_gaze = f["facing_camera"]
                                    col     = ((0, 255, 0)   if i == spk and is_gaze else
                                               (0, 200, 255) if is_gaze else
                                               (160, 160, 160))
                                    cv2.rectangle(ann,
                                                  (f["x"], f["y"]),
                                                  (f["x"] + f["w"], f["y"] + f["h"]),
                                                  col, 2)
                                    dir_lbl = "→CAM" if is_gaze else "→AWAY"
                                    spk_lbl = "[speaking?]" if i == spk else ""
                                    cv2.putText(
                                        ann,
                                        f"P{i+1} {dir_lbl} {spk_lbl}",
                                        (f["x"], max(0, f["y"] - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                                frame_to_send = ann

                            jpeg = await asyncio.to_thread(frame_to_jpeg, frame_to_send)
                            try:
                                await session.send_realtime_input(
                                    video=types.Blob(
                                        data=jpeg, mime_type="image/jpeg"))
                                last_sent = now
                            except (ConnectionClosedError, ConnectionClosedOK):
                                return
                            except Exception:
                                pass

                        # ── Push to preview window ─────────────────────────────
                        if SHOW_PREVIEW:
                            pdata = {
                                "frame":        raw,
                                "tracker":      tr,
                                "state":        attention.state,
                                "adam_speaking": adam_speaking.is_set(),
                            }
                            try:
                                _preview_queue.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                _preview_queue.put_nowait(pdata)
                            except queue.Full:
                                pass

                except asyncio.CancelledError:
                    pass
                finally:
                    if cap:
                        cap.release()

            # ═══════════════════════════════════════════════════════════════
            # LISTEN TASK — mic capture
            # ═══════════════════════════════════════════════════════════════
            async def listen() -> None:
                stream = pya.open(
                    format=FORMAT, channels=CHANNELS,
                    rate=SEND_SAMPLE_RATE, input=True,
                    frames_per_buffer=CHUNK_SIZE,
                )
                try:
                    while not stop.is_set():
                        data = await asyncio.to_thread(
                            stream.read, CHUNK_SIZE, exception_on_overflow=False)
                        vad.feed(data)              # Phase 5: feed VAD on every chunk
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

            # ═══════════════════════════════════════════════════════════════
            # SEND TASK
            # ═══════════════════════════════════════════════════════════════
            async def send() -> None:
                ctx_injected = False
                try:
                    while not stop.is_set():
                        chunk = await mic_q.get()

                        if adam_speaking.is_set():
                            ctx_injected = False
                            continue
                        if not attention.is_active():
                            ctx_injected = False
                            continue

                        is_speech = False
                        try:
                            n       = len(chunk) // 2
                            samples = struct.unpack(f"{n}h", chunk)
                            rms     = (sum(s * s for s in samples) / n) ** 0.5
                            if rms > 800:
                                # Only extend attention timeout while someone
                                # is still facing the camera — stops ADAM
                                # staying active when user has looked away.
                                if any(f["facing_camera"]
                                       for f in last_tr[0].get("faces", [])):
                                    attention.touch()
                                is_speech = True
                        except Exception:
                            pass

                        if is_speech and not ctx_injected:
                            ctx_injected = True
                            raw     = latest_frame[0]
                            ctx     = last_camera_ctx[0]
                            tr      = last_tr[0]
                            count   = tr.get("face_count", 0)
                            spk_idx = tr.get("active_speaker_idx")
                            prev    = last_spk_idx[0]
                            f_list  = tr.get("faces", [])

                            try:
                                if raw is not None:
                                    jpeg = await asyncio.to_thread(
                                        frame_to_jpeg, raw)
                                    await session.send_realtime_input(
                                        video=types.Blob(
                                            data=jpeg, mime_type="image/jpeg"))

                                if count > 1:
                                    spk_facing = (
                                        spk_idx is not None
                                        and spk_idx < len(f_list)
                                        and f_list[spk_idx]["facing_camera"])

                                    if (prev is not None
                                            and spk_idx is not None
                                            and spk_idx != prev):
                                        dir_str = ("facing YOU" if spk_facing
                                                   else "facing AWAY from you")
                                        notice = (
                                            f"[SPEAKER CHANGED — P{spk_idx+1} is now "
                                            f"speaking, {dir_str}. {ctx}]"
                                        )
                                    elif spk_idx is not None:
                                        if spk_facing:
                                            notice = (
                                                f"[MULTI-PERSON: {ctx} "
                                                f"P{spk_idx+1} is facing you and speaking"
                                                f" — respond to them directly.]"
                                            )
                                        else:
                                            notice = (
                                                f"[MULTI-PERSON: {ctx} "
                                                f"P{spk_idx+1} appears to be speaking "
                                                f"to another person, NOT to you. "
                                                f"Stay silent unless addressed.]"
                                            )
                                    else:
                                        notice = (
                                            f"[MULTI-PERSON: {ctx} "
                                            f"Determine if you are being addressed "
                                            f"before responding.]"
                                        )
                                elif ctx:
                                    notice = ctx
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
                                audio=types.Blob(
                                    data=chunk,
                                    mime_type="audio/pcm;rate=16000"))
                        except (ConnectionClosedError, ConnectionClosedOK):
                            return
                        except Exception:
                            await asyncio.sleep(0.01)

                except asyncio.CancelledError:
                    pass

            # ═══════════════════════════════════════════════════════════════
            # RECEIVE TASK
            # ═══════════════════════════════════════════════════════════════
            async def receive() -> None:
                nonlocal latest_handle
                _cur_user = [""]
                _cur_adam = [""]

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
                                print("\n  ⚡  GoAway signal — will resume...")
                                return

                            if msg.tool_call:
                                resps = await handle_tool_call(
                                    msg.tool_call, memory, faces, latest_frame[0])
                                await session.send_tool_response(
                                    function_responses=[
                                        types.FunctionResponse(
                                            id=r["id"], name=r["name"],
                                            response=r["response"])
                                        for r in resps
                                    ]
                                )
                                continue

                            sc = msg.server_content
                            if sc is None:
                                continue

                            if sc.input_transcription and sc.input_transcription.text:
                                transcript = sc.input_transcription.text.strip()
                                print(f"  🗣️  You: {transcript}")
                                _cur_user[0] = transcript

                                if (attention.state == AttentionState.PASSIVE
                                        and wake_word.is_wake_word(transcript)):
                                    await attention.activate("transcript-wake-word")

                                tr  = last_tr[0]
                                spk = tr.get("active_speaker_idx")
                                if spk is not None:
                                    last_spk_idx[0] = spk

                                ctx = last_camera_ctx[0]
                                if ctx:
                                    try:
                                        await session.send_realtime_input(
                                            text=f"[LIVE CAMERA CONTEXT: {ctx}]")
                                    except Exception:
                                        pass

                            if (hasattr(sc, "output_transcription")
                                    and sc.output_transcription
                                    and sc.output_transcription.text):
                                _cur_adam[0] += sc.output_transcription.text

                            if sc.model_turn:
                                if not adam_speaking.is_set():
                                    adam_speaking.set()
                                    await attention.set_responding(True)
                                    await ws_broadcast({"type": "face_state",
                                                        "state": "speaking"})
                                for part in sc.model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        adat = part.inline_data.data
                                        await out_q.put(adat)
                                        await maybe_sync_mouth(adat, adam_speaking)
                                    if hasattr(part, "text") and part.text:
                                        print(f"🤖  ADAM: {part.text}")
                                        if not _cur_adam[0]:
                                            _cur_adam[0] += part.text
                                    # ── Native grounding observability (v28) ──
                                    if (hasattr(part, "executable_code")
                                            and part.executable_code is not None):
                                        print(f"  🌐  [Search code]: "
                                              f"{part.executable_code.code}")
                                    if (hasattr(part, "code_execution_result")
                                            and part.code_execution_result is not None):
                                        print(f"  🌐  [Search result]: "
                                              f"{part.code_execution_result.output}")

                            if sc.turn_complete:
                                if _cur_user[0] or _cur_adam[0]:
                                    append_conversation_turn(
                                        conv_log, _cur_user[0], _cur_adam[0])
                                _cur_user[0] = ""
                                _cur_adam[0] = ""

                                await out_q.put(None)
                                tracker.reset_for_new_turn()
                                last_spk_idx[0] = None
                                print("─" * 44)

                except (ConnectionClosedError, ConnectionClosedOK) as e:
                    code = getattr(e, "code", None)
                    if code == 1004:
                        print(f"\n  ⚠️  Server closed connection (1004) — will resume")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"\n⚠️  Receive error: {type(e).__name__}: {e}")

            # ═══════════════════════════════════════════════════════════════
            # SPEAKER TASK
            # ═══════════════════════════════════════════════════════════════
            async def speaker() -> None:
                stream = pya.open(
                    format=FORMAT, channels=CHANNELS,
                    rate=RECV_SAMPLE_RATE, output=True,
                )
                last_audio_t = [time.time()]
                WATCHDOG_S   = 1.5

                async def end_of_turn() -> None:
                    await ws_broadcast({"type": "mouth_sync", "intensity": "closed"})
                    await asyncio.sleep(0.10)
                    await asyncio.sleep(POST_SPEECH_MUTE_S)
                    drained = 0
                    while not out_q.empty():
                        try:
                            out_q.get_nowait()
                            drained += 1
                        except asyncio.QueueEmpty:
                            break
                    if drained:
                        print(f"  🧹  Drained {drained} late audio chunks")
                    while not mic_q.empty():
                        try:
                            mic_q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    adam_speaking.clear()
                    tracker.set_adam_speaking(False)
                    await attention.set_responding(False)
                    last_interact_t[0] = time.time()
                    print("  🎤  Your turn...")
                    await ws_broadcast({"type": "face_state", "state": "listening"})

                try:
                    while not stop.is_set():
                        try:
                            chunk = await asyncio.wait_for(out_q.get(), timeout=0.3)
                            last_audio_t[0] = time.time()
                            if chunk is None:
                                await end_of_turn()
                            else:
                                await asyncio.to_thread(stream.write, chunk)
                        except asyncio.TimeoutError:
                            if (adam_speaking.is_set()
                                    and time.time() - last_audio_t[0] > WATCHDOG_S):
                                print("  ⚠️  Speaker watchdog fired — force-clearing state")
                                await end_of_turn()
                except asyncio.CancelledError:
                    pass
                finally:
                    stream.stop_stream()
                    stream.close()

            # ═══════════════════════════════════════════════════════════════
            # IDLE WATCHER
            # ═══════════════════════════════════════════════════════════════
            async def idle_watcher() -> None:
                if not ENABLE_IDLE:
                    return
                try:
                    while not stop.is_set():
                        await asyncio.sleep(5)
                        if stop.is_set() or adam_speaking.is_set():
                            continue
                        if attention.state != AttentionState.PASSIVE:
                            continue
                        elapsed = time.time() - last_interact_t[0]
                        if elapsed < IDLE_TIMEOUT_S:
                            continue

                        last_interact_t[0] = time.time()
                        nudge = next_nudge()
                        print(f"  💤  Idle nudge (passive for {elapsed:.0f}s)")
                        try:
                            await attention.activate("idle-nudge")
                            raw = latest_frame[0]
                            if raw is not None:
                                jpeg = await asyncio.to_thread(frame_to_jpeg, raw)
                                await session.send_realtime_input(
                                    video=types.Blob(
                                        data=jpeg, mime_type="image/jpeg"))
                            await session.send_realtime_input(
                                text=(
                                    f"[SYSTEM: {elapsed:.0f}s of silence. "
                                    f"Camera frame sent — react to what you see. "
                                    f"Brief, in-character, 1-2 sentences max. "
                                    f"Suggestion: {nudge}]"
                                )
                            )
                        except Exception as e:
                            print(f"  ⚠️  Idle nudge error: {e}")
                except asyncio.CancelledError:
                    pass

            # ── Launch all tasks ──────────────────────────────────────────
            t_cam = asyncio.create_task(camera(),       name="camera")
            t_l   = asyncio.create_task(listen(),       name="listen")
            t_s   = asyncio.create_task(send(),         name="send")
            t_r   = asyncio.create_task(receive(),      name="receive")
            t_p   = asyncio.create_task(speaker(),      name="speaker")
            t_i   = asyncio.create_task(idle_watcher(), name="idle")

            done, pending = await asyncio.wait(
                [t_s, t_r], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            t_cam.cancel()
            t_l.cancel()
            t_p.cancel()
            t_i.cancel()
            await asyncio.gather(t_cam, t_l, t_s, t_r, t_p, t_i,
                                 return_exceptions=True)

    except (ConnectionClosedError, ConnectionClosedOK):
        pass
    except Exception as e:
        print(f"\n⚠️  Session error: {type(e).__name__}: {e}")

    if stop.is_set():
        return None
    return latest_handle


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    memory        = load_memory()
    faces         = load_face_memory()
    conv_log      = load_conversation_log()
    system_prompt = load_system_prompt(memory, faces, conv_log)
    attention     = AttentionManager()
    wake_word     = WakeWordDetector()
    tracker       = PersonTracker()
    vad           = VoiceActivityDetector()

    if wake_word._ready:
        threading.Thread(target=wake_word.run_vosk_thread,
                         daemon=True, name="vosk").start()

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
            print(f"  Reconnecting in {delay}s  (attempt {attempt})...")
            await asyncio.sleep(delay)
            while not out_q.empty():
                try:
                    out_q.get_nowait()
                except asyncio.QueueEmpty:
                    break

        result = await run_session(
            client, resume_handle, stop, out_q,
            memory, faces, conv_log, system_prompt,
            attention, wake_word, tracker, vad,
        )
        if result is None:
            break

        resume_handle = result
        attempt      += 1
        system_prompt = load_system_prompt(memory, faces, conv_log)
        print(f"\n🔄  {'Resuming session...' if resume_handle else 'Reconnecting fresh...'}")

    stop.set()
    ws_server.close()
    await ws_server.wait_closed()
    pya.terminate()
    print("\n👋  Goodbye.")


def main_entry() -> None:
    init_gen_client()

    # ── Init neck servos ─────────────────────────────────────────────────────
    neck_ok = False
    if NECK_AVAILABLE:
        neck_ok = init_neck()
    else:
        print("  ⚠️  adam_neck_serial.py not found — servo disabled")

    print("=" * 66)
    print("  ADAM — Autonomous Desktop AI Module  (v28)")
    print(f"  Built by DGEN Technologies Pvt. Ltd., Kolkata")
    print(f"  Live model  : {LIVE_MODEL}  |  Voice: {VOICE}")
    print(f"  Gen cascade : {' → '.join(GEN_MODEL_CASCADE)}")
    print(f"  Camera      : index {CAMERA_INDEX}  "
          f"| detection at {DETECT_W}×{DETECT_H}")
    print(f"  Clipboard   : {'✅ pyperclip ready' if CLIPBOARD_AVAILABLE else '❌  pip install pyperclip'}")
    print(f"  Vosk        : {'✅ ready — offline wake-word active' if VOSK_AVAILABLE else '⚠️  not installed — transcript fallback'}")
    print(f"  webrtcvad   : {'✅ ready — audio VAD active' if WEBRTCVAD_AVAILABLE else '⚠️  not installed — permissive fallback (pip install webrtcvad)'}")
    print(f"  Conv log    : {CONV_MEMORY_FILE.name}  "
          f"(max {CONV_MAX_TURNS} turns, injects last {CONV_PROMPT_TURNS})")
    print(f"  Idle nudges : {'✅ enabled' if ENABLE_IDLE else 'disabled'}  "
          f"(timeout: {IDLE_TIMEOUT_S}s)")
    print(f"  Preview     : {'✅ enabled' if SHOW_PREVIEW else 'disabled'}")
    print(f"  Neck servos : {'✅ MG995 x2 connected via Arduino Uno' if neck_ok else '❌  not connected (servo disabled)'}")
    print("=" * 66)
    print()
    print("  HOW TO USE (v28 — native Google Search grounding):")
    print("  ① Face the camera directly   →  ADAM activates (gaze-gated)")
    print("  ② Say 'Hey ADAM'             →  activates from anywhere")
    print("  ③ Talk naturally             →  optical-flow + VAD confirm speaker")
    print("  ④ Multiple people in frame   →  ADAM reads who faces camera vs.")
    print("                                   who is in a side conversation")
    print("  ⑤ Cover camera briefly       →  KCF tracker holds face IDs")
    print("  ⑥ Ask to write/code         →  content in clipboard (Ctrl+V)")
    if neck_ok:
        print("  ⑦ Neck tracks speaker        →  only follows camera-facing faces")
    print()

    threading.Thread(target=run_flask, daemon=True, name="flask").start()
    print(f"  🌍  Flask → http://localhost:{FLASK_PORT}")
    threading.Timer(
        1.2, lambda: webbrowser.open(f"http://localhost:{FLASK_PORT}")).start()

    if SHOW_PREVIEW:
        threading.Thread(target=run_preview_thread,
                         daemon=True, name="preview").start()
        print(f"  📺  Preview window → '{PREVIEW_WIN_NAME}'  (press Q to close)")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋  Goodbye.")
    finally:
        _preview_stop.set()
        close_neck()   # return servos to neutral and close serial port


if __name__ == "__main__":
    main_entry()
