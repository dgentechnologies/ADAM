# ADAM — Autonomous Desktop AI Module

ADAM is an AI-powered desktop assistant built by **DGEN Technologies Pvt. Ltd.**
Powered by Google Gemini Live API, it features real-time voice conversation,
emotion-driven face animations, live camera vision, and a persistent memory system.

---

## Project Structure

```
ADAM/
├── adam_live_v19_attention.py   # ★ Latest — v19.1: Gemini Live + camera + smart attention
├── adam_live_v18_camera.py      # v18: Gemini Live + OpenCV camera + face recognition
├── adam_live_v17.py             # v17: Gemini Live + Flask face UI + WebSocket
├── adam_live_v9.py              # v9:  Gemini Live API (standalone, no camera)
├── adam_live_v9_legacy.py       # v9:  Legacy variant with Google Search tool
├── adam_voice_elevenlabs.py     # Classic: Google STT + ElevenLabs TTS
├── adam_native_audio.py         # Native audio: speech-segmented Gemini input
├── wake_word_vosk.py            # Wake word detector (Vosk offline model)
├── wake_word_google.py          # Wake word detector (Google Speech Recognition)
├── adam_face.html               # Face animation UI (served via Flask)
├── system_prompt.txt            # ADAM's full personality & behaviour prompt
├── adam_memory.json             # Persistent conversation memory (auto-generated)
├── requirements_native_audio.txt
├── README_native_audio.md
└── design/
    └── media/
        ├── body.jpeg
        └── generated_design.jpeg
```

---

## Quick Start

### Latest version (recommended)

```bash
pip install --upgrade google-genai pyaudio python-dotenv websockets flask opencv-python Pillow
```

Set your API key:

```bash
# Linux / macOS
export GOOGLE_API_KEY="your_key_here"

# Windows PowerShell
$env:GOOGLE_API_KEY = "your_key_here"
```

Run:

```bash
python adam_live_v19_attention.py
```

### Classic voice-only version

```bash
pip install speechrecognition elevenlabs playsound python-dotenv google-generativeai
python adam_voice_elevenlabs.py
```

---

## Version History

| File | Version | Key Features |
|------|---------|-------------|
| `adam_live_v19_attention.py` | v19.1 | Camera + smart attention (face gaze, wake word, timeout) |
| `adam_live_v18_camera.py` | v18 | Camera + face recognition + persistent visual memory |
| `adam_live_v17.py` | v17 | Gemini Live + face UI (WebSocket + Flask) |
| `adam_live_v9.py` | v9 | Gemini Live API + session resumption + voice picker |
| `adam_native_audio.py` | — | Native audio with speech-segmentation |
| `adam_voice_elevenlabs.py` | — | Classic STT + ElevenLabs TTS pipeline |

---

## Built by

**DGEN Technologies Pvt. Ltd.** — Kolkata, India  
*"Innovate. Integrate. Inspire." | Made in India.*

- Website: [dgentechnologies.com](https://dgentechnologies.com)
- Twitter/X: [@dgen_tec](https://twitter.com/dgen_tec)
- Instagram: [@dgen_technologies](https://instagram.com/dgen_technologies)
- LinkedIn: [dgentechnologies](https://linkedin.com/company/dgentechnologies)
