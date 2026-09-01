# ADAM — Autonomous Desktop AI Module

ADAM is an AI-powered desktop assistant built by **DGEN Technologies Pvt. Ltd.**  
Powered by the Google Gemini Live API, it features real-time bidirectional voice conversation, emotion-driven face animations, live camera vision & face tracking, async web search, hardware neck servo actuation, and persistent memory.

---

## Project Directory Layout

```
ADAM/
├── adamV30.py                  # ★ Latest Entry Launcher (v30)
├── adamV30_test_async_search.py# Core Runtime (v30 async search, vision, neck tracking)
├── adamV29.py                  # Stable Precursor (v29 search cost optimizations)
├── system_prompt.txt           # Unified token-optimized system prompt
├── adam_face.html              # Face animation UI (served via Flask/WebSocket)
├── adam_memory.json            # Persistent conversation & user memory
├── adam_conversations.json     # Conversation transcript logs
├── adam_faces.json             # Facial recognition embeddings/data
│
├── emotions/                   # HTML/CSS UI assets for emotion state animations
├── faces/                      # Reference profile photos for facial recognition
├── tools/                      # Diagnostic and verification utilities
│   ├── adam_neck_serial.py     # Standalone Arduino/Serial neck test script
│   └── verify_compile.py       # Arduino/ESP32 compilation check script
│
├── MP-MC codes/                # Microprocessor / Microcontroller deployment code
│   ├── adam.py                 # Raspberry Pi / Embedded core daemon
│   ├── laptop_agent.py         # Companion agent
│   ├── esp32_cam/              # ESP32 Camera stream firmware
│   └── ADAM_Display/           # Embedded screen drivers
│
├── UNO_code/                   # Microcontroller & Arduino firmware
│   ├── picopixel_lvgl_ui/      # LVGL graphical UI for PicoPixel/ESP32
│   ├── esp32_tft_emotions/     # TFT emotion display firmware
│   └── adam_neck_servo/        # Servo neck controller firmware
│
├── mobileAPP/                  # Mobile companion application (Turborepo monorepo)
├── adam-web-demo/              # Web demonstration & relay server
├── Hardware_Docs/              # Schematics, blueprints, and setup documentation
├── design/                     # Hardware CAD / industrial design assets
└── old_versions/               # Archived historic versions (v9 through v28)
```

---

## Quick Start (v30)

### 1. Requirements & Dependencies

```bash
pip install --upgrade google-genai pyaudio python-dotenv websockets flask opencv-python Pillow pyperclip pyserial webrtcvad vosk duckduckgo-search
```

*(Optional) Download an offline Vosk speech model from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) if using offline wake-word detection.*

### 2. Environment Setup

Create or update your `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Run ADAM

```bash
python adamV30.py
```

---

## Built by

**DGEN Technologies Pvt. Ltd.** — Kolkata, India  
*"Innovate. Integrate. Inspire." | Made in India.*

- Website: [dgentechnologies.com](https://dgentechnologies.com)
- Twitter/X: [@dgen_tec](https://twitter.com/dgen_tec)
- Instagram: [@dgen_technologies](https://instagram.com/dgen_technologies)
- LinkedIn: [dgentechnologies](https://linkedin.com/company/dgentechnologies)
