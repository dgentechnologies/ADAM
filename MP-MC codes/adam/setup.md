# ADAM v40 — Raspberry Pi Zero 2 W Setup (fresh Debian 13 "trixie")

Definitive, tested bring-up guide for the **current** ADAM architecture (the
split `adam/` package with `main.py` as entrypoint). This supersedes the older
`ADAM_Pi_Zero_2W_Setup_Guide.md` / `ADAM_Raspberry_Pi_Complete_Setup.md` guides,
which describe a superseded display/audio flow (Pi-side ILI9341/luma.lcd,
pyaudio). Where those docs and this one disagree, **this one and the code win.**

Verified end-to-end on: Raspberry Pi Zero 2 W · Debian 13 (trixie) ·
kernel 6.18.x aarch64 · Python 3.13.5 · Google voiceHAT · Gemini Live.

---

## 1. Architecture (what actually runs)

```
                       ┌──────────────────────── Raspberry Pi Zero 2 W ───────────────────────┐
  INMP441 x2  ─I2S──▶  │  arecord (voiceHAT capture, S32_LE 48k 2ch)                           │
  (dual mic)           │        │                                                              │
                       │        ▼   audio_utils DSP (S32→S16 mono 16k, DOA via GCC-PHAT)        │
  MAX98357A  ◀─I2S──   │  aplay (voiceHAT playback, S16_LE 48k 2ch)  ◀── Gemini audio out       │
  (speaker)            │        ▲                                                              │
                       │   ┌────┴──── session.py (asyncio: listen/send/receive/speaker/…)  ────┤
                       │   │         └── google-genai  ⇄  Gemini Live (gemini-3.1-flash-live)   │
                       │   │                                                                    │
  Pan servo  ◀─GPIO12─ │  hardware.servo_pan()  (gpiozero AngularServo, direct PWM)             │
                       │                                                                        │
  ESP32-CAM  ◀─UART──▶ │  esp32_link.py  /dev/serial0 @ 921600  (PL011)                          │
     │                 │     • RECV  'F'+len+JPEG (camera frames), 'T' touch, 'G' gesture       │
     └─▶ Pico          │     • SEND  "TILT:<deg>", "EMO:<emotion>", "CAM:ON/OFF"                 │
        (tilt servo    └────────────────────────────────────────────────────────────────────── ┘
         + TFT face)
```

- **Audio** — Google voiceHAT soundcard (2× INMP441 I2S mics + MAX98357A I2S amp).
  Driven via `arecord`/`aplay` subprocess pipes, **not** pyaudio.
- **Pan servo** — driven **directly** by the Pi (gpiozero on **GPIO 12**).
- **Tilt servo + TFT face + camera** — the Pi does **not** drive these directly.
  It talks over one UART (`/dev/serial0` @ 921600) to the **ESP32‑CAM**, which
  relays to the **Pico**. Camera JPEG frames come back over the same UART.
- **Wake word** — Vosk offline STT (`vosk-model-small-en-us-0.15`) for idle
  re-attention; preloaded once at startup.
- **Graceful degradation** — if the ESP32 isn't wired/powered, ADAM prints a
  warning and runs **audio-only** (no vision/touch/tilt/face). This is normal.

> ESP32‑CAM firmware (`esp32_cam.ino`) and Pico firmware are **separate** and
> flashed with the Arduino toolchain — out of scope for this Pi guide.

---

## 2. Hardware / wiring (authoritative values are in `config.py`)

| Function            | Pi pin / interface        | Notes                                   |
|---------------------|---------------------------|-----------------------------------------|
| Pan servo signal    | **GPIO 12** (PWM)          | `NECK_GPIO_PIN`; PW 0.5–2.5 ms          |
| ESP32‑CAM UART TX   | **GPIO 14** (TXD → ESP RX) | `/dev/serial0` @ **921600**             |
| ESP32‑CAM UART RX   | **GPIO 15** (RXD ← ESP TX) | cross TX/RX; common GND                 |
| Mics + speaker      | voiceHAT (I2S)             | card **`sndrpigooglevoi`**              |

> If any wiring differs from these, **trust the pin numbers in `config.py`** —
> it is the single source of truth.

---

## 3. First boot, SSH, base update

Flash Raspberry Pi OS (Debian 13, 64‑bit) with Raspberry Pi Imager; in the
Imager's OS‑customisation set the **hostname**, **user/password**, **Wi‑Fi**, and
**enable SSH**. Then:

```bash
ssh pi@<hostname>.local          # e.g. ssh pi@adam-pi.local
sudo apt update && sudo apt full-upgrade -y
```

---

## 4. Enable interfaces + free the UART for the ESP32

Edit `/boot/firmware/config.txt` so it contains these lines (most are present by
default on a fresh image — add the ones that are missing):

```ini
dtparam=i2c_arm=on
dtparam=spi=on
dtparam=i2s=on
dtparam=audio=on
dtoverlay=googlevoicehat-soundcard   # makes the voiceHAT enumerate (card "sndrpigooglevoi")
enable_uart=1                        # exposes the UART on GPIO14/15
dtoverlay=disable-bt                 # frees the PL011 -> /dev/serial0 becomes ttyAMA0 (stable @921600)
```

Why `disable-bt`: by default `/dev/serial0` maps to the **mini‑UART** (`ttyS0`),
whose baud rate drifts with the core clock and is unreliable at 921600.
`disable-bt` hands the stable **PL011** (`ttyAMA0`) to `/dev/serial0`. ADAM does
not use Bluetooth. (To restore BT later: remove that line and reboot.)

**Disable the serial login console** (otherwise a getty holds `/dev/serial0` and
the ESP32 link fails). Remove `console=serial0,115200` from
`/boot/firmware/cmdline.txt` (keep `console=tty1`), then:

```bash
sudo sed -i 's/console=serial0,115200 //g' /boot/firmware/cmdline.txt
sudo systemctl disable --now serial-getty@ttyS0.service
```

Make sure the `pi` user can access the port and GPIO (default on Pi OS):

```bash
groups pi        # expect: ... dialout audio video gpio spi i2c ...
sudo usermod -aG dialout,audio,gpio,spi,i2c pi   # only if any are missing
```

---

## 5. System packages (apt)

These provide the ARM‑native, apt‑maintained builds of numpy/serial/GPIO so the
venv doesn't have to compile them:

```bash
sudo apt install -y \
  python3-venv python3-dev python3-numpy python3-serial \
  python3-gpiozero python3-lgpio python3-requests \
  alsa-utils git
```

---

## 6. Get the ADAM code onto the Pi

Put the entire `adam/` package (all `*.py` + `SystemPrompt.txt` + `song*.wav`) in
**`/home/pi/adam`**. Any transfer works — e.g. from your dev machine:

```bash
rsync -avz --exclude venv --exclude '__pycache__' --exclude 'vosk-model*' \
      ./adam/ pi@adam-pi.local:/home/pi/adam/
```

`~/adam` must end up containing `main.py`, `config.py`, `session.py`, the other
modules, `SystemPrompt.txt`, and `song1.wav`/`song2.wav`/`song3.wav`.

---

## 7. Python virtual environment + dependencies

Create the venv **with system site‑packages** so it reuses apt's numpy/serial/
gpiozero/lgpio/requests, then pip‑install only what apt doesn't provide:

```bash
python3 -m venv --system-site-packages ~/adam/venv
~/adam/venv/bin/pip install --upgrade pip
~/adam/venv/bin/pip install google-genai websockets vosk zeroconf ddgs python-dotenv
```

> Debian 13 is PEP‑668 "externally managed" — you **must** use the venv;
> `pip install` outside it is blocked. Always run ADAM with
> `~/adam/venv/bin/python`, never the system `python3`.

Verify the split resolved correctly (pip pkgs → venv, native pkgs → system):

```bash
~/adam/venv/bin/python - <<'PY'
import importlib.util as u
for m in ["google.genai","websockets","vosk","zeroconf","ddgs","dotenv",
          "numpy","serial","gpiozero","lgpio","requests"]:
    s=u.find_spec(m); print(f"{m:14}", "OK" if s else "MISSING", s.origin if s else "")
PY
```

---

## 8. Vosk offline wake‑word model

```bash
cd ~/adam
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
rm vosk-model-small-en-us-0.15.zip
```

Result: `~/adam/vosk-model-small-en-us-0.15/` (≈68 MB). Path is configurable via
`VOSK_MODEL_PATH`; the default is this folder.

---

## 9. API key (`.env`)

Create **`/home/pi/adam/.env`** (loaded automatically by `config.py`):

```bash
cat > ~/adam/.env <<'EOF'
GEMINI_API_KEY=your_real_key_here
# Optional overrides:
# PI_UART_PORT=/dev/serial0
# PI_UART_BAUD=921600
# CAPTURE_DEVICE=plughw:sndrpigooglevoi,0
# PLAYBACK_DEVICE=plughw:sndrpigooglevoi,0
# LAPTOP_AGENT_IP=192.168.1.50     # static fallback if mDNS discovery fails
# AGENT_TOKEN=shared_secret
EOF
chmod 600 ~/adam/.env
```

`config.py` raises `ValueError: GEMINI_API_KEY not set` if this is missing.

---

## 10. Optimize for the Zero 2 W (console boot)

The Pi Zero 2 W has only ~415 MB RAM. Boot to console (no desktop) to free
~100 MB and stop swapping (ADAM's peak RSS is ~140 MB):

```bash
sudo systemctl set-default multi-user.target
sudo systemctl disable lightdm.service
```

Reversible any time: `sudo systemctl set-default graphical.target && sudo systemctl enable lightdm`.

**Reboot now** to apply the boot/UART/BT/console changes:

```bash
sudo reboot
```

---

## 11. Verify

After reboot, confirm the environment:

```bash
# UART freed and on the stable PL011:
readlink -f /dev/serial0            # -> /dev/ttyAMA0
sudo fuser /dev/serial0 || echo "serial0 free"

# voiceHAT present (card "sndrpigooglevoi", usually index 1):
arecord -l ; aplay -l

# Memory headroom (Swap used should be ~0):
free -h
```

**Audio round‑trip** (record 2 s, then hear it back):

```bash
arecord -D plughw:sndrpigooglevoi,0 -f S32_LE -r 48000 -c 2 -d 2 /tmp/t.wav
aplay   -D plughw:sndrpigooglevoi,0 /tmp/t.wav ; rm /tmp/t.wav
```

> **Address the card by name (`plughw:sndrpigooglevoi,0`), not by index.**
> ALSA numbers the voiceHAT *after* the HDMI card, so its index isn't fixed, and
> HDMI (card 0) has **no capture** device — `plughw:0,0` breaks the mics.

**Manual run** of ADAM (Ctrl‑C to stop):

```bash
cd ~/adam && ./venv/bin/python main.py
```

You should see the banner, `✅ Connected to Gemini Live`, the task list
(listen/send/receive/speaker/camera/gesture), and `🎤 Mic RMS: …`. The
`UART … no data received … audio-only mode` line is expected until the ESP32‑CAM
is wired and powered.

---

## 12. Run at boot (systemd service)

Create `/etc/systemd/system/adam.service`:

```ini
[Unit]
Description=ADAM v40 — Autonomous Desktop AI Module
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/pi/adam
ExecStart=/home/pi/adam/venv/bin/python -u /home/pi/adam/main.py
KillSignal=SIGTERM
TimeoutStopSec=15
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now adam.service
```

`main.py` handles SIGTERM gracefully (camera off, servo centered, memory flushed),
so `systemctl stop`/`restart` shut down cleanly.

### Operating it

```bash
sudo systemctl status adam         # is it running?
journalctl -u adam -f              # live logs
sudo systemctl restart adam        # after a code/.env change
sudo systemctl stop adam           # stop (e.g. to run main.py by hand)
sudo systemctl disable --now adam  # don't start at boot
```

---

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `ValueError: GEMINI_API_KEY not set` | Populate `~/adam/.env` (§9); it's loaded from `BASE_DIR/.env`. |
| `arecord: ... No such ...` / mics dead | Use `plughw:sndrpigooglevoi,0` (name, not index). Check `arecord -l`. HDMI card 0 has no capture. |
| `Permission denied` on `/dev/serial0` | `sudo usermod -aG dialout pi`, re‑login/reboot. |
| ESP32 "no data received … audio‑only" | Expected with no ESP32. Else: power/wiring, TX↔RX not swapped, baud 921600, `readlink -f /dev/serial0` = `ttyAMA0`, no getty on it. |
| Serial console still grabs the port | `console=serial0,115200` still in `cmdline.txt`, or `serial-getty@ttyS0` enabled (§4). |
| Servo doesn't move / `pan_servo unavailable` | Check GPIO 12 wiring + servo power (servos need their own 5 V + common GND); `pi` in `gpio` group. |
| Pi reboots/hangs under load (OOM) | Ensure console boot (§10). Check `free -h` shows Swap present and RAM ~140 MB in use. |
| `externally-managed-environment` on pip | You're outside the venv. Use `~/adam/venv/bin/pip`. |

---

## 14. Reclaim SD space

The `/` filesystem is small. Safe to remove any time (all regenerate or aren't
needed at runtime):

```bash
rm -rf ~/.cache/pip                                   # pip download cache
sudo apt clean                                        # apt package cache
find ~/adam -name __pycache__ -type d -prune -exec rm -rf {} +   # bytecode (rebuilt on next run)
sudo journalctl --vacuum-size=20M                     # cap systemd logs
```

The required footprint under `~/adam` is: the `*.py` modules + `SystemPrompt.txt`
(~0.3 MB), `venv` (~75 MB), `vosk-model-small-en-us-0.15` (~68 MB), and the
`song*.wav` files you actually reference in `config.py`'s `SONG_FILE_PATHS`.
