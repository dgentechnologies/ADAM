# ADAM — Raspberry Pi Zero 2 W Complete Setup Guide

**Purpose:** Prepare a Raspberry Pi Zero 2 W so the ADAM code can run reliably, including the ILI9341 TFT display and the required Python/SPI/GPIO environment.

> **Important:** This guide is based on the ADAM v3 ILI9341 setup document and the setup steps completed during this session. The supplied ADAM document does **not** specify a sound-card model or sound-card GPIO/I2S wiring, so the sound-card section below is intentionally marked as hardware-dependent rather than inventing pins or packages.

---

# 1. Hardware

## Raspberry Pi

- Raspberry Pi Zero 2 W
- microSD card
- Stable 5V power supply
- Wi-Fi network

## TFT

- ILI9341 SPI TFT display

## TFT wiring

| ILI9341 pin | Pi Zero 2 W physical pin | GPIO | Purpose |
|---|---:|---:|---|
| VCC | 1 | 3.3V | Display power |
| GND | 20 | GND | Ground |
| CS | 24 | GPIO8 | SPI CE0 |
| RESET | 22 | GPIO25 | Display reset |
| DC | 18 | GPIO24 | Data/command |
| MOSI/SDI | 19 | GPIO10 | SPI MOSI |
| SCK/CLK | 23 | GPIO11 | SPI clock |
| MISO | 21 | GPIO9 | Optional for display-only use |
| LED/BL | 1 | 3.3V | Backlight |

### CRITICAL

- Use **3.3V** for the ILI9341.
- Do **not** connect the TFT to 5V.
- Power the Pi off before changing the wiring.

---

# 2. Flash Raspberry Pi OS

Use Raspberry Pi Imager.

## OS

Select:

**Raspberry Pi OS (64-bit)**

The ADAM setup uses the Lite/headless style workflow: SSH is used from Windows and no desktop GUI is required.

## Raspberry Pi Imager customisation

Set:

### Hostname

```text
adam-pi
```

### Username

```text
pi
```

### Password

Use the password you actually configured for this Pi.

During this setup the password used was:

```text
adam2026
```

### Wi-Fi

Set the Wi-Fi network that the Pi will use.

For the current home network:

```text
SSID: DASGUPTA
Password: Probhaboti@2022
Country: IN
```

### Services

Enable:

- SSH
- Password authentication

---

# 3. First boot

Insert the microSD card and power the Pi.

Wait approximately 1–2 minutes on the first boot.

The green activity LED should blink during SD-card activity.

---

# 4. Check that the Pi is reachable from Windows

On the Windows laptop, open **PowerShell**.

Run:

```powershell
ping adam-pi.local
```

If the Pi is reachable, you should get replies.

Then connect:

```powershell
ssh pi@adam-pi.local
```

Enter the Pi password.

> SSH passwords do not show characters while you type. This is normal.

A successful login ends at a prompt similar to:

```text
pi@adam-pi:~ $
```

---

# 5. If the Pi was reflashed and Windows reports a host-key error

If you see:

```text
WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!
```

remove the old saved key:

```powershell
ssh-keygen -R adam-pi.local
```

Then reconnect:

```powershell
ssh pi@adam-pi.local
```

Accept the new host key with:

```text
yes
```

---

# 6. Enable SPI

Inside the SSH session:

```bash
sudo raspi-config
```

Enter the Pi password if requested.

Navigate to:

```text
3 Interface Options
    -> I4 SPI
        -> Yes
        -> OK
        -> Finish
```

If a reboot option is shown, reboot.

The important verification is:

```bash
ls /dev/spi*
```

Expected result:

```text
/dev/spidev0.0
/dev/spidev0.1
```

If those two devices appear, SPI is enabled.

---

# 7. Update the Pi

Run:

```bash
sudo apt update && sudo apt upgrade -y
```

Wait for it to finish.

---

# 8. Install the Python and hardware packages

Run:

```bash
sudo apt install -y python3 python3-pip python3-venv git
```

Then:

```bash
sudo apt install -y python3-rpi.gpio python3-spidev
```

The ADAM setup uses:

- Python 3
- RPi.GPIO
- spidev
- Pillow
- luma.lcd

---

# 9. Install Pillow

Run:

```bash
sudo pip3 install Pillow --break-system-packages
```

---

# 10. Install the ILI9341 driver

Run:

```bash
sudo pip3 install luma.lcd --break-system-packages
```

---

# 11. Give the Pi user SPI/GPIO access

Run:

```bash
sudo usermod -aG spi,gpio pi
```

Then:

```bash
newgrp spi
```

If you later get a permission error for `/dev/spidev0.0`, reboot the Pi and reconnect.

---

# 12. Connect the TFT

With the Pi powered off, wire the ILI9341 exactly as follows:

```text
Pi Zero 2W                  ILI9341

Pin 1  (3.3V)  -----------> VCC
Pin 20 (GND)   -----------> GND
Pin 24 (GPIO8) -----------> CS
Pin 22 (GPIO25)-----------> RESET
Pin 18 (GPIO24)-----------> DC
Pin 19 (GPIO10)-----------> MOSI/SDI
Pin 23 (GPIO11)-----------> SCK/CLK
Pin 1  (3.3V)  -----------> LED/BL
```

MISO is optional for display-only operation.

Power the Pi back on.

---

# 13. Transfer `adam_tft.py` from Windows

The Python file is currently located on the Windows laptop at:

```text
D:\Downloads\adam_tft.py
```

## IMPORTANT

Run the `scp` command from **Windows PowerShell**, not from inside the Pi SSH shell.

First, if you are currently inside SSH:

```bash
exit
```

You should return to something like:

```text
PS C:\Users\HP>
```

Now run:

```powershell
scp "D:\Downloads\adam_tft.py" pi@adam-pi.local:/home/pi/adam_tft.py
```

Enter the Pi password when prompted.

---

# 14. Confirm the file is on the Pi

Reconnect:

```powershell
ssh pi@adam-pi.local
```

Then:

```bash
ls -l /home/pi/adam_tft.py
```

The file should be listed.

---

# 15. Test the TFT driver

Run:

```bash
python3 /home/pi/adam_tft.py
```

The ADAM TFT program should report messages similar to:

```text
[adam_tft] ILI9341 initialised OK
[adam_tft] Render thread started
[adam_tft] Cycling all emotions (5 s each) — Ctrl+C to stop
  -> idle
  -> speaking
  -> happy
  -> sad
  -> angry
  -> panic
  -> surprised
  -> shy
  -> sleep
  -> thinking
  -> reconnecting
  -> love
  -> confused
  -> rizz
```

A successful:

```text
[adam_tft] ILI9341 initialised OK
```

confirms that the TFT initialization succeeded.

To test one emotion:

```bash
python3 /home/pi/adam_tft.py happy
```

Other supported emotions from the ADAM TFT code/documentation:

```text
idle
speaking
happy
sad
angry
panic
surprised
shy
sleep
thinking
reconnecting
love
confused
rizz
```

---

# 16. Integrate the TFT into `adamV25.py`

In the main ADAM Python program:

## Import

```python
from adam_tft import TFTEmotionRenderer
```

## Create renderer

```python
tft = TFTEmotionRenderer()
```

## Startup

```python
tft.start()
tft.set_emotion("idle")
```

## Whenever ADAM changes emotion

For example:

```python
current_emotion = "happy"
tft.set_emotion("happy")
```

## Shutdown

```python
tft.stop()
```

---

# 17. ADAM emotion names

Use only the supported names:

```text
idle
speaking
happy
sad
angry
panic
surprised
shy
sleep
thinking
reconnecting
love
confused
rizz
```

---

# 18. Sound card / audio setup

## Important limitation

The supplied ADAM TFT setup document does **not** specify:

- sound-card model
- audio HAT/USB device model
- I2S pins
- DAC/ADC pins
- audio driver package
- microphone wiring
- speaker wiring
- required ALSA device name

Therefore, do **not** invent or copy random sound-card GPIO settings into the Pi.

Before configuring audio, identify the exact hardware model used by ADAM.

### Once the exact sound-card hardware is known, document:

```text
Sound card model:
Connection type: USB / I2S / HAT / other
Speaker output:
Microphone input:
Required packages:
Required overlays:
Required GPIO/I2S pins:
ALSA device:
Test command:
```

This guide intentionally leaves those values blank until the actual sound-card hardware is identified.

---

# 19. Check audio devices

After the sound hardware is physically connected, these commands can be used to inspect what the Pi sees:

```bash
aplay -l
```

For recording devices:

```bash
arecord -l
```

Do not modify `/boot` overlays or I2S configuration based only on these commands; use the exact sound-card documentation/model for the required configuration.

---

# 20. Run the complete ADAM program

Once the ADAM project files are copied to the Pi:

```bash
cd /home/pi/adam
```

Then run:

```bash
python3 adamV25.py
```

The exact required project directory and any additional Python packages depend on the rest of the ADAM codebase.

If the complete ADAM project is copied into `/home/pi/adam`, the expected basic workflow is:

```bash
cd /home/pi/adam
python3 adamV25.py
```

---

# 21. Optional: Run TFT automatically at boot

Only use this standalone TFT service if `adamV25.py` is **not** already responsible for starting the TFT renderer.

Create:

```bash
sudo nano /etc/systemd/system/adam-tft.service
```

Use:

```ini
[Unit]
Description=ADAM TFT Emotion Display
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/adam_tft.py idle
WorkingDirectory=/home/pi
StandardOutput=journal
StandardError=journal
Restart=on-failure
User=pi
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable adam-tft.service
sudo systemctl start adam-tft.service
```

Check status:

```bash
sudo systemctl status adam-tft.service
```

View logs:

```bash
sudo journalctl -u adam-tft.service -f
```

### Important

If `adamV25.py` starts the TFT itself using:

```python
tft.start()
```

then a separate TFT systemd service is not needed.

---

# 22. Troubleshooting

## `No module named 'luma'`

Run:

```bash
sudo pip3 install luma.lcd --break-system-packages
```

## Permission denied for `/dev/spidev0.0`

Run:

```bash
sudo usermod -aG spi pi
```

Then reboot and reconnect.

## TFT stays black

Check:

- VCC -> 3.3V
- GND -> GND
- DC -> GPIO24 / physical pin 18
- CS -> GPIO8 / physical pin 24
- RESET -> GPIO25 / physical pin 22
- MOSI -> GPIO10 / physical pin 19
- SCK -> GPIO11 / physical pin 23
- SPI is enabled

## TFT shows white/garbage

The source document recommends trying:

```text
spi_speed_hz=16_000_000
```

in `TFTEmotionRenderer()`.

## Animation is slow

The source document suggests reducing:

```text
FPS_TARGET = 20
```

in `adam_tft.py`.

## `ImportError: RPi.GPIO`

Run:

```bash
sudo apt install -y python3-rpi.gpio
```

## SSH does not connect

Check:

```powershell
ping adam-pi.local
```

Make sure the laptop and Pi are on the same Wi-Fi network.

---

# 23. Final ADAM readiness checklist

Before running ADAM, verify all of the following:

- [ ] Raspberry Pi Zero 2 W boots
- [ ] Pi is connected to the correct Wi-Fi
- [ ] `ssh pi@adam-pi.local` works
- [ ] SPI is enabled
- [ ] `/dev/spidev0.0` exists
- [ ] `/dev/spidev0.1` exists
- [ ] Python 3 is installed
- [ ] `python3-rpi.gpio` is installed
- [ ] `python3-spidev` is installed
- [ ] Pillow is installed
- [ ] `luma.lcd` is installed
- [ ] `pi` has SPI/GPIO permissions
- [ ] ILI9341 is wired to the documented pins
- [ ] `/home/pi/adam_tft.py` exists
- [ ] `python3 /home/pi/adam_tft.py` initializes the ILI9341 successfully
- [ ] ADAM project files are copied to the Pi
- [ ] `adamV25.py` is present
- [ ] Exact sound-card hardware is identified before configuring audio
- [ ] Audio device is detected with `aplay -l` / `arecord -l` when applicable

---

# 24. Useful commands

## Connect from Windows

```powershell
ssh pi@adam-pi.local
```

## Copy a file from Windows to Pi

```powershell
scp "D:\Downloads\FILE_NAME.py" pi@adam-pi.local:/home/pi/FILE_NAME.py
```

## Check SPI

```bash
ls /dev/spi*
```

## Check TFT file

```bash
ls -l /home/pi/adam_tft.py
```

## Run TFT test

```bash
python3 /home/pi/adam_tft.py
```

## Run one emotion

```bash
python3 /home/pi/adam_tft.py happy
```

## Check audio playback devices

```bash
aplay -l
```

## Check audio recording devices

```bash
arecord -l
```

## Reboot

```bash
sudo reboot
```

## Shut down safely

```bash
sudo poweroff
```

---

# Source

**ADAM v3 — ILI9341 TFT Direct on Pi Zero 2W — Complete Wiring + Upload + Setup Guide, Dgen Technologies Pvt. Ltd., May 2026.**

The TFT wiring, SPI setup, Python packages, `adam_tft.py` upload/test flow, ADAM emotion integration, systemd example, and TFT troubleshooting in this guide follow that source.
