# ADAM --- Raspberry Pi Zero 2 W Complete Setup & Bring-Up Guide

This is the step-by-step procedure for preparing the **Raspberry Pi Zero
2 W** to run ADAM reliably.

It covers:

-   Raspberry Pi OS and SSH
-   Python virtual environment
-   SPI / I2S / GPIO configuration
-   ILI9341 TFT
-   Pan servo on GPIO12
-   Two INMP441 microphones
-   MAX98357A I2S speaker amplifier
-   Google voiceHAT ALSA sound card
-   Correct audio-device selection
-   ESP32-CAM UART link
-   `.env` configuration
-   Hardware tests
-   Final ADAM launch
-   Troubleshooting

> **Important:** Do not change several things at once. Bring ADAM up in
> the order below and verify each subsystem before moving to the next
> one.

------------------------------------------------------------------------

# 1. Hardware Target

## Raspberry Pi

-   Raspberry Pi Zero 2 W
-   Raspberry Pi OS
-   40-pin GPIO header
-   Wi-Fi enabled
-   SSH enabled

## Audio

-   2 × INMP441 I2S microphones
-   1 × MAX98357A I2S amplifier
-   1 × 40 mm / 3 W speaker

## Motion

-   Pan servo
-   Servo signal: **GPIO12 / physical pin 32**
-   Servo power: **external 5 V supply**
-   Servo ground must be common with Pi ground

## Display

For the current ADAM Python setup that prints:

``` text
[adam_tft] ILI9341 initialised OK
```

the Pi-side `adam_tft.py` path is being used.

### Pi-side ILI9341 wiring

  TFT        Raspberry Pi
  ---------- -----------------
  MOSI       GPIO10 / pin 19
  SCLK/SCK   GPIO11 / pin 23
  CS/CE      GPIO8 / pin 24
  DC         GPIO24 / pin 18
  RST        GPIO25 / pin 22
  VCC        3.3 V / pin 17
  GND        GND / pin 20
  LED/BLK    3.3 V / pin 1

The older hardware guide specifies this Pi-side SPI wiring. The newer
ADAM v32 architecture can instead place the TFT on the Pico; **do not
mix the two architectures**. If your running code says
`[adam_tft] ILI9341 initialised OK`, use the Pi-side wiring above for
that codebase.

------------------------------------------------------------------------

# 2. Important I2S Bus Wiring

The I2S clock lines are shared.

### INMP441 microphones

  Signal       Raspberry Pi
  ------------ -----------------
  BCLK / SCK   GPIO18 / pin 12
  WS / LRC     GPIO19 / pin 35
  SD / DOUT    GPIO20 / pin 38
  VDD          3.3 V
  GND          GND

For stereo:

-   Mic 1 L/R → **GND** = LEFT
-   Mic 2 L/R → **3.3 V** = RIGHT

Both microphones share:

-   BCLK
-   WS
-   GND
-   3.3 V

Only their L/R selection differs.

### MAX98357A

  MAX98357A     Raspberry Pi
  ------------- -----------------
  BCLK          GPIO18 / pin 12
  LRC           GPIO19 / pin 35
  DIN           GPIO21 / pin 40
  VIN           External 5 V
  GND           Common GND
  SPK+ / SPK-   Speaker

### Critical I2S rule

GPIO18 and GPIO19 are shared by:

-   Mic 1
-   Mic 2
-   MAX98357A

This is expected.

GPIO20 is the microphone data input.

GPIO21 is the speaker amplifier data output.

The microphone and speaker therefore do **not** use the same data pin.

------------------------------------------------------------------------

# 3. Power and Ground

Use a proper common-ground arrangement.

## Servo

Do **not** power the servo from the Pi's 5 V rail.

Use:

``` text
External 5 V → Servo VCC
External GND → Servo GND
Pi GND       → External GND
Pi GPIO12    → Servo signal
```

## MAX98357A

Use the external 5 V rail where appropriate:

``` text
External 5 V → MAX98357A VIN
External GND → MAX98357A GND
Pi GND       → External GND
```

## INMP441

Power the microphones from **3.3 V**, not 5 V.

------------------------------------------------------------------------

# 4. Boot Configuration

Open:

``` bash
sudo nano /boot/firmware/config.txt
```

On older Raspberry Pi OS versions the file may instead be:

``` bash
sudo nano /boot/config.txt
```

Make sure these are enabled:

``` ini
dtparam=spi=on
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```

You already had:

``` ini
dtparam=spi=on
```

and:

``` ini
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```

at the bottom of the file. That is correct.

Do not add the same line repeatedly.

Save in nano:

``` text
Ctrl+O
Enter
Ctrl+X
```

Then reboot:

``` bash
sudo reboot
```

------------------------------------------------------------------------

# 5. Verify the Sound Card After Reboot

Run:

``` bash
aplay -l
```

and:

``` bash
arecord -l
```

The Google voiceHAT card should appear.

Expected capture device:

``` text
card 1: sndrpigooglevoi [snd_rpi_googlevoicehat_soundcar]
device 0: Google voiceHAT SoundCard HiFi
```

The important part is:

``` text
sndrpigooglevoi
```

and:

``` text
card 1
device 0
```

Your HDMI device can remain present. It does not mean ADAM is using
HDMI.

------------------------------------------------------------------------

# 6. Do NOT Assume `hw:0,0`

This is extremely important for the current ADAM Pi.

Your system has shown:

``` text
card 1: sndrpigooglevoi
```

while HDMI is another card.

Therefore:

``` text
hw:0,0
```

may point to HDMI.

For your current machine, the Google voiceHAT device is:

``` text
hw:1,0
```

or preferably:

``` text
plughw:sndrpigooglevoi,0
```

Use the Google voiceHAT device explicitly.

------------------------------------------------------------------------

# 7. Recommended ADAM Audio Configuration

In the ADAM Python configuration, use:

``` python
CAPTURE_DEVICE   = "plughw:sndrpigooglevoi,0"
CAPTURE_FORMAT   = "S32_LE"
CAPTURE_RATE     = 48000
CAPTURE_CHANNELS = 2

PLAYBACK_DEVICE   = "plughw:sndrpigooglevoi,0"
PLAYBACK_FORMAT   = "S16_LE"
PLAYBACK_RATE     = 48000
PLAYBACK_CHANNELS = 2
```

If your ALSA installation does not accept the named form, use:

``` python
CAPTURE_DEVICE   = "plughw:1,0"
PLAYBACK_DEVICE  = "plughw:1,0"
```

Do **not** leave the ADAM code at:

``` python
plughw:0,0
```

if card 0 is HDMI.

The ADAM source currently contains:

``` python
CAPTURE_DEVICE = "plughw:0,0"
PLAYBACK_DEVICE = "plughw:0,0"
```

so this must be checked against the actual card numbering on the Pi.

------------------------------------------------------------------------

# 8. Check ALSA Playback Devices

Run:

``` bash
aplay -L
```

You should see entries similar to:

``` text
hw:CARD=sndrpigooglevoi,DEV=0
plughw:CARD=sndrpigooglevoi,DEV=0
default:CARD=sndrpigooglevoi
```

The Google voiceHAT entries are the ones ADAM should use.

HDMI entries such as:

``` text
vc4hdmi
hdmi:CARD=vc4hdmi
```

are not the ADAM speaker.

------------------------------------------------------------------------

# 9. Test Speaker BEFORE Connecting Microphones

This is the recommended bring-up order.

The ADAM component reference explicitly recommends testing the
DAC/speaker first and then wiring/testing the microphones.

Run:

``` bash
speaker-test -D plughw:sndrpigooglevoi,0 -c 2 -r 48000 -F S32_LE
```

If the named device does not work, try:

``` bash
speaker-test -D hw:1,0 -c 2 -r 48000 -F S32_LE
```

You should hear the test noise.

Because the amplifier is physically one speaker, hearing alternating
LEFT/RIGHT test channels does not mean you have two physical speakers.
It is a stereo I2S stream being sent to a single amplifier/speaker.

Stop with:

``` text
Ctrl+C
```

------------------------------------------------------------------------

# 10. Test Microphone Hardware

First verify the capture card:

``` bash
arecord -l
```

Then test the stereo microphones:

``` bash
arecord -D plughw:sndrpigooglevoi,0 -f S32_LE -r 48000 -c 2 -d 5 mic_test.wav
```

Play it back:

``` bash
aplay -D plughw:sndrpigooglevoi,0 mic_test.wav
```

If stereo recording works, test the two microphones separately.

Speak near the LEFT microphone.

Then speak near the RIGHT microphone.

The channels should respond differently.

------------------------------------------------------------------------

# 11. If `arecord` Says "Channels Count Non Available"

For example:

``` text
arecord: set_params:1398: Channels count non available
```

do not randomly change the Pi configuration.

First inspect the actual hardware:

``` bash
arecord -l
```

Then inspect supported formats:

``` bash
arecord --dump-hw-params -D hw:1,0
```

The Google voiceHAT card may expose a fixed channel count.

For the current ADAM stereo setup, the expected capture configuration
is:

``` text
S32_LE
48000 Hz
2 channels
```

Use `plughw` when you want ALSA to perform compatible format conversion.

------------------------------------------------------------------------

# 12. Microphone Volume Is NOT an ALSA Mixer Setting Here

If:

``` bash
alsamixer
```

shows:

``` text
This sound device does not have any controls.
```

that is not automatically a fault.

The I2S sound card can expose no traditional mixer controls.

Do not assume:

``` text
no mixer controls = microphone broken
```

The correct test is actual capture:

``` bash
arecord ...
```

and then inspect whether the recorded samples change when you speak.

------------------------------------------------------------------------

# 13. Python Audio Device Check

Inside the ADAM virtual environment:

``` bash
cd ~/adam
source venv/bin/activate
```

Start Python:

``` bash
python
```

Then:

``` python
import pyaudio

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    print(i, p.get_device_info_by_index(i))
```

A working Google voiceHAT device should show something similar to:

``` text
name: snd_rpi_googlevoicehat_soundcar
maxInputChannels: 2
maxOutputChannels: 2
defaultSampleRate: 48000.0
```

This confirms that Python sees both:

-   2 input channels
-   2 output channels

Exit:

``` python
exit()
```

------------------------------------------------------------------------

# 14. Python Dependencies

Go to the ADAM directory:

``` bash
cd ~/adam
```

Activate the environment:

``` bash
source venv/bin/activate
```

Confirm:

``` bash
which python
python --version
```

It should point into:

``` text
~/adam/venv/
```

Install/update the required packages:

``` bash
pip install --upgrade google-genai pyaudio python-dotenv websockets
pip install pyserial numpy requests zeroconf --break-system-packages
```

Optional:

``` bash
pip install webrtcvad vosk
pip install ddgs
```

------------------------------------------------------------------------

# 15. ADAM `.env` File

Inside:

``` text
~/adam
```

create:

``` bash
nano .env
```

Put:

``` env
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
ESP32_IP="192.168.1.13"
```

If using the wired UART version, also check whether these are required
by your code:

``` env
PI_UART_PORT="/dev/serial0"
PI_UART_BAUD="921600"
```

Save:

``` text
Ctrl+O
Enter
Ctrl+X
```

Never publish the `.env` file or commit it to Git.

------------------------------------------------------------------------

# 16. ESP32-CAM UART Configuration

The current ADAM code uses:

``` text
/dev/serial0
```

at:

``` text
921600 baud
```

The code expects the serial interface to be available.

Add the user to dialout:

``` bash
sudo usermod -a -G dialout pi
```

Check:

``` bash
groups
```

You should eventually see:

``` text
dialout
```

Check the serial device:

``` bash
ls -l /dev/serial0
```

If the serial login shell is using the port, disable it:

``` bash
sudo systemctl disable --now serial-getty@ttyAMA0.service
```

Then reboot:

``` bash
sudo reboot
```

After reboot:

``` bash
ls -l /dev/serial0
```

The ADAM program should be able to open it.

------------------------------------------------------------------------

# 17. Servo Configuration

ADAM pan servo:

``` text
GPIO12
Physical pin 32
```

The current code defines:

``` python
NECK_GPIO_PIN = 12
```

Connect:

``` text
Servo signal → GPIO12 / pin 32
Servo VCC    → external 5 V
Servo GND    → common GND
```

The GPIO Zero warning:

``` text
PWMSoftwareFallback
```

means GPIO Zero is using software PWM instead of pigpio.

It is a warning, not necessarily a failure.

If servo jitter becomes a problem, configure the pigpio pin factory.

------------------------------------------------------------------------

# 18. TFT SPI Configuration

SPI must be enabled:

``` ini
dtparam=spi=on
```

Verify:

``` bash
ls /dev/spidev*
```

You should normally see something such as:

``` text
/dev/spidev0.0
/dev/spidev0.1
```

For the Pi-side ILI9341 setup:

``` text
MOSI → GPIO10
SCLK → GPIO11
CS   → GPIO8
DC   → GPIO24
RST  → GPIO25
```

The current program should print:

``` text
[adam_tft] ILI9341 initialised OK
```

when initialization succeeds.

------------------------------------------------------------------------

# 19. Check the ADAM Folder

After SSH login:

``` bash
cd ~/adam
```

Show the files:

``` bash
ls -lah
```

Useful expected files include:

``` text
adam_main_wifi.py
adam_tft.py
.env
system_prompt.txt
venv/
```

If your main program has a different filename, use that exact filename.

------------------------------------------------------------------------

# 20. Check the Virtual Environment

From:

``` text
~/adam
```

run:

``` bash
source venv/bin/activate
```

Your prompt should become:

``` text
(venv) pi@adam-pi:~/adam $
```

Check:

``` bash
which python
```

Expected:

``` text
/home/pi/adam/venv/bin/python
```

------------------------------------------------------------------------

# 21. Test Order --- ALWAYS Use This Order

Do not immediately run the complete ADAM program after changing
hardware.

Use:

## Test 1 --- SPI

``` bash
ls /dev/spidev*
```

## Test 2 --- Speaker

``` bash
speaker-test -D plughw:sndrpigooglevoi,0 -c 2 -r 48000 -F S32_LE
```

Verify actual sound.

## Test 3 --- Microphones

``` bash
arecord -D plughw:sndrpigooglevoi,0 -f S32_LE -r 48000 -c 2 -d 5 mic_test.wav
```

Verify that the recorded audio changes when speaking.

## Test 4 --- Python sees audio

``` bash
python
```

then:

``` python
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(i, p.get_device_info_by_index(i))
```

Verify:

``` text
maxInputChannels: 2
maxOutputChannels: 2
```

## Test 5 --- Servo

Run the servo test and verify GPIO12 moves the pan servo.

## Test 6 --- TFT

Verify:

``` text
[adam_tft] ILI9341 initialised OK
```

## Test 7 --- ESP32 UART

Verify:

``` text
/dev/serial0
```

and the ESP32 sends data.

## Test 8 --- Complete ADAM

Only after all individual tests pass.

------------------------------------------------------------------------

# 22. Run ADAM

From:

``` bash
cd ~/adam
```

activate the environment:

``` bash
source venv/bin/activate
```

Then:

``` bash
python adam_main_wifi.py
```

A healthy startup should show things similar to:

``` text
Environment loaded
Connecting to ESP32
ILI9341 initialised OK
Pan Servo initialized on GPIO 12
...
```

The exact startup text depends on the current ADAM code version.

------------------------------------------------------------------------

# 23. IMPORTANT: Audio Device Must Match the Code

Before running ADAM, inspect the configuration inside the main Python
file.

Search:

``` bash
grep -n "CAPTURE_DEVICE\|PLAYBACK_DEVICE\|CAPTURE_RATE\|PLAYBACK_RATE\|CAPTURE_CHANNELS\|PLAYBACK_CHANNELS" adam_main_wifi.py
```

If the program uses the older values:

``` python
CAPTURE_DEVICE = "plughw:0,0"
PLAYBACK_DEVICE = "plughw:0,0"
```

and your actual Google voiceHAT card is:

``` text
card 1
```

change them to:

``` python
CAPTURE_DEVICE = "plughw:sndrpigooglevoi,0"
PLAYBACK_DEVICE = "plughw:sndrpigooglevoi,0"
```

This is one of the most important ADAM audio checks.

------------------------------------------------------------------------

# 24. Why HDMI Can Appear Even When ADAM Uses the I2S Speaker

It is normal for:

``` bash
aplay -L
```

to show:

``` text
vc4hdmi
```

and:

``` text
sndrpigooglevoi
```

at the same time.

The Raspberry Pi has more than one audio device.

ADAM should explicitly select:

``` text
sndrpigooglevoi
```

for the I2S microphone/amplifier.

Do not rely on the system default if you need predictable robot hardware
behavior.

------------------------------------------------------------------------

# 25. Why Stereo Is Used With One Speaker

The microphones are stereo because ADAM needs:

``` text
LEFT microphone
RIGHT microphone
```

for sound-direction / DoA processing.

The MAX98357A output is also configured as a stereo stream by ALSA, but
the physical robot has only one speaker.

That is acceptable.

The important distinction is:

``` text
2-channel microphone INPUT
        ↓
ADAM processing
        ↓
2-channel audio OUTPUT stream
        ↓
MAX98357A
        ↓
one physical speaker
```

------------------------------------------------------------------------

# 26. I2S Bring-Up Rule

The recommended assembly order is:

``` text
1. Configure I2S
2. Connect/test MAX98357A
3. Confirm speaker works
4. Connect INMP441 microphones
5. Confirm stereo capture
6. Run ADAM audio
```

Do not troubleshoot microphones and speaker simultaneously if you can
avoid it.

The same I2S clock lines are shared, so a bad connection on
GPIO18/GPIO19 can affect the entire audio bus.

------------------------------------------------------------------------

# 27. If Speaker Works but Microphone Shows Zero

Check in this exact order:

### A. Confirm card

``` bash
arecord -l
```

### B. Confirm Python sees input

``` bash
python
```

``` python
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(i, p.get_device_info_by_index(i))
```

### C. Direct ALSA capture

``` bash
arecord -D plughw:sndrpigooglevoi,0 -f S32_LE -r 48000 -c 2 -d 5 mic_test.wav
```

### D. Inspect the recording

``` bash
aplay -D plughw:sndrpigooglevoi,0 mic_test.wav
```

### E. Verify microphone wiring

``` text
BCLK → GPIO18
WS   → GPIO19
SD   → GPIO20
VDD  → 3.3V
GND  → GND
```

### F. Verify L/R

``` text
Left mic  L/R → GND
Right mic L/R → 3.3V
```

------------------------------------------------------------------------

# 28. If Speaker Does Not Work

Check:

``` bash
aplay -l
```

Then:

``` bash
speaker-test -D plughw:sndrpigooglevoi,0 -c 2 -r 48000 -F S32_LE
```

If that fails, try:

``` bash
speaker-test -D hw:1,0 -c 2 -r 48000 -F S32_LE
```

Then check:

``` bash
aplay -L
```

Do not immediately change the boot configuration if the card already
appears correctly.

------------------------------------------------------------------------

# 29. If `alsamixer` Says "No Controls"

This can be normal for this I2S device.

The message:

``` text
This sound device does not have any controls.
```

does not by itself mean the Google voiceHAT sound card is broken.

Use actual capture/playback tests instead.

------------------------------------------------------------------------

# 30. If `raspi-config` Audio Is Missing

On newer Raspberry Pi OS versions, the old:

``` text
System Options → Audio
```

selection may not be available or may report:

``` text
raspi-config cannot configure audio when PulseAudio or PipeWire are in use.
```

That is not the method to use for selecting ADAM's I2S card.

For ADAM, explicitly select:

``` text
sndrpigooglevoi
```

in ALSA/Python.

The boot overlay is what creates the Google voiceHAT card:

``` ini
dtoverlay=googlevoicehat-soundcard
```

------------------------------------------------------------------------

# 31. Final Configuration Checklist

Before declaring the Pi ready:

-   [ ] Raspberry Pi Zero 2 W boots normally
-   [ ] Wi-Fi works
-   [ ] SSH works
-   [ ] `~/adam` exists
-   [ ] `venv` works
-   [ ] `.env` exists
-   [ ] Gemini API key is valid
-   [ ] ESP32 IP is correct
-   [ ] SPI is enabled
-   [ ] I2S is enabled
-   [ ] `googlevoicehat-soundcard` overlay is enabled
-   [ ] `aplay -l` shows Google voiceHAT
-   [ ] `arecord -l` shows Google voiceHAT
-   [ ] ADAM uses the Google voiceHAT device, not HDMI
-   [ ] Speaker test works
-   [ ] Microphone capture works
-   [ ] Stereo microphones produce LEFT/RIGHT data
-   [ ] Servo works on GPIO12
-   [ ] TFT initializes
-   [ ] `/dev/serial0` exists
-   [ ] `pi` has `dialout`
-   [ ] ESP32-CAM UART works
-   [ ] ADAM starts without fatal errors

------------------------------------------------------------------------

# 32. Known Good Current Audio Values

For the current Pi configuration where the Google voiceHAT card is:

``` text
card 1
device 0
```

use:

``` python
CAPTURE_DEVICE   = "plughw:sndrpigooglevoi,0"
CAPTURE_FORMAT   = "S32_LE"
CAPTURE_RATE     = 48000
CAPTURE_CHANNELS = 2

PLAYBACK_DEVICE   = "plughw:sndrpigooglevoi,0"
PLAYBACK_FORMAT   = "S16_LE"
PLAYBACK_RATE     = 48000
PLAYBACK_CHANNELS = 2
```

The critical point is that **`0,0` is not automatically the correct
device**. Always verify the card number/name after boot.

------------------------------------------------------------------------

# 33. Final Launch Command

Every time you connect over SSH:

``` bash
cd ~/adam
source venv/bin/activate
python adam_main_wifi.py
```

If the program is stopped with `Ctrl+C`, it can be started again using
the same commands.

------------------------------------------------------------------------

# 34. Safe Shutdown

Do not simply remove Pi power while ADAM is running.

Stop the program:

``` text
Ctrl+C
```

Wait for it to exit cleanly.

Then shut down:

``` bash
sudo shutdown -h now
```

This is especially important because ADAM may write memory/conversation
files to the SD card.

------------------------------------------------------------------------

# 35. Quick Diagnostic Command Block

If something stops working, run this complete block:

``` bash
cd ~/adam
source venv/bin/activate

echo "=== SYSTEM ==="
uname -a

echo "=== SPI ==="
ls -l /dev/spidev* 2>/dev/null

echo "=== AUDIO PLAYBACK ==="
aplay -l

echo "=== AUDIO CAPTURE ==="
arecord -l

echo "=== ALSA DEVICES ==="
aplay -L | grep -i -E "google|voicehat|hdmi"

echo "=== SERIAL ==="
ls -l /dev/serial0

echo "=== GROUPS ==="
groups

echo "=== ADAM AUDIO CONFIG ==="
grep -n "CAPTURE_DEVICE\|PLAYBACK_DEVICE\|CAPTURE_RATE\|PLAYBACK_RATE\|CAPTURE_CHANNELS\|PLAYBACK_CHANNELS" adam_main_wifi.py
```

Then test the speaker:

``` bash
speaker-test -D plughw:sndrpigooglevoi,0 -c 2 -r 48000 -F S32_LE
```

Then test the microphones:

``` bash
arecord -D plughw:sndrpigooglevoi,0 -f S32_LE -r 48000 -c 2 -d 5 mic_test.wav
```

------------------------------------------------------------------------

# 36. The Golden Rule for ADAM Bring-Up

Do not debug everything through `adam_main_wifi.py`.

Use this sequence:

``` text
                    RASPBERRY PI
                         │
              ┌──────────┴──────────┐
              │                     │
             SPI                   I2S
              │                     │
             TFT              ┌─────┴─────┐
                              │           │
                            MICS       MAX98357A
                              │           │
                           INPUT       SPEAKER
                              │
                         Test separately
                              │
                         Then run ADAM
                              │
                       ESP32 UART/Wi-Fi
                              │
                         Full ADAM system
```

If the individual hardware tests pass but ADAM still does not respond,
then troubleshoot the **ADAM software/audio pipeline**, rather than
rewiring the hardware.

------------------------------------------------------------------------

## Current ADAM-specific warning

The current source configuration found in the ADAM code uses:

``` python
CAPTURE_DEVICE = "plughw:0,0"
PLAYBACK_DEVICE = "plughw:0,0"
```

but your Pi has shown the Google voiceHAT as **card 1**, while HDMI is
also present.

Therefore, before the final ADAM run, make the audio device explicit:

``` python
CAPTURE_DEVICE = "plughw:sndrpigooglevoi,0"
PLAYBACK_DEVICE = "plughw:sndrpigooglevoi,0"
```

Then test the speaker and microphone directly with ALSA before starting
ADAM.

This avoids accidentally sending ADAM's audio to HDMI or trying to
capture from the wrong card.
