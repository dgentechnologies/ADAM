"""
esp32_link.py — ADAM v40 wired UART link to the ESP32-CAM (Flow 2)
==============================================================================
Single serial link that carries EVERYTHING between the Pi and the ESP32-CAM
over /dev/serial0 @ 921600:

  RECEIVES (framed, tag byte + payload):
    • TAG_FRAME   'F' + <uint32 len LE> + JPEG bytes   → frame_q
    • TAG_TOUCH   'T' + 4 bytes (0/1 each)             → touch_q
    • TAG_GESTURE 'G' + 1 byte (0-3)                   → gesture_q
  SENDS (newline-terminated text lines):
    • "EMO:<emotion>"  → relayed by ESP32 to the Pico TFT face
    • "TILT:<deg>"     → relayed by ESP32 to the tilt servo
    • "CAM:ON/OFF"     → camera duty-cycling

Design points that matter:
  • Reads run on their own daemon thread with a byte-accurate resync state
    machine (validates JPEG FFD8/FFD9 and a sane frame length) so a desync
    can't spin at 100% CPU and starve the audio pipeline.
  • Writes run on a SEPARATE daemon thread fed by a queue — send_line() only
    enqueues and returns instantly, so a blocking pyserial write can never
    stall the asyncio event loop (that stall used to truncate the start of
    user speech right when the camera toggled).

Wire protocol tag/baud constants come from config.py.
"""

import time
import struct
import threading
import queue as sync_queue

import serial

from config import PI_UART_PORT, PI_UART_BAUD, TAG_FRAME, TAG_TOUCH, TAG_GESTURE


class ESP32Link:
    def __init__(self, port: str, baud: int):
        self.port = port
        self.baud = baud
        self._ser: serial.Serial | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self.frame_q: sync_queue.Queue = sync_queue.Queue(maxsize=2)
        self.gesture_q: sync_queue.Queue = sync_queue.Queue(maxsize=20)
        self.touch_q: sync_queue.Queue = sync_queue.Queue(maxsize=20)

        # ── Background write queue ──────────────────────────────────────
        # FIX: send_line() used to call self._ser.write() directly, which
        # is a BLOCKING pyserial call. Called from an async coroutine (as
        # it was, e.g. from the camera() task on every CAM:ON/CAM:OFF
        # transition), this stalls the entire asyncio event loop for
        # however long the OS write takes — on the same 921600-baud UART
        # that's also carrying continuous frame/touch/gesture reads, that
        # was long enough to cause listen()'s mic read to miss a beat
        # right as a user started talking, truncating the start of their
        # sentence (observed correlating almost 1:1 with camera on/off
        # log lines). A tiny background thread with its own queue means
        # every caller — sync or async — just enqueues instantly and the
        # actual blocking write happens off the event loop, always.
        self._write_q: sync_queue.Queue = sync_queue.Queue(maxsize=64)
        self._write_thread: threading.Thread | None = None

        self._connected = False
        self._ever_received_data = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def receiving_data(self) -> bool:
        return self._ever_received_data

    def start(self) -> None:
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=1.0)
            self._connected = True
            print(f"  ✅  UART port opened ({self.port} @ {self.baud}) — "
                  f"waiting to confirm ESP32-CAM is actually wired/powered...")
        except PermissionError as e:
            self._connected = False
            print(f"  ❌  Permission denied opening {self.port}: {e}")
            print("      Fix on the Pi:")
            print("        sudo usermod -a -G dialout pi")
            print("        sudo raspi-config → Interface Options → Serial Port")
            print("          login shell over serial → No")
            print("          serial port hardware enabled → Yes")
            print("        sudo systemctl disable --now serial-getty@ttyAMA0.service")
            print("        sudo reboot")
            print("      ADAM will run WITHOUT vision/touch (audio-only mode) until fixed.")
            return
        except Exception as e:
            self._connected = False
            print(f"  ⚠️  Could not open {self.port}: {e}")
            print("      ADAM will run WITHOUT vision/touch (audio-only mode). "
                  "Check wiring + raspi-config serial settings "
                  "(login shell over serial must be OFF).")
            return

        self._thread = threading.Thread(target=self._read_loop, daemon=True,
                                        name="esp32-uart-reader")
        self._thread.start()

        self._write_thread = threading.Thread(target=self._write_loop, daemon=True,
                                               name="esp32-uart-writer")
        self._write_thread.start()

        def _watch():
            time.sleep(10.0)
            if self._connected and not self._ever_received_data:
                print(f"  ⚠️  UART port is open but no data received from ESP32-CAM "
                      f"in 10s — running WITHOUT vision/touch (audio-only mode). "
                      f"This means the port opened OK but nothing is arriving: "
                      f"check ESP32-CAM is powered, TX/RX aren't swapped, and "
                      f"baud rate matches ({PI_UART_BAUD}).")
        threading.Thread(target=_watch, daemon=True, name="esp32-uart-watchdog").start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._write_thread:
            # Wake the writer loop immediately instead of waiting for its
            # queue timeout, so shutdown isn't needlessly slow.
            try:
                self._write_q.put_nowait(None)
            except sync_queue.Full:
                pass
            self._write_thread.join(timeout=2.0)
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass

    def _write_loop(self) -> None:
        """Runs on its own dedicated thread. All actual blocking pyserial
        writes happen here — send_line() just enqueues and returns
        immediately, so it's safe to call from anywhere (async coroutine,
        sync helper, doesn't matter) without ever stalling the event loop."""
        while not self._stop.is_set():
            try:
                text = self._write_q.get(timeout=0.5)
            except sync_queue.Empty:
                continue
            if text is None:  # shutdown sentinel
                break
            if not self._connected or not self._ser:
                continue
            try:
                self._ser.write((text.strip() + "\n").encode("utf-8"))
            except Exception as e:
                print(f"  ⚠️  UART write failed: {e}")

    def send_line(self, text: str) -> None:
        # Non-blocking — just enqueues for the dedicated writer thread.
        # Safe to call from any context (async coroutine or sync code)
        # without risk of stalling the asyncio event loop. See
        # _write_loop() for where the actual blocking pyserial write
        # happens.
        if not self._connected:
            return
        try:
            self._write_q.put_nowait(text)
        except sync_queue.Full:
            # Queue backed up (shouldn't normally happen — writer thread
            # keeps up easily with our command rate) — drop the oldest
            # pending command rather than blocking the caller.
            try:
                self._write_q.get_nowait()
            except sync_queue.Empty:
                pass
            try:
                self._write_q.put_nowait(text)
            except sync_queue.Full:
                pass

    def _read_exact(self, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n and not self._stop.is_set():
            chunk = self._ser.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf) if len(buf) == n else None

    def _read_loop(self) -> None:
        # ── FIX #1: proper resync instead of trust-next-byte-blindly ────────
        # The previous version read a tag byte, then on a bad/garbage length
        # just printed a warning and `continue`d — which re-reads a FRESH
        # byte from the top of the loop that could ITSELF be mid-JPEG noise.
        # That never actually re-establishes framing sync; it just spins on
        # noise indefinitely (this is exactly the endless
        # "Suspicious frame length ..." storm you were seeing), burning CPU
        # the audio pipeline needs and starving arecord's reads, which is
        # what produced corrupted audio -> Gemini 1007 errors downstream.
        #
        # This version:
        #   1. Only accepts a byte as a tag, then validates what follows.
        #   2. On a bad length, does NOT jump — it naturally continues the
        #      same byte-by-byte read(1) loop, which is the correct way to
        #      hunt for the next real tag byte mid-stream.
        #   3. Validates actual JPEG SOI/EOI markers (FFD8...FFD9) before
        #      ever trusting a "plausible" length, catching the case where a
        #      garbage length happens to fall in a believable range.
        #   4. Rate-limits its own warning prints so a desync storm doesn't
        #      become its own CPU/IO cost.
        last_warn_t = 0.0
        warn_count = 0

        def warn_resync(msg: str) -> None:
            nonlocal last_warn_t, warn_count
            warn_count += 1
            now = time.time()
            if now - last_warn_t > 2.0:
                print(f"  ⚠️  UART resync: {msg} ({warn_count} since last report)")
                last_warn_t = now
                warn_count = 0

        while not self._stop.is_set():
            try:
                tag_byte = self._ser.read(1)
                if not tag_byte:
                    # No data available right now — normal idle time between
                    # frames (1 FPS camera => long gaps). Sleep briefly so
                    # this thread doesn't spin at 100% CPU on empty reads,
                    # which was starving mic/audio threads on the 2-core Pi.
                    time.sleep(0.01)
                    continue

                tag = tag_byte[0]

                if tag == TAG_FRAME:
                    len_bytes = self._read_exact(4)
                    if len_bytes is None:
                        continue
                    (frame_len,) = struct.unpack("<I", len_bytes)
                    if frame_len == 0 or frame_len > 200_000:
                        # NOT a real frame tag — a stray byte from inside
                        # another frame's JPEG data that happened to match
                        # 'F'. Do NOT restart from a fresh read(1) trusting
                        # the very next byte either — just fall through to
                        # the top of the while loop, which reads ONE byte
                        # at a time until it finds a genuinely valid tag.
                        warn_resync(f"garbage frame length {frame_len}, "
                                    f"scanning for next valid tag")
                        self._ever_received_data = True
                        continue

                    jpeg = self._read_exact(frame_len)
                    if jpeg is None:
                        continue
                    # Sanity-check real JPEG framing before trusting this as
                    # a good frame — catches cases where the length happened
                    # to look plausible (e.g. 4000-80000) but the bytes
                    # weren't actually a frame boundary.
                    if not (jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9"):
                        warn_resync(f"frame length {frame_len} looked "
                                    f"plausible but JPEG markers didn't "
                                    f"match — discarding")
                        continue
                    self._ever_received_data = True
                    try:
                        self.frame_q.put_nowait(jpeg)
                    except sync_queue.Full:
                        try:
                            self.frame_q.get_nowait()
                        except sync_queue.Empty:
                            pass
                        self.frame_q.put_nowait(jpeg)

                elif tag == TAG_TOUCH:
                    payload = self._read_exact(4)
                    if payload is None:
                        continue
                    if any(b not in (0, 1) for b in payload):
                        warn_resync("garbage touch payload, ignoring")
                        continue
                    self._ever_received_data = True
                    try:
                        self.touch_q.put_nowait(list(payload))
                    except sync_queue.Full:
                        pass

                elif tag == TAG_GESTURE:
                    payload = self._read_exact(1)
                    if payload is None:
                        continue
                    if payload[0] > 3:
                        warn_resync("garbage gesture code, ignoring")
                        continue
                    self._ever_received_data = True
                    try:
                        self.gesture_q.put_nowait(payload[0])
                    except sync_queue.Full:
                        pass

                # else: byte didn't match any known tag — expected noise
                # while resyncing after a garbage frame. Loop back and read
                # the next byte; no sleep needed since data IS actively
                # arriving (differs from the "no data at all" idle case
                # handled above).

            except Exception as e:
                print(f"  ⚠️  UART reader error: {e}")
                time.sleep(0.5)


esp_link = ESP32Link(PI_UART_PORT, PI_UART_BAUD)
