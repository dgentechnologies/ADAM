"""
ADAM v32 — Sprite-based Face Player  (Pico side)
=====================================================
Plays back pre-rendered animation frames instead of computing shapes live.
Frames are generated on a PC by generate_frames.py, RLE-compressed, and
stored as .bin files on the Pico's flash filesystem.

Why this exists: the live-math renderer (arcs/trig recomputed every frame,
possibly 2-4x per frame for band-safe RAM) was CPU-bound and choppy, on top
of an earlier RAM ceiling issue. This version moves all the expensive work
to your PC, once, offline. The Pico's runtime job is just: read compressed
bytes from flash -> decode into a small buffer -> push to screen. That's
drastically cheaper than live trig, so frame rate should be smooth even
with many emotions added later, since adding an emotion only adds a
flash file, not more Pico-side CPU cost per frame.

Pinout: unchanged from your existing files (see ADAM v32 Blueprint §D).

USAGE:
  1. Run generate_frames.py on your PC for each emotion you want.
  2. Copy the resulting <emotion>.bin files onto the Pico's flash
     filesystem (Thonny file browser drag-drop, or `mpremote cp`).
  3. Run this file on the Pico. TESTING_MODE cycles through every .bin
     file found in flash; LIVE MODE picks based on UART commands same
     as before (command name must match a "<name>.bin" file present).
"""

import machine, time, gc, struct, os

W, H = 320, 240
TESTING_MODE = True

# Band height for the small scratch buffer we decode into and push to the
# display. Same RAM-safety reasoning as the live renderer: this board
# can't allocate ~150KB contiguously, so we work in horizontal strips.
# Sprite playback is far cheaper per band than live math was, so more
# bands per frame is not the bottleneck it used to be -- but keep the
# buffer modest anyway to leave headroom for UART/other logic.
BAND_H = 60
assert H % BAND_H == 0
N_BANDS = H // BAND_H
BAND_PIXELS = W * BAND_H

gc.collect()
_band_buf = bytearray(W * BAND_H * 2)


class ST7789:
    def __init__(self):
        self._spi = machine.SPI(0,
            baudrate=40_000_000,
            polarity=1, phase=1,
            sck=machine.Pin(18),
            mosi=machine.Pin(19))
        self._cs  = machine.Pin(17, machine.Pin.OUT)
        self._dc  = machine.Pin(16, machine.Pin.OUT)
        self._rst = machine.Pin(20, machine.Pin.OUT)
        self._reset()
        self._init()

    def _reset(self):
        self._rst(1); time.sleep_ms(50)
        self._rst(0); time.sleep_ms(50)
        self._rst(1); time.sleep_ms(50)

    def _cmd(self, c):
        self._dc(0); self._cs(0)
        self._spi.write(bytearray([c]))
        self._cs(1)

    def _dat(self, d):
        self._dc(1); self._cs(0)
        self._spi.write(bytearray(d))
        self._cs(1)

    def _init(self):
        self._cmd(0x11); time.sleep_ms(120)
        self._cmd(0x36); self._dat([0x60])
        self._cmd(0x3A); self._dat([0x55])
        self._cmd(0x20)
        self._cmd(0x13)
        self._cmd(0x29); time.sleep_ms(50)

    def show(self):
        self._cmd(0x2A); self._dat([0x00,0x00,(W-1)>>8,(W-1)&0xFF])
        self._cmd(0x2B); self._dat([0x00,0x00,(H-1)>>8,(H-1)&0xFF])
        self._cmd(0x2C)

    def show_band(self, band_buf):
        self._dc(1); self._cs(0)
        self._spi.write(band_buf)
        self._cs(1)


class SpriteClip:
    """Loads and plays one emotion's .bin file. Keeps the file handle
    open and seeks to each frame's data on demand rather than loading
    the whole clip into RAM -- flash reads are fast enough that this
    is not a bottleneck, and it means clip size is bounded only by
    flash space, not RAM."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        header = self.f.read(13)
        magic, version, w, h, n_frames, delay_ms = struct.unpack("<4sBHHHH", header)
        if magic != b"ADFB":
            raise ValueError("Bad sprite file magic: %r" % magic)
        if w != W or h != H:
            raise ValueError("Sprite file dims %dx%d don't match display %dx%d" % (w, h, W, H))
        self.n_frames = n_frames
        self.delay_ms = delay_ms

        # Read the frame index table: n_frames * (offset:u32, length:u32)
        index_bytes = self.f.read(8 * n_frames)
        self.frame_offsets = []
        self.frame_lengths = []
        for i in range(n_frames):
            off, length = struct.unpack_from("<II", index_bytes, i * 8)
            self.frame_offsets.append(off)
            self.frame_lengths.append(length)

        # Data section starts right after the index table
        self._data_start = self.f.tell()

    def decode_band(self, frame_idx, band_index, out_buf):
        """Decodes ONLY the pixels belonging to `band_index` from the
        given frame's RLE stream into out_buf (a bytearray sized for
        one band, W*BAND_H*2 bytes). Streams the RLE pairs from flash
        without ever materializing the full 153.6KB decompressed frame
        in RAM -- we track a running pixel-position counter and only
        write bytes that fall within [band_start_px, band_end_px).
        """
        band_start_px = band_index * BAND_PIXELS
        band_end_px = band_start_px + BAND_PIXELS

        self.f.seek(self._data_start + self.frame_offsets[frame_idx])
        remaining = self.frame_lengths[frame_idx]

        pos_px = 0          # running position in the FULL frame, in pixels
        out_off = 0          # write cursor into out_buf, in bytes
        chunk_size = 512      # read RLE pairs in chunks to limit call overhead

        while remaining > 0 and pos_px < band_end_px:
            to_read = min(chunk_size, remaining)
            # Keep reads aligned to 4-byte (count,color) pairs
            to_read -= to_read % 4
            if to_read == 0:
                to_read = remaining if remaining <= 4 else 4
            chunk = self.f.read(to_read)
            remaining -= len(chunk)

            n_pairs = len(chunk) // 4
            for i in range(n_pairs):
                count, color = struct.unpack_from("<HH", chunk, i * 4)
                run_start = pos_px
                run_end = pos_px + count
                pos_px = run_end

                if run_end <= band_start_px:
                    continue        # run entirely before this band
                if run_start >= band_end_px:
                    remaining = 0    # run entirely after this band -- done
                    break

                # Clip the run to the band's pixel range
                clip_start = max(run_start, band_start_px)
                clip_end = min(run_end, band_end_px)
                clip_count = clip_end - clip_start

                # Write clip_count copies of the 2-byte color at the right
                # offset within out_buf
                write_off = (clip_start - band_start_px) * 2
                lo = color & 0xFF
                hi = (color >> 8) & 0xFF
                for _ in range(clip_count):
                    out_buf[write_off] = lo
                    out_buf[write_off + 1] = hi
                    write_off += 2

    def close(self):
        self.f.close()


def find_available_clips():
    """Scans the flash filesystem for <name>.bin sprite files."""
    clips = {}
    for fname in os.listdir():
        if fname.endswith(".bin"):
            name = fname[:-4]
            try:
                clips[name] = SpriteClip(fname)
                print("Loaded clip:", name, "-", clips[name].n_frames, "frames")
            except Exception as e:
                print("Skipping", fname, "- failed to load:", e)
    return clips


def play_frame(clip, frame_idx, tft):
    tft.show()
    for band in range(N_BANDS):
        clip.decode_band(frame_idx, band, _band_buf)
        tft.show_band(_band_buf)


def main():
    print("ADAM Pico Sprite Player starting...")
    tft = ST7789()
    print("ST7789 OK")

    clips = find_available_clips()
    if not clips:
        print("No .bin sprite files found on flash. Run generate_frames.py")
        print("on your PC and copy the .bin file(s) here first.")
        return

    names = list(clips.keys())
    print("Available:", ", ".join(names))

    if TESTING_MODE:
        print("TESTING MODE — cycling clips every 4s")
        clip_idx = 0
        frame_idx = 0
        last_frame_t = time.ticks_ms()
        last_switch_t = time.ticks_ms()

        while True:
            now = time.ticks_ms()
            clip = clips[names[clip_idx]]

            if time.ticks_diff(now, last_frame_t) >= clip.delay_ms:
                play_frame(clip, frame_idx, tft)
                frame_idx = (frame_idx + 1) % clip.n_frames
                last_frame_t = now
                gc.collect()

            if time.ticks_diff(now, last_switch_t) > 4000:
                clip_idx = (clip_idx + 1) % len(names)
                frame_idx = 0
                print("→", names[clip_idx])
                last_switch_t = now

    else:
        print("LIVE MODE — waiting for UART emotion commands on GP1")
        uart = machine.UART(0, baudrate=115200,
                             tx=machine.Pin(0), rx=machine.Pin(1))
        rxbuf = b""
        MAX_RXBUF_LEN = 256

        current_name = "idle" if "idle" in clips else names[0]
        frame_idx = 0
        last_frame_t = time.ticks_ms()

        while True:
            now = time.ticks_ms()

            if uart.any():
                chunk = uart.read(uart.any())
                if chunk:
                    rxbuf += chunk
                if len(rxbuf) > MAX_RXBUF_LEN:
                    print("⚠️  UART rxbuf overflow — discarding", len(rxbuf), "bytes")
                    rxbuf = b""
                while b"\n" in rxbuf:
                    line, rxbuf = rxbuf.split(b"\n", 1)
                    try:
                        cmd = line.decode("utf-8").strip().lower()
                    except UnicodeError:
                        print("⚠️  Dropped malformed UART line (non-UTF8 bytes)")
                        continue
                    if cmd in clips:
                        if cmd != current_name:
                            current_name = cmd
                            frame_idx = 0
                            print("→", current_name)
                    elif cmd:
                        print("⚠️  Unknown/missing clip:", repr(cmd))

            clip = clips[current_name]
            if time.ticks_diff(now, last_frame_t) >= clip.delay_ms:
                play_frame(clip, frame_idx, tft)
                frame_idx = (frame_idx + 1) % clip.n_frames
                last_frame_t = now
                gc.collect()

main()