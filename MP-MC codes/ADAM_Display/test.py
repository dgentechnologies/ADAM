"""
ADAM v32 — Raspberry Pi Pico Face Renderer  (FIXED)
=====================================================
Driver  : ST7789 320×240 (2.4 inch)
SPI     : polarity=1, phase=1  @  40 MHz
Colors  : RGB565 byte-swapped for ST7789 big-endian SPI
UART    : GP1 RX ← ESP32-CAM GPIO 3 (U0RXD, repurposed relay TX)  @  115200 baud

Pinout (matches ADAM v32 Blueprint §D):
  GP19 (Pin 25)  → TFT MOSI
  GP18 (Pin 24)  → TFT SCLK
  GP17 (Pin 22)  → TFT CS
  GP16 (Pin 21)  → TFT DC
  GP20 (Pin 26)  → TFT RST
  GP1  (Pin  2)  ← UART0 RX  (from ESP32-CAM GPIO 3, one-directional relay)
  3V3 OUT        → TFT VCC + LED
  GND            → TFT GND

UART emotion commands (plain ASCII + newline):
  idle | speaking | happy | sad | angry | panic
  surprised | shy | sleep | thinking | reconnecting
  love | confused | rizz | confetti

Set TESTING_MODE = True to auto-cycle all emotions
without the ESP32-CAM connected.

── FIXES APPLIED (vs the version that was disconnecting the Pico) ──
1. random.randint() replaced with a getrandbits()-based helper.
   randint is NOT guaranteed present on minimal MicroPython builds —
   if missing, it throws at IMPORT time, before anything prints,
   which matches "crashes with zero output."
2. Confetti piece count cut 22 → 12, and the per-piece polygon fill
   replaced with a cheap plus-sign draw (4 short lines) instead of a
   scanline-filled rotated square. The old fill did a nested O(size)
   scanline loop x22 pieces x30fps — a real CPU/allocation spike that
   can starve SPI timing or fragment RAM on a Pico's ~110KB free heap.
3. Stroke thicknesses (T_EYE/T_MOUTH/T_THIN) pulled back down to the
   old file's values (5/5/2-3 instead of 8/8/4). _ellipse_outline()
   loops `for d in range(t)`, calling a full 72-step _arc() each time
   — at t=8 that's ~8x the line-draw work per outline vs t=5. This was
   almost certainly the single biggest per-frame CPU spike.
4. Everything else (UART hardening, geometry) kept as-is from your
   working "FINAL" file.
"""

# FIX: root cause confirmed by the user's actual board output — even
# as the LITERAL FIRST allocation on a freshly power-cycled board with
# 224KB+ free, a single 153,600-byte bytearray still raised MemoryError.
# That rules out import ordering / fragmentation from prior code; the
# heap arena on this MicroPython v1.28.0 RP2040 build genuinely cannot
# hand out one 150KB contiguous block, likely due to how the allocator
# partitions the arena internally on this port.
#
# Real fix: BAND RENDERING. Instead of one 320x240 (153.6KB) buffer,
# use a small BAND_H-row-tall buffer (default 60 rows -> 320*60*2 =
# 38.4KB, comfortably small) and render each frame in multiple passes,
# one per band, sending each band to its correct region of the screen
# via the ST7789 column/row address window before writing it.
#
# To keep every existing draw_xxx() function completely unchanged (they
# all call fb.fill(...), fb.line(...), fb.fill_rect(...) with absolute
# 0-239 screen Y coordinates), `fb` below is NOT a raw framebuf.FrameBuffer.
# It's a tiny shim (_BandFB) that:
#   - holds one small real FrameBuffer sized W x BAND_H
#   - exposes .fill / .line / .fill_rect / .pixel with the SAME signatures
#   - offsets and clips every Y coordinate against the currently active
#     band (set via fb.set_band(band_index) each pass)
# So draw_happy(ms) etc. keep writing "MY=184" style absolute coordinates
# exactly as before; the shim quietly discards anything outside the
# current band and shifts what's left into the small real buffer.
import gc
gc.collect()

W, H = 320, 240
TESTING_MODE = True   # False = live UART from ESP32-CAM

import machine, time, math, framebuf

try:
    import random
    _HAVE_RANDOM = hasattr(random, "getrandbits")
except ImportError:
    _HAVE_RANDOM = False

# Band height chosen so W * BAND_H * 2 stays comfortably allocatable on
# this board. Bigger BAND_H = fewer redraw passes per frame = smoother
# animation, at the cost of a bigger single allocation. 60 (38.4KB) was
# confirmed working; try raising this if you want less choppiness --
# test with the quick allocation check below before committing to a
# value, since this board's heap can't always give the full 150KB.
BAND_H = 120
assert H % BAND_H == 0, "BAND_H must divide H evenly"
N_BANDS = H // BAND_H

gc.collect()
try:
    _band_buf = bytearray(W * BAND_H * 2)
except MemoryError:
    _free_now = gc.mem_free() if hasattr(gc, "mem_free") else "?"
    print("FATAL: even a", W*BAND_H*2, "byte band buffer failed to allocate.")
    print("Free heap:", _free_now, "bytes.")
    print("Try lowering BAND_H further (e.g. 40 or 30) and re-run.")
    raise

_real_fb = framebuf.FrameBuffer(_band_buf, W, BAND_H, framebuf.RGB565)


class _BandFB:
    """Drop-in shim so every existing draw_xxx()/_line()/_rect()/etc.
    function can keep using absolute 0..H-1 screen Y coordinates
    without modification. Internally offsets into the small real
    per-band FrameBuffer and clips anything outside the active band.
    Only implements the handful of framebuf methods this file actually
    calls: fill, line, fill_rect, pixel."""

    def __init__(self, real_fb, band_h):
        self._fb = real_fb
        self._band_h = band_h
        self._y0 = 0          # top of the currently active band, in screen coords
        self._y1 = band_h     # bottom (exclusive) of the currently active band

    def set_band(self, band_index):
        self._y0 = band_index * self._band_h
        self._y1 = self._y0 + self._band_h

    def fill(self, col):
        self._fb.fill(col)

    def pixel(self, x, y, col=None):
        if not (self._y0 <= y < self._y1):
            return None if col is None else None
        by = y - self._y0
        if col is None:
            return self._fb.pixel(x, by)
        self._fb.pixel(x, by, col)

    def line(self, x0, y0, x1, y1, col):
        # Clip the line's Y span to the active band before handing off.
        # framebuf has no native clipping, so if either endpoint is
        # outside the band we skip it entirely -- fine for this file's
        # usage since all shapes are built from many short line segments
        # (arcs, zigzags, thick-line offsets), so losing a segment that
        # falls in a different band just means it gets drawn on that
        # band's pass instead.
        lo = min(y0, y1); hi = max(y0, y1)
        if hi < self._y0 or lo >= self._y1:
            return
        self._fb.line(x0, y0 - self._y0, x1, y1 - self._y0, col)

    def fill_rect(self, x, y, w, h, col):
        y_top = y
        y_bot = y + h
        clip_top = max(y_top, self._y0)
        clip_bot = min(y_bot, self._y1)
        if clip_bot <= clip_top:
            return
        self._fb.fill_rect(x, clip_top - self._y0, w, clip_bot - clip_top, col)


fb = _BandFB(_real_fb, BAND_H)

# ─────────────────────────────────────────────────────────────
# COLOR  (RGB565, bytes swapped for ST7789 SPI big-endian)
# ─────────────────────────────────────────────────────────────
def _c(r, g, b):
    v = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    return ((v & 0xFF) << 8) | (v >> 8)

BG      = _c(  0,   0,   0)
WHITE   = _c(255, 255, 255)
DIM     = _c(100, 100, 100)
PINK    = _c(255,  90, 110)
BLUE    = _c( 70, 130, 255)
RED     = _c(220,  30,  30)
ORANGE  = _c(255, 140,   0)
YELLOW  = _c(255, 230,  60)
GREEN   = _c( 60, 220, 110)
PURPLE  = _c(170,  90, 240)
CYAN    = _c( 70, 220, 230)

CONFETTI_COLORS = [PINK, BLUE, YELLOW, ORANGE, GREEN, PURPLE, CYAN, RED]

def _grey(b):           # brightness 0.0-1.0  → swapped RGB565
    v = int(max(0, min(1, b)) * 255)
    return _c(v, v, v)

def _rand_range(a, b):
    """Small helper — int in [a, b] inclusive.
    FIX: does NOT rely on random.randint (not guaranteed present on
    minimal MicroPython ports — missing randint throws AttributeError
    at import time, before anything prints, which matches a silent
    boot-time crash). Uses getrandbits() only, with a manual modulo
    fallback, and a deterministic fallback if random is unavailable
    at all so confetti positions are still spread out predictably."""
    if b <= a:
        return a
    span = b - a + 1
    if _HAVE_RANDOM:
        # FIX: int.bit_length() is not available on this MicroPython
        # build (AttributeError). Compute the needed bit width manually
        # instead of relying on that method.
        bits = 1
        n = span
        while n > 1:
            n >>= 1
            bits += 1
        v = random.getrandbits(bits) % span
        return a + v
    # Fallback: cheap deterministic pseudo-spread using time-based seed
    # so behaviour is still varied across pieces without needing random.
    t = time.ticks_us() & 0xFFFF
    return a + (t % span)

# ─────────────────────────────────────────────────────────────
# FACE GEOMETRY  — scaled up to fill most of the 320×240 panel
# ─────────────────────────────────────────────────────────────
EL  = 78        # left  eye centre X
ER  = 242       # right eye centre X
EY  = 92        # eye centre Y
EYE_RX = 42     # eye ellipse half-width
EYE_RY = 42     # eye ellipse half-height
MY  = 184       # mouth centre Y
CX  = 160       # horizontal centre

# FIX: pulled stroke thickness back to old-file values. The new file's
# T_EYE=8/T_MOUTH=8/T_THIN=4 made _ellipse_outline() (which loops
# `for d in range(t)`, each doing a full 72-step arc) roughly 60% more
# expensive per call — multiplied across every eye/frame/emotion this
# was the most likely source of a frame-time spike bad enough to glitch
# SPI/USB on-device, even though it never showed as a Python exception.
T_EYE   = 5
T_MOUTH = 5
T_THIN  = 3

# ─────────────────────────────────────────────────────────────
# ST7789 DRIVER
# ─────────────────────────────────────────────────────────────
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
        self._cmd(0x11); time.sleep_ms(120)   # Sleep Out
        self._cmd(0x36); self._dat([0x60])     # MADCTL
        self._cmd(0x3A); self._dat([0x55])     # 16-bit colour
        self._cmd(0x20)                        # Inversion OFF
        self._cmd(0x13)                        # Normal display on
        self._cmd(0x29); time.sleep_ms(50)     # Display on

    def show(self):
        # FIX: with band rendering, there's no single full-frame _buf
        # to blast anymore. show() now just prepares the FULL-SCREEN
        # address window once; each band's pixel data is streamed to
        # the display separately via show_band() as it's rendered,
        # so the address window auto-increments across band boundaries
        # and we never need more than BAND_H rows of the screen buffered
        # in RAM at once.
        self._cmd(0x2A); self._dat([0x00,0x00,(W-1)>>8,(W-1)&0xFF])
        self._cmd(0x2B); self._dat([0x00,0x00,(H-1)>>8,(H-1)&0xFF])
        self._cmd(0x2C)

    def show_band(self, band_buf):
        # Streams one band's worth of pixel data. Must be called once
        # per band, in top-to-bottom order, immediately after show()
        # sets up the address window -- the ST7789's RAMWR (0x2C)
        # auto-increments through the window on successive writes, so
        # sequential band pushes fill the screen correctly without
        # needing to re-set the address window each time.
        self._dc(1); self._cs(0)
        self._spi.write(band_buf)
        self._cs(1)


# ─────────────────────────────────────────────────────────────
# DRAWING PRIMITIVES
# ─────────────────────────────────────────────────────────────

def _line(x0, y0, x1, y1, col, t=4):
    dx = abs(x1-x0); dy = abs(y1-y0)
    steep = dy > dx
    h = t >> 1
    for d in range(-h, h+1):
        if steep:
            fb.line(x0+d, y0, x1+d, y1, col)
        else:
            fb.line(x0, y0+d, x1, y1+d, col)

def _arc(cx, cy, rx, ry, a0_deg, a1_deg, col, t=4, steps=48):
    prev = None
    for i in range(steps+1):
        a = math.radians(a0_deg + (a1_deg - a0_deg) * i / steps)
        x = int(cx + rx * math.cos(a))
        y = int(cy + ry * math.sin(a))
        if prev:
            _line(prev[0], prev[1], x, y, col, t)
        prev = (x, y)

def _ellipse_outline(cx, cy, rx, ry, col, t=4):
    for d in range(t):
        _arc(cx, cy, rx-d, ry-d, 0, 360, col, 1, 72)

def _fill_ellipse(cx, cy, rx, ry, col):
    for dy in range(-ry, ry+1):
        if ry == 0: continue
        dx = int(rx * math.sqrt(max(0, 1-(dy/ry)**2)))
        x0 = max(0, cx-dx); x1 = min(W-1, cx+dx)
        yy = cy+dy
        if 0 <= yy < H and x1 >= x0:
            fb.fill_rect(x0, yy, x1-x0+1, 1, col)

def _rect(cx, cy, w, h, col):
    r = h // 2
    fb.fill_rect(cx - w//2 + r, cy - h//2, w - 2*r, h, col)
    _fill_ellipse(cx - w//2 + r, cy, r, r, col)
    _fill_ellipse(cx + w//2 - r, cy, r, r, col)

def _zigzag(pts, col, t=5):
    for i in range(len(pts)-1):
        _line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], col, t)

def _sparkle(sx, sy, arm, bright, col=None):
    if bright < 0.03: return
    c = col if col else _grey(bright)
    da = int(arm*0.7)
    _line(sx, sy-arm, sx, sy+arm, c, 2)
    _line(sx-arm, sy, sx+arm, sy, c, 2)
    fb.line(sx-da, sy-da, sx+da, sy+da, c)
    fb.line(sx+da, sy-da, sx-da, sy+da, c)

def _heart(cx, cy, sz, col, fill=False, t=3):
    pts = []
    for i in range(61):
        a = 2*math.pi*i/60
        sc = sz/17.0
        x = int(cx + sc*16*(math.sin(a)**3))
        y = int(cy - sc*(13*math.cos(a)-5*math.cos(2*a)-2*math.cos(3*a)-math.cos(4*a)))
        pts.append((x, y))
    if fill:
        min_y = max(0, min(p[1] for p in pts))
        max_y = min(H-1, max(p[1] for p in pts))
        for yy in range(min_y, max_y+1):
            xs = [p[0] for p in pts if p[1] == yy]
            if xs:
                fb.fill_rect(min(xs), yy, max(xs)-min(xs)+1, 1, col)
    else:
        for i in range(len(pts)-1):
            _line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], col, t)

def _confetti_plus(cx, cy, size, angle_rad, col):
    """FIX: cheap confetti piece — a small rotated plus/X made of two
    short lines, instead of a scanline-filled rotated square. Visually
    reads fine as a tumbling piece at this size, and is O(size) line
    draws instead of an O(size^2) scanline polygon fill x12-22 pieces
    x30fps. This was the other big per-frame cost in the version that
    was crashing."""
    h = size / 2.0
    cs, sn = math.cos(angle_rad), math.sin(angle_rad)
    x0 = int(cx - h*cs); y0 = int(cy - h*sn)
    x1 = int(cx + h*cs); y1 = int(cy + h*sn)
    x2 = int(cx - h*sn); y2 = int(cy + h*cs)
    x3 = int(cx + h*sn); y3 = int(cy - h*cs)
    _line(x0, y0, x1, y1, col, 2)
    _line(x2, y2, x3, y3, col, 2)


# ─────────────────────────────────────────────────────────────
# EMOTION RENDERERS
# ─────────────────────────────────────────────────────────────

def draw_idle(ms):
    fb.fill(BG)
    blink   = (ms % 6000) < 140
    breathe = math.sin(ms * 0.001047)
    by      = int(breathe * 3)
    ey      = 4 if blink else 16
    _rect(EL, EY+by, EYE_RX*2+6, ey, WHITE)
    _rect(ER, EY+by, EYE_RX*2+6, ey, WHITE)
    mw = int(42 * (1.0 + 0.04*breathe))
    _rect(CX, MY+by, mw, 9, DIM)

def draw_speaking(ms):
    fb.fill(BG)
    blink = (ms % 3800) < 100
    ph    = (ms % 420) / 420.0
    mw    = 84 if ph < 0.33 else (56 if ph < 0.66 else 30)
    ey    = 4 if blink else 16
    _rect(EL, EY, EYE_RX*2+6, ey, WHITE)
    _rect(ER, EY, EYE_RX*2+6, ey, WHITE)
    _rect(CX, MY, mw, 14, WHITE)

def draw_happy(ms):
    fb.fill(BG)
    blink  = (ms % 4000) < 120
    bounce = int(math.sin(ms * 0.00286) * 4)
    for off, sx, sy, arm in [(0,40,48,11),(600,280,50,11),(1100,298,140,8)]:
        ts = (ms+off) % 2200
        if 300 < ts < 1400:
            br = (ts-300)/550.0 if ts < 850 else (1400-ts)/550.0
            _sparkle(sx, sy, int(arm*br+1), br)
    if blink:
        fb.fill_rect(EL-EYE_RX-2, EY, EYE_RX*2+4, 6, WHITE)
        fb.fill_rect(ER-EYE_RX-2, EY, EYE_RX*2+4, 6, WHITE)
    else:
        _arc(EL, EY+6, EYE_RX+4, EYE_RY-2, 180, 360, WHITE, T_EYE)
        _arc(ER, EY+6, EYE_RX+4, EYE_RY-2, 180, 360, WHITE, T_EYE)
    _arc(CX, MY-14+bounce, 46, 24, 0, 180, WHITE, T_MOUTH)

def draw_sad(ms):
    fb.fill(BG)
    by = int(math.sin(ms * 0.00157) * 6)
    _arc(EL, EY-6+by, EYE_RX+4, EYE_RY-2, 0, 180, WHITE, T_EYE)
    _arc(ER, EY-6+by, EYE_RX+4, EYE_RY-2, 0, 180, WHITE, T_EYE)
    _arc(CX, MY+12+by, 44, 20, 180, 360, WHITE, T_MOUTH)
    for po, ex in [(0.2, EL), (1.0, ER)]:
        tp = math.fmod(ms*0.001+po, 1.5)
        if tp < 0.55: continue
        d  = (tp-0.55)/0.95
        ty = EY+40+by+int(d*46)
        av = int(min(1, max(0, 1-(tp-0.55)/0.95)) * 200)
        tc = _c(av//4, av//3, av)
        _fill_ellipse(ex, ty, 8, 11, tc)

def draw_angry(ms):
    fb.fill(BG)
    t = ms % 100
    sx = -2 if t<25 else (2 if t<75 else 0)
    sy =  1 if t<50 else -1
    _line(EL-EYE_RX+sx, EY-EYE_RY+sy, EL+EYE_RX+sx, EY+EYE_RY+sy, WHITE, T_EYE+1)
    _line(ER+EYE_RX+sx, EY-EYE_RY+sy, ER-EYE_RX+sx, EY+EYE_RY+sy, WHITE, T_EYE+1)
    mp = [(92,MY),(110,MY-20),(128,MY),(146,MY+20),(160,MY),
          (178,MY-20),(196,MY),(214,MY+20),(228,MY)]
    _zigzag([(x+sx,y+sy) for x,y in mp], WHITE, T_MOUTH)
    vp = (ms%600)/600.0
    vb = vp*2 if vp<0.5 else (1-vp)*2
    vc = _c(int(200*vb+55), int(50*vb), int(50*vb))
    _line(EL-24, EY-46, EL-34, EY-62, vc, T_THIN)
    _line(EL-32, EY-52, EL-42, EY-48, vc, T_THIN)
    _line(ER+24, EY-46, ER+34, EY-62, vc, T_THIN)
    _line(ER+32, EY-52, ER+42, EY-48, vc, T_THIN)
    ht = ms%1200
    if ht < 960:
        hd = ht/960.0
        hb = min(1.0, (1-abs(hd*2-1)) * 2)
        hc = _c(int(120*hb), int(60*hb), 0)
        hy = EY-10 - int(hd*36)
        fb.line(EL, hy, EL, hy-10, hc)
        fb.line(ER, hy, ER, hy-10, hc)

def draw_panic(ms):
    fb.fill(BG)
    t  = ms % 70
    sx = -3 if t<20 else (3 if t<40 else -1)
    sy =  1 if t<35 else -1
    sf = (ms%550)/550.0
    sb = sf*2 if sf<0.5 else (1-sf)*2
    sc = _c(int(180*sb), int(210*sb), int(255*sb))
    _line(EL-54+sx, EY-26+sy, EL-66+sx, EY+sy, sc, T_THIN-1)
    _line(EL-64+sx, EY-12+sy, EL-74+sx, EY-8+sy, sc, 2)
    _line(ER+54+sx, EY-26+sy, ER+66+sx, EY+sy, sc, T_THIN-1)
    _line(ER+64+sx, EY-12+sy, ER+74+sx, EY-8+sy, sc, 2)
    _ellipse_outline(EL+sx, EY+sy, EYE_RX+5, EYE_RY+8, WHITE, T_EYE)
    _ellipse_outline(ER+sx, EY+sy, EYE_RX+5, EYE_RY+8, WHITE, T_EYE)
    pp = (ms%300)/300.0
    px = 0 if pp<0.45 else (6 if pp<0.55 else (-6 if pp<0.65 else 0))
    py = 0 if pp<0.45 else (-4 if pp<0.55 else (4 if pp<0.65 else 0))
    _fill_ellipse(EL+sx+px, EY+sy+py, 9, 9, WHITE)
    _fill_ellipse(ER+sx+px, EY+sy+py, 9, 9, WHITE)
    mz = [(100,MY),(116,MY-16),(134,MY),(152,MY+16),(170,MY),
          (188,MY-16),(206,MY),(220,MY+10),(228,MY+4)]
    _zigzag([(x+sx,y+sy) for x,y in mz], WHITE, T_MOUTH)
    for po, ex in [(0.2, EL), (1.3, ER)]:
        sw = math.fmod(ms*0.001+po, 2.2)
        if sw < 0.35: continue
        d  = (sw-0.35)/1.85
        swy = EY-40 + int(d*58)
        av  = int(min(1.0, max(0.0, 1.0-abs(d-0.5)*2+0.1)) * 200)
        tc  = _c(av//4, av//3, av)
        _fill_ellipse(ex, swy, 7, 10, tc)

def draw_surprised(ms):
    fb.fill(BG)
    phase = math.fmod(ms*0.001, 2.8)
    es    = min(1.0, phase/0.18)
    erx   = int((EYE_RX+6)*es)
    ery   = int((EYE_RY+10)*es)
    if 0.25 < phase < 1.9:
        lb = min(1.0,(phase-0.25)/0.2) * (((1.9-phase)/0.2) if phase>1.7 else 1.0)
        lc = _grey(lb*0.55)
        _line(EL-30,EY-34, EL-54,EY-50, lc, T_THIN-1)
        _line(EL-38,EY-14, EL-64,EY-10, lc, T_THIN-1)
        _line(EL-30,EY+12, EL-58,EY+18, lc, T_THIN-1)
        _line(ER+30,EY-34, ER+54,EY-50, lc, T_THIN-1)
        _line(ER+38,EY-14, ER+64,EY-10, lc, T_THIN-1)
        _line(ER+30,EY+12, ER+58,EY+18, lc, T_THIN-1)
    if erx > 2 and ery > 2:
        _ellipse_outline(EL, EY, erx, ery, WHITE, T_EYE)
        _ellipse_outline(ER, EY, erx, ery, WHITE, T_EYE)
    if phase > 0.18:
        ms2 = min(1.0,(phase-0.18)/0.2)
        mrx = max(3, int(14*ms2))
        mry = max(3, int(13*ms2))
        _ellipse_outline(CX, MY, mrx, mry, WHITE, T_MOUTH-2)

def draw_shy(ms):
    fb.fill(BG)
    blink = (ms%5000) < 120
    bi    = 0.35 + 0.3*(math.sin(ms*0.00251)*0.5+0.5)
    by    = int(math.sin(ms*0.00105)*7)
    br = int(bi*200)
    bc = _c(br, br//3, br//2)
    _fill_ellipse(EL-18, EY+38+by, 32, 16, bc)
    _fill_ellipse(ER+18, EY+38+by, 32, 16, bc)
    if blink:
        fb.fill_rect(EL-EYE_RX-2, EY+by, EYE_RX*2+4, 6, WHITE)
        fb.fill_rect(ER-EYE_RX-2, EY+by, EYE_RX*2+4, 6, WHITE)
    else:
        _arc(EL, EY+6+by, EYE_RX+4, EYE_RY-2, 180, 360, WHITE, T_EYE)
        _arc(ER, EY+6+by, EYE_RX+4, EYE_RY-2, 180, 360, WHITE, T_EYE)
    _arc(CX, MY-4+by, 20, 15, 0, 180, WHITE, T_MOUTH-2)
    hf = math.fmod(ms*0.001+1.5, 3.0)
    if 0.1 < hf < 2.6:
        d  = hf/2.6
        hx = 288+int(12*d); hy = 82-int(40*d)
        av = int(min(1,max(0, hf/0.4 if hf<0.4 else ((2.6-hf)/0.4 if hf>2.2 else 1.0)))*160)
        hc = _c(av, av//6, av//5)
        _heart(hx, hy, 12, hc, fill=False, t=3)

def draw_sleep(ms):
    fb.fill(BG)
    by = int(math.sin(ms*0.00126)*2)
    _arc(EL, EY-4+by, EYE_RX+4, EYE_RY-2, 0, 180, WHITE, T_EYE)
    _arc(ER, EY-4+by, EYE_RX+4, EYE_RY-2, 0, 180, WHITE, T_EYE)
    for i, (zx,zy,sc) in enumerate([(262,70,4),(278,50,3),(292,34,2)]):
        zp = math.fmod((ms+i*500)*0.001, 4.0)
        if zp < 0.1 or zp > 3.8: continue
        za = min(1.0, zp/0.25 if zp<0.25 else ((4.0-zp)/0.5 if zp>3.5 else 1.0))
        zy2 = zy - int(zp*10)
        zc  = _grey(za*0.9)
        sw = 8*sc; sh = 8*sc
        _line(zx,     zy2,    zx+sw, zy2,    zc, sc)
        _line(zx+sw,  zy2,    zx,    zy2+sh, zc, sc)
        _line(zx,     zy2+sh, zx+sw, zy2+sh, zc, sc)

def draw_thinking(ms):
    fb.fill(BG)
    so = int(math.sin(ms*0.00393)*10)
    lw = [(EL-EYE_RX+so,EY),(EL-EYE_RX//2+so,EY-EYE_RY//2),
          (EL+so,EY),(EL+EYE_RX//2+so,EY+EYE_RY//2),(EL+EYE_RX+so,EY)]
    rw = [(ER-EYE_RX-so,EY),(ER-EYE_RX//2-so,EY-EYE_RY//2),
          (ER-so,EY),(ER+EYE_RX//2-so,EY+EYE_RY//2),(ER+EYE_RX-so,EY)]
    _zigzag(lw, WHITE, T_EYE)
    _zigzag(rw, WHITE, T_EYE)
    mp = math.sin(ms*0.00314)*0.5+0.5
    mw = 80+int(mp*12)
    _line(CX-mw//2, MY, CX+mw//2, MY, _grey(0.6+mp*0.4), T_MOUTH)
    if (ms//900)%2 == 0:
        fb.fill_rect(CX+mw//2+6, MY-9, 5, 18, WHITE)

def draw_reconnecting(ms):
    fb.fill(BG)
    angle = (ms%1200)/1200.0*2*math.pi
    rp = (ms%1200)/1200.0
    rb = max(0.0, 0.4-rp*0.4)
    if rb > 0.04:
        rc = _c(int(rb*120), int(rb*200), int(rb*255))
        rr = int((EYE_RX+6)*(1.0+rp*1.3))
        _arc(EL, EY, rr, rr, 0, 360, rc, 1, 40)
        _arc(ER, EY, rr, rr, 0, 360, rc, 1, 40)
    for ex, sign in [(EL, 1), (ER, -1)]:
        for k in range(4):
            a  = angle*sign + k*math.pi/2
            al = EYE_RX+4
            tw = T_EYE-2 if k%2==0 else T_THIN
            x0 = ex+int(math.cos(a)*al); y0 = EY+int(math.sin(a)*al)
            x1 = ex-int(math.cos(a)*al); y1 = EY-int(math.sin(a)*al)
            _line(x0,y0,x1,y1, WHITE, tw)
        for k in range(4):
            a  = angle*sign + k*math.pi/2 + math.pi/4
            al = int((EYE_RX+4)*0.72)
            x0 = ex+int(math.cos(a)*al); y0 = EY+int(math.sin(a)*al)
            x1 = ex-int(math.cos(a)*al); y1 = EY-int(math.sin(a)*al)
            _line(x0,y0,x1,y1, WHITE, T_THIN-1)
    mp = math.sin(ms*0.00449)*0.5+0.5
    _line(CX-42, MY, CX+42, MY, _grey(0.5+mp*0.5), T_MOUTH)
    dp = (ms//470)%3
    for d in range(3):
        dc = WHITE if d==dp else DIM
        _fill_ellipse(CX-14+d*14, MY+26, 7, 7, dc)

def draw_love(ms):
    fb.fill(BG)
    t = math.fmod(ms*0.000909, 1.0)
    sc = 1.0
    if   t < 0.14: sc = 1.0 + t/0.14*0.25
    elif t < 0.28: sc = 1.25 - (t-0.14)/0.14*0.25
    elif t < 0.42: sc = 1.0  + (t-0.28)/0.14*0.14
    elif t < 0.56: sc = 1.14 - (t-0.42)/0.14*0.14
    rp = math.fmod(ms*0.000909, 1.0)
    rb = max(0.0, 0.45-rp*0.45)
    if rb > 0.04:
        rc = _c(int(rb*255), int(rb*70), int(rb*90))
        rr = int((EYE_RX+6)*(1.0+rp*1.6))
        _arc(EL, EY, rr, rr, 0, 360, rc, 1, 40)
        _arc(ER, EY, rr, rr, 0, 360, rc, 1, 40)
    _heart(EL, EY, int(40*sc), WHITE, fill=False, t=T_EYE-2)
    _heart(ER, EY, int(40*sc), WHITE, fill=False, t=T_EYE-2)
    sb = int(math.sin(ms*0.00314)*4)
    _arc(CX, MY-14+sb, 46, 24, 0, 180, WHITE, T_MOUTH)
    for po, hx0, hy0 in [(0.5,286,150),(1.2,36,166),(1.8,300,112)]:
        p = math.fmod(ms*0.001+po, 2.5)
        if p < 0.1 or p > 2.4: continue
        d  = p/2.5
        hx = hx0+int(14*d); hy = hy0-int(64*d)
        a  = p/0.4 if p<0.4 else ((2.5-p)/0.5 if p>2.0 else 1.0)
        av = int(max(0,min(1,a))*180)
        hs = max(5, int(14*(1.0-d*0.4)))
        _heart(hx, hy, hs, _c(av, av//5, av//5), fill=False, t=2)

def draw_confused(ms):
    fb.fill(BG)
    to_x = int(math.sin(ms*0.00157)*7)
    for ex, cw in [(EL, True), (ER, False)]:
        base = (ms%2400)/2400.0*2*math.pi * (1 if cw else -1)
        for rad in range(8, EYE_RX+4, 6):
            frac = rad/(EYE_RX+4)
            a0 = base + frac*2*math.pi
            a1 = a0 + math.pi*1.6
            _arc(ex+to_x, EY, rad, rad, math.degrees(a0), math.degrees(a1), WHITE, T_THIN-1, 20)
        _fill_ellipse(ex+to_x, EY, 5, 5, WHITE)
    mp = math.sin(ms*0.00314)*0.5+0.5
    mw = int(86*(1.0-0.18*mp))
    _line(CX-mw//2+to_x, MY, CX+mw//2+to_x, MY, WHITE, T_MOUTH)
    qy = 38 - int(math.sin(ms*0.00251)*14)
    _arc(258, qy+12, 13, 13, 200, 380, WHITE, T_THIN, 20)
    _fill_ellipse(258, qy+32, 4, 4, WHITE)
    _fill_ellipse(258, qy+42, 4, 4, WHITE)

def draw_rizz(ms):
    fb.fill(BG)
    phase   = math.fmod(ms*0.000357, 1.0)
    winking = 0.10 < phase < 0.45
    lo      = int(math.sin(ms*0.00224)*5)
    wh = 5 if winking else 16
    _rect(EL+lo, EY, EYE_RX*2+6, wh, WHITE)
    rh_scale = 0.55 if winking else 1.0
    _arc(ER+lo, EY+6, EYE_RX+4, int((EYE_RY-2)*rh_scale), 180, 360, WHITE, T_EYE)
    _arc(CX+lo+10, MY-10, 32, 15, 0, 160, WHITE, T_MOUTH)
    if 0.50 < phase < 0.80:
        gp = (phase-0.50)/0.30
        gs = math.sin(gp*math.pi)
        _sparkle(282, 50, int(13*gs)+1, gs, _c(int(255*gs), int(255*gs), int(100*gs)))

# ── confetti — lighter version ───────────────────────────────
_CONFETTI_N = 12   # FIX: was 22 — cut for CPU/RAM headroom
_confetti_pieces = []
for _i in range(_CONFETTI_N):
    _confetti_pieces.append({
        "x":        _rand_range(10, W-10),
        "y0":       _rand_range(-H, 0),
        "speed":    _rand_range(60, 140) / 100.0,
        "size":     _rand_range(6, 12),
        "spin":     (_rand_range(2, 9) / 1000.0) * (1 if _i % 2 == 0 else -1),
        "sway_amp": _rand_range(6, 20),
        "sway_hz":  _rand_range(8, 22) / 10000.0,
        "color":    CONFETTI_COLORS[_i % len(CONFETTI_COLORS)],
        "period":   _rand_range(2600, 4200),
    })

def draw_confetti(ms):
    fb.fill(BG)
    bounce = int(math.sin(ms * 0.00314) * 4)
    _arc(EL, EY+6, EYE_RX+4, EYE_RY-2, 180, 360, WHITE, T_EYE)
    _arc(ER, EY+6, EYE_RX+4, EYE_RY-2, 180, 360, WHITE, T_EYE)
    _arc(CX, MY-14+bounce, 50, 26, 0, 180, WHITE, T_MOUTH)

    for p in _confetti_pieces:
        t = math.fmod(ms + p["y0"]*3, p["period"])
        frac = t / p["period"]
        y = int(frac * (H + 40)) - 20
        x = int(p["x"] + math.sin(ms * p["sway_hz"] + p["y0"]) * p["sway_amp"])
        angle = ms * p["spin"]
        if -20 <= y <= H+20:
            _confetti_plus(x, y, p["size"], angle, p["color"])


# ─────────────────────────────────────────────────────────────
# DISPATCH TABLE
# ─────────────────────────────────────────────────────────────
EMOTIONS = {
    "idle":         draw_idle,
    "speaking":     draw_speaking,
    "happy":        draw_happy,
    "sad":          draw_sad,
    "angry":        draw_angry,
    "panic":        draw_panic,
    "surprised":    draw_surprised,
    "shy":          draw_shy,
    "sleep":        draw_sleep,
    "thinking":     draw_thinking,
    "reconnecting": draw_reconnecting,
    "love":         draw_love,
    "confused":     draw_confused,
    "rizz":         draw_rizz,
    "confetti":     draw_confetti,
}

EMOTION_LIST = list(EMOTIONS.keys())

def _render_frame(emotion, ms, tft):
    """FIX: replaces the old single 'draw once, blast whole buffer once'
    call. With band rendering there's no full-screen buffer to draw into
    directly -- instead we call the SAME draw_xxx(ms) function once per
    band, letting the _BandFB shim clip each call's output to whichever
    band is currently active, and push that band to the display before
    moving to the next one.

    Note this means each draw_xxx(ms) runs N_BANDS times per displayed
    frame (once per band) -- more total CPU per frame than the original
    single-buffer approach, but each individual allocation/window stays
    tiny, which is the actual fix for the MemoryError on this board.
    If frame rate feels too slow once this is confirmed working, the
    first easy speedup is raising BAND_H (e.g. 60 -> 80 or 120) to cut
    the number of passes, as long as W*BAND_H*2 still allocates fine.
    """
    tft.show()  # sets the full-screen address window once
    draw_fn = EMOTIONS[emotion]
    for band in range(N_BANDS):
        fb.set_band(band)
        draw_fn(ms)
        tft.show_band(_band_buf)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("ADAM Pico TFT starting...")
    tft = ST7789()
    print("ST7789 OK")
    print("Available emotions:", ", ".join(EMOTION_LIST))
    print("Band rendering: BAND_H=%d, %d bands per frame" % (BAND_H, N_BANDS))

    emotion = "idle"

    if TESTING_MODE:
        print("TESTING MODE — cycling emotions every 4 s")
        idx        = 0
        last_sw    = time.ticks_ms()
        start_ms   = time.ticks_ms()

        while True:
            ms = time.ticks_diff(time.ticks_ms(), start_ms)
            if time.ticks_diff(time.ticks_ms(), last_sw) > 4000:
                idx     = (idx+1) % len(EMOTION_LIST)
                emotion = EMOTION_LIST[idx]
                print("→", emotion)
                last_sw = time.ticks_ms()

            _render_frame(emotion, ms, tft)
            gc.collect()
            time.sleep_ms(33)

    else:
        print("LIVE MODE — waiting for UART emotion commands on GP1")
        uart     = machine.UART(0, baudrate=115200,
                                tx=machine.Pin(0), rx=machine.Pin(1))
        rxbuf    = b""
        start_ms = time.ticks_ms()
        MAX_RXBUF_LEN = 256

        while True:
            ms = time.ticks_diff(time.ticks_ms(), start_ms)

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
                    if cmd in EMOTIONS:
                        emotion = cmd
                        print("→", emotion)
                    elif cmd:
                        print("⚠️  Unknown emotion command:", repr(cmd))

            _render_frame(emotion, ms, tft)
            gc.collect()
            time.sleep_ms(33)

main()