"""
ADAM v32 — Raspberry Pi Pico Face Renderer
=====================================================
Driver  : ST7789 320x240 (2.4" SPI TFT)
UART    : Pico GP14 TX (Pin 19) -> ESP32 RX
          Pico GP15 RX (Pin 20) <- ESP32 TX  @ 115200 baud
          (CTS/RTS pins on RP2040; uses PIO UART fallback)

Wiring (matches ADAM Blueprint):
  GP19 (Pin 25) -> TFT MOSI
  GP18 (Pin 24) -> TFT SCLK
  GP17 (Pin 22) -> TFT CS
  GP16 (Pin 21) -> TFT DC
  GP20 (Pin 26) -> TFT RST
  GP14 (Pin 19) -> ESP32 RX (UART TX)
  GP15 (Pin 20) <- ESP32 TX (UART RX)
  3V3 OUT       -> TFT VCC + LED
  GND           -> Common GND (TFT + ESP32)

Supported UART Commands:
  idle | speaking | happy | sad | angry | panic | surprised
  shy  | sleep | thinking | reconnecting | love | confused | rizz
"""

import machine, time, math, framebuf, gc
try:
    import rp2
except ImportError:
    rp2 = None

# Config
TESTING_MODE = False   # True: auto-cycle emotions | False: live UART
UART_TX_PIN  = 14      # Pico GP14 (Pin 19)
UART_RX_PIN  = 15      # Pico GP15 (Pin 20)
UART_BAUD    = 115200
W, H         = 320, 240
BAND_H       = 60      # 38.4KB buffer (saves 115KB RAM vs full 153.6KB)
N_BANDS      = H // BAND_H

# Framebuffer & Memory Allocation
gc.collect()
try:
    _band_buf = bytearray(W * BAND_H * 2)
except MemoryError:
    BAND_H = 40
    N_BANDS = H // BAND_H
    _band_buf = bytearray(W * BAND_H * 2)

_real_fb = framebuf.FrameBuffer(_band_buf, W, BAND_H, framebuf.RGB565)

# Fast Pure-Integer Cohen-Sutherland Line Clipping (0 heap allocations)
def _clip_line(x0, y0, x1, y1, w, h):
    xm, ym = w - 1, h - 1
    c0 = (1 if x0 < 0 else (2 if x0 > xm else 0)) | (4 if y0 < 0 else (8 if y0 > ym else 0))
    c1 = (1 if x1 < 0 else (2 if x1 > xm else 0)) | (4 if y1 < 0 else (8 if y1 > ym else 0))
    while True:
        if not (c0 | c1): return x0, y0, x1, y1
        if c0 & c1: return None
        c = c0 or c1
        if c & 4:   # TOP
            x = x0 + (x1 - x0) * -y0 // (y1 - y0); y = 0
        elif c & 8: # BOTTOM
            x = x0 + (x1 - x0) * (ym - y0) // (y1 - y0); y = ym
        elif c & 2: # RIGHT
            y = y0 + (y1 - y0) * (xm - x0) // (x1 - x0); x = xm
        else:       # LEFT
            y = y0 + (y1 - y0) * -x0 // (x1 - x0); x = 0
        if c == c0:
            x0, y0 = x, y
            c0 = (1 if x0 < 0 else (2 if x0 > xm else 0)) | (4 if y0 < 0 else (8 if y0 > ym else 0))
        else:
            x1, y1 = x, y
            c1 = (1 if x1 < 0 else (2 if x1 > xm else 0)) | (4 if y1 < 0 else (8 if y1 > ym else 0))

class _BandFB:
    __slots__ = ('_fb', '_band_h', '_y0', '_y1')
    def __init__(self, real_fb, band_h):
        self._fb = real_fb
        self._band_h = band_h
        self._y0 = 0
        self._y1 = band_h

    def set_band(self, idx):
        self._y0 = idx * self._band_h
        self._y1 = self._y0 + self._band_h

    def fill(self, col):
        self._fb.fill(col)

    def pixel(self, x, y, col=None):
        if 0 <= x < 320 and self._y0 <= y < self._y1:
            if col is None: return self._fb.pixel(x, y - self._y0)
            self._fb.pixel(x, y - self._y0, col)

    def line(self, x0, y0, x1, y1, col):
        by0, by1 = y0 - self._y0, y1 - self._y0
        bh = self._band_h
        # Trivial reject
        if (by0 < 0 and by1 < 0) or (by0 >= bh and by1 >= bh) or (x0 < 0 and x1 < 0) or (x0 >= 320 and x1 >= 320):
            return
        # Trivial accept
        if 0 <= x0 < 320 and 0 <= x1 < 320 and 0 <= by0 < bh and 0 <= by1 < bh:
            self._fb.line(x0, by0, x1, by1, col)
            return
        c = _clip_line(x0, by0, x1, by1, 320, bh)
        if c: self._fb.line(c[0], c[1], c[2], c[3], col)

    def fill_rect(self, x, y, w, h, col):
        yt, yb = max(y, self._y0), min(y + h, self._y1)
        if yb <= yt: return
        xl, xr = max(0, x), min(320, x + w)
        if xr <= xl: return
        self._fb.fill_rect(xl, yt - self._y0, xr - xl, yb - yt, col)

fb = _BandFB(_real_fb, BAND_H)

# Colors (RGB565 byte-swapped for big-endian SPI)
def _c(r, g, b):
    v = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    return ((v & 0xFF) << 8) | (v >> 8)

BG     = _c(  0,   0,   0)
WHITE  = _c(255, 255, 255)
DIM    = _c(100, 100, 100)
PINK   = _c(255,  90, 110)
BLUE   = _c( 70, 130, 255)
RED    = _c(220,  30,  30)

def _grey(b):
    v = int(max(0, min(1, b)) * 255)
    return _c(v, v, v)

# Geometry
EL, ER = 75, 245     # Eye center X positions (wider gap between eyes)
EY     = 97          # Eye vertical center
EYE_RX = 42          # Eye half-width (total capsule width = 42*2+4 = 88px)
EYE_RY = 42          # Eye half-height for arcs and expressions
EYE_H  = 20          # Idle/speaking capsule height (original 12 -> 20: sleek capsule, not circular)
MY, CX = 178, 160    # Mouth position and screen horizontal center

# ST7789 Driver
class ST7789:
    def __init__(self):
        self._spi = machine.SPI(0, baudrate=40_000_000, polarity=1, phase=1,
                                sck=machine.Pin(18), mosi=machine.Pin(19))
        self._cs  = machine.Pin(17, machine.Pin.OUT)
        self._dc  = machine.Pin(16, machine.Pin.OUT)
        self._rst = machine.Pin(20, machine.Pin.OUT)
        self._cmd_buf = bytearray(1)
        self._reset()
        self._init()

    def _reset(self):
        self._rst(1); time.sleep_ms(50)
        self._rst(0); time.sleep_ms(50)
        self._rst(1); time.sleep_ms(50)

    def _cmd(self, c):
        self._dc(0); self._cs(0)
        self._cmd_buf[0] = c
        self._spi.write(self._cmd_buf)
        self._cs(1)

    def _dat(self, d):
        self._dc(1); self._cs(0)
        self._spi.write(d)
        self._cs(1)

    def _init(self):
        self._cmd(0x11); time.sleep_ms(120)
        self._cmd(0x36); self._dat(b'\xA0')  # b'\xA0' = 180° rotated landscape
        self._cmd(0x3A); self._dat(b'\x55')
        self._cmd(0x20)
        self._cmd(0x13)
        self._cmd(0x29); time.sleep_ms(50)

    def show(self):
        self._cmd(0x2A); self._dat(b'\x00\x00\x01\x3F')
        self._cmd(0x2B); self._dat(b'\x00\x00\x00\xEF')
        self._cmd(0x2C)

    def show_band(self, band_buf):
        self._dc(1); self._cs(0)
        self._spi.write(band_buf)
        self._cs(1)

# Drawing Primitives with Band-Bounds Early Exit
def _line(x0, y0, x1, y1, col, t=4):
    if t <= 1:
        fb.line(x0, y0, x1, y1, col)
        return
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    steep = dy > dx
    h = t >> 1
    for d in range(-h, h + 1):
        if steep: fb.line(x0 + d, y0, x1 + d, y1, col)
        else:     fb.line(x0, y0 + d, x1, y1 + d, col)

def _arc(cx, cy, rx, ry, a0_deg, a1_deg, col, t=4, steps=24):
    if cy + ry + t < fb._y0 or cy - ry - t >= fb._y1: return
    step_r = (a1_deg - a0_deg) / steps * 0.01745329
    base_r = a0_deg * 0.01745329
    px, py = None, None
    for i in range(steps + 1):
        a = base_r + step_r * i
        x = int(cx + rx * math.cos(a))
        y = int(cy + ry * math.sin(a))
        if px is not None:
            _line(px, py, x, y, col, t)
        px, py = x, y

def _ellipse_outline(cx, cy, rx, ry, col, t=4):
    if cy + ry + t < fb._y0 or cy - ry - t >= fb._y1: return
    for d in range(t):
        _arc(cx, cy, rx - d, ry - d, 0, 360, col, 1, 24)

def _fill_ellipse(cx, cy, rx, ry, col):
    if rx <= 0 or ry <= 0 or cy + ry < fb._y0 or cy - ry >= fb._y1: return
    dy_min = max(-ry, fb._y0 - cy)
    dy_max = min(ry, fb._y1 - 1 - cy)
    inv_r2 = 1.0 / (ry * ry)
    for dy in range(dy_min, dy_max + 1):
        dx = int(rx * math.sqrt(max(0.0, 1.0 - dy * dy * inv_r2)))
        fb.fill_rect(cx - dx, cy + dy, (dx << 1) + 1, 1, col)

def _rect(cx, cy, w, h, col):
    if cy + (h >> 1) < fb._y0 or cy - (h >> 1) >= fb._y1: return
    r = h >> 1
    fb.fill_rect(cx - (w >> 1) + r, cy - r, w - (r << 1), h, col)
    _fill_ellipse(cx - (w >> 1) + r, cy, r, r, col)
    _fill_ellipse(cx + (w >> 1) - r, cy, r, r, col)

def _sparkle(sx, sy, arm, bright, col=None):
    if bright < 0.03 or sy + arm < fb._y0 or sy - arm >= fb._y1: return
    c = col or _grey(bright)
    da = int(arm * 0.7)
    _line(sx, sy - arm, sx, sy + arm, c, 2)
    _line(sx - arm, sy, sx + arm, sy, c, 2)
    fb.line(sx - da, sy - da, sx + da, sy + da, c)
    fb.line(sx + da, sy - da, sx - da, sy + da, c)

def _heart(cx, cy, sz, col, t=3):
    if cy + int(sz * 1.3) < fb._y0 or cy - sz - t >= fb._y1: return
    sc = sz / 17.0
    px, py = None, None
    for i in range(25):
        a = 0.261799 * i
        x = int(cx + sc * 16 * (math.sin(a)**3))
        y = int(cy - sc * (13*math.cos(a) - 5*math.cos(2*a) - 2*math.cos(3*a) - math.cos(4*a)))
        if px is not None:
            _line(px, py, x, y, col, t)
        px, py = x, y

# Emotion Renderers
def draw_idle(ms):
    fb.fill(BG)
    blink = (ms % 6000) < 140
    by = int(math.sin(ms * 0.001047) * 2)
    ey = 4 if blink else EYE_H
    _rect(EL, EY + by, EYE_RX * 2 + 4, ey, WHITE)
    _rect(ER, EY + by, EYE_RX * 2 + 4, ey, WHITE)
    _rect(CX, MY + by, int(30 * (1.0 + 0.04 * math.sin(ms * 0.001047))), 6, DIM)

def draw_speaking(ms):
    fb.fill(BG)
    ey = 4 if (ms % 3800) < 100 else EYE_H
    ph = (ms % 420) / 420.0
    mw = 64 if ph < 0.33 else (42 if ph < 0.66 else 22)
    _rect(EL, EY, EYE_RX * 2 + 4, ey, WHITE)
    _rect(ER, EY, EYE_RX * 2 + 4, ey, WHITE)
    _rect(CX, MY, mw, 10, WHITE)

def draw_happy(ms):
    fb.fill(BG)
    for off, sx, sy, arm in ((0, 50, 55, 9), (600, 268, 58, 9), (1100, 285, 130, 6)):
        ts = (ms + off) % 2200
        if 300 < ts < 1400:
            br = (ts - 300) / 550.0 if ts < 850 else (1400 - ts) / 550.0
            _sparkle(sx, sy, int(arm * br + 1), br)
    if (ms % 4000) < 120:
        fb.fill_rect(EL - EYE_RX - 2, EY, EYE_RX * 2 + 4, 5, WHITE)
        fb.fill_rect(ER - EYE_RX - 2, EY, EYE_RX * 2 + 4, 5, WHITE)
    else:
        _arc(EL, EY + 6, EYE_RX + 2, EYE_RY - 4, 180, 360, WHITE, 5)
        _arc(ER, EY + 6, EYE_RX + 2, EYE_RY - 4, 180, 360, WHITE, 5)
    _arc(CX, MY - 12 + int(math.sin(ms * 0.00286) * 3), 34, 17, 0, 180, WHITE, 5)

def draw_sad(ms):
    fb.fill(BG)
    by = int(math.sin(ms * 0.00157) * 5)
    _arc(EL, EY - 6 + by, EYE_RX + 2, EYE_RY - 4, 0, 180, WHITE, 5)
    _arc(ER, EY - 6 + by, EYE_RX + 2, EYE_RY - 4, 0, 180, WHITE, 5)
    _arc(CX, MY + 10 + by, 32, 14, 180, 360, WHITE, 5)
    for po, ex in ((0.2, EL), (1.0, ER)):
        tp = math.fmod(ms * 0.001 + po, 1.5)
        if tp >= 0.55:
            d = (tp - 0.55) / 0.95
            av = int(max(0, 1.0 - d) * 200)
            _fill_ellipse(ex, EY + 36 + by + int(d * 44), 6, 9, _c(av >> 2, av // 3, av))

_ANG_MOUTH = ((100,0), (115,-15), (130,0), (145,15), (160,0), (175,-15), (190,0), (205,15), (220,0))
def draw_angry(ms):
    fb.fill(BG)
    t = ms % 100
    sx = -2 if t < 25 else (2 if t < 75 else 0)
    sy = 1 if t < 50 else -1
    _line(EL - EYE_RX + sx, EY - EYE_RY + sy, EL + EYE_RX + sx, EY + EYE_RY + sy, WHITE, 6)
    _line(ER + EYE_RX + sx, EY - EYE_RY + sy, ER - EYE_RX + sx, EY + EYE_RY + sy, WHITE, 6)
    for i in range(8):
        _line(_ANG_MOUTH[i][0] + sx, MY + _ANG_MOUTH[i][1] + sy, _ANG_MOUTH[i+1][0] + sx, MY + _ANG_MOUTH[i+1][1] + sy, WHITE, 5)
    vp = (ms % 600) / 600.0
    vb = vp * 2 if vp < 0.5 else (1 - vp) * 2
    vc = _c(int(200 * vb + 55), int(50 * vb), int(50 * vb))
    _line(EL - 22, EY - 38, EL - 30, EY - 52, vc, 2)
    _line(EL - 28, EY - 44, EL - 36, EY - 40, vc, 2)
    _line(ER + 22, EY - 38, ER + 30, EY - 52, vc, 2)
    _line(ER + 28, EY - 44, ER + 36, EY - 40, vc, 2)
    ht = ms % 1200
    if ht < 960:
        hd = ht / 960.0
        hb = min(1.0, (1.0 - abs(hd * 2 - 1)) * 2)
        hc = _c(int(120 * hb), int(60 * hb), 0)
        hy = EY - 6 - int(hd * 32)
        fb.line(EL, hy, EL, hy - 8, hc)
        fb.line(ER, hy, ER, hy - 8, hc)

_PAN_MOUTH = ((108,0), (122,-12), (138,0), (154,12), (170,0), (186,-12), (202,0), (214,8), (220,4))
def draw_panic(ms):
    fb.fill(BG)
    t = ms % 70
    sx = -3 if t < 20 else (3 if t < 40 else -1)
    sy = 1 if t < 35 else -1
    sf = (ms % 550) / 550.0
    sb = sf * 2 if sf < 0.5 else (1 - sf) * 2
    sc = _c(int(180 * sb), int(210 * sb), int(255 * sb))
    _line(EL - 46 + sx, EY - 22 + sy, EL - 56 + sx, EY + sy, sc, 3)
    _line(EL - 54 + sx, EY - 10 + sy, EL - 62 + sx, EY - 6 + sy, sc, 2)
    _line(ER + 46 + sx, EY - 22 + sy, ER + 56 + sx, EY + sy, sc, 3)
    _line(ER + 54 + sx, EY - 10 + sy, ER + 62 + sx, EY - 6 + sy, sc, 2)
    _ellipse_outline(EL + sx, EY + sy, EYE_RX + 3, EYE_RY + 6, WHITE, 5)
    _ellipse_outline(ER + sx, EY + sy, EYE_RX + 3, EYE_RY + 6, WHITE, 5)
    pp = (ms % 300) / 300.0
    px = 0 if pp < 0.45 else (5 if pp < 0.55 else (-5 if pp < 0.65 else 0))
    py = 0 if pp < 0.45 else (-3 if pp < 0.55 else (3 if pp < 0.65 else 0))
    _fill_ellipse(EL + sx + px, EY + sy + py, 7, 7, WHITE)
    _fill_ellipse(ER + sx + px, EY + sy + py, 7, 7, WHITE)
    for i in range(8):
        _line(_PAN_MOUTH[i][0] + sx, MY + _PAN_MOUTH[i][1] + sy, _PAN_MOUTH[i+1][0] + sx, MY + _PAN_MOUTH[i+1][1] + sy, WHITE, 5)
    for po, ex in ((0.2, EL), (1.3, ER)):
        sw = math.fmod(ms * 0.001 + po, 2.2)
        if sw >= 0.35:
            d = (sw - 0.35) / 1.85
            av = int(min(1.0, max(0.0, 1.1 - abs(d - 0.5) * 2)) * 200)
            _fill_ellipse(ex, EY - 36 + int(d * 55), 5, 8, _c(av >> 2, av // 3, av))

def draw_surprised(ms):
    fb.fill(BG)
    phase = math.fmod(ms * 0.001, 2.8)
    es = min(1.0, phase / 0.18)
    erx = int((EYE_RX + 4) * es)
    ery = int((EYE_RY + 8) * es)
    if 0.25 < phase < 1.9:
        lb = min(1.0, (phase - 0.25) / 0.2) * (((1.9 - phase) / 0.2) if phase > 1.7 else 1.0)
        lc = _grey(lb * 0.55)
        for sign, ex in ((-1, EL), (1, ER)):
            _line(ex + sign * 25, EY - 28, ex + sign * 46, EY - 42, lc, 2)
            _line(ex + sign * 32, EY - 12, ex + sign * 56, EY - 8,  lc, 2)
            _line(ex + sign * 26, EY + 10, ex + sign * 50, EY + 16, lc, 2)
    if erx > 2 and ery > 2:
        _ellipse_outline(EL, EY, erx, ery, WHITE, 5)
        _ellipse_outline(ER, EY, erx, ery, WHITE, 5)
    if phase > 0.18:
        ms2 = min(1.0, (phase - 0.18) / 0.2)
        _ellipse_outline(CX, MY, max(2, int(10 * ms2)), max(2, int(9 * ms2)), WHITE, 4)

def draw_shy(ms):
    fb.fill(BG)
    by = int(math.sin(ms * 0.00105) * 6)
    br = int((0.35 + 0.15 * (math.sin(ms * 0.00251) + 1)) * 200)
    bc = _c(br, br // 3, br >> 1)
    _fill_ellipse(EL - 16, EY + 32 + by, 26, 13, bc)
    _fill_ellipse(ER + 16, EY + 32 + by, 26, 13, bc)
    if (ms % 5000) < 120:
        fb.fill_rect(EL - EYE_RX - 2, EY + by, EYE_RX * 2 + 4, 5, WHITE)
        fb.fill_rect(ER - EYE_RX - 2, EY + by, EYE_RX * 2 + 4, 5, WHITE)
    else:
        _arc(EL, EY + 6 + by, EYE_RX + 2, EYE_RY - 4, 180, 360, WHITE, 5)
        _arc(ER, EY + 6 + by, EYE_RX + 2, EYE_RY - 4, 180, 360, WHITE, 5)
    _arc(CX, MY - 4 + by, 14, 11, 0, 180, WHITE, 5)
    hf = math.fmod(ms * 0.001 + 1.5, 3.0)
    if 0.1 < hf < 2.6:
        d = hf / 2.6
        av = int(min(1, max(0, hf / 0.4 if hf < 0.4 else ((2.6 - hf) / 0.4 if hf > 2.2 else 1.0))) * 160)
        _heart(278 + int(10 * d), 88 - int(36 * d), 9, _c(av, av // 6, av // 5), 2)

def draw_sleep(ms):
    fb.fill(BG)
    by = int(math.sin(ms * 0.00126) * 2)
    _arc(EL, EY - 4 + by, EYE_RX + 2, EYE_RY - 4, 0, 180, WHITE, 5)
    _arc(ER, EY - 4 + by, EYE_RX + 2, EYE_RY - 4, 0, 180, WHITE, 5)
    for i, (zx, zy, sc) in enumerate(((258, 74, 3), (272, 56, 2), (282, 42, 1))):
        zp = math.fmod((ms + i * 500) * 0.001, 4.0)
        if 0.1 <= zp <= 3.8:
            za = min(1.0, zp / 0.25 if zp < 0.25 else ((4.0 - zp) / 0.5 if zp > 3.5 else 1.0))
            zy2 = zy - int(zp * 10)
            zc = _grey(za * 0.9)
            sw, sh = 8 * sc, 8 * sc
            _line(zx, zy2, zx + sw, zy2, zc, sc)
            _line(zx + sw, zy2, zx, zy2 + sh, zc, sc)
            _line(zx, zy2 + sh, zx + sw, zy2 + sh, zc, sc)

_THK_WAVE = ((-EYE_RX,0), (-EYE_RX//2,-EYE_RY//2), (0,0), (EYE_RX//2,EYE_RY//2), (EYE_RX,0))
def draw_thinking(ms):
    fb.fill(BG)
    so = int(math.sin(ms * 0.00393) * 8)
    for i in range(4):
        _line(EL + _THK_WAVE[i][0] + so, EY + _THK_WAVE[i][1], EL + _THK_WAVE[i+1][0] + so, EY + _THK_WAVE[i+1][1], WHITE, 5)
        _line(ER + _THK_WAVE[i][0] - so, EY + _THK_WAVE[i][1], ER + _THK_WAVE[i+1][0] - so, EY + _THK_WAVE[i+1][1], WHITE, 5)
    mp = math.sin(ms * 0.00314) * 0.5 + 0.5
    mw = 62 + int(mp * 10)
    _line(CX - (mw >> 1), MY, CX + (mw >> 1), MY, _grey(0.6 + mp * 0.4), 5)
    if (ms // 900) % 2 == 0:
        fb.fill_rect(CX + (mw >> 1) + 4, MY - 7, 4, 14, WHITE)

def draw_reconnecting(ms):
    fb.fill(BG)
    angle = (ms % 1200) * 0.005236
    rp = (ms % 1200) / 1200.0
    rb = max(0.0, 0.4 - rp * 0.4)
    if rb > 0.04:
        rc = _c(int(rb * 120), int(rb * 200), int(rb * 255))
        rr = int((EYE_RX + 4) * (1.0 + rp * 1.3))
        _arc(EL, EY, rr, rr, 0, 360, rc, 1, 24)
        _arc(ER, EY, rr, rr, 0, 360, rc, 1, 24)
    for ex, sign in ((EL, 1), (ER, -1)):
        for k in range(4):
            a = angle * sign + k * 1.570796
            al = EYE_RX + 2
            ca, sa = int(math.cos(a) * al), int(math.sin(a) * al)
            _line(ex + ca, EY + sa, ex - ca, EY - sa, WHITE, 5 if k % 2 == 0 else 3)
            ad = a + 0.785398
            ald = int(al * 0.72)
            cad, sad = int(math.cos(ad) * ald), int(math.sin(ad) * ald)
            _line(ex + cad, EY + sad, ex - cad, EY - sad, WHITE, 2)
    _line(CX - 32, MY, CX + 32, MY, _grey(0.5 + (math.sin(ms * 0.00449) * 0.5 + 0.5) * 0.5), 5)
    dp = (ms // 470) % 3
    for d in range(3):
        _fill_ellipse(CX - 10 + d * 10, MY + 20, 5, 5, WHITE if d == dp else DIM)

def draw_love(ms):
    fb.fill(BG)
    t = math.fmod(ms * 0.000909, 1.0)
    sc = 1.0 + t/0.14*0.25 if t < 0.14 else (1.25 - (t-0.14)/0.14*0.25 if t < 0.28 else (1.0 + (t-0.28)/0.14*0.14 if t < 0.42 else (1.14 - (t-0.42)/0.14*0.14 if t < 0.56 else 1.0)))
    rb = max(0.0, 0.45 - t * 0.45)
    if rb > 0.04:
        rc = _c(int(rb * 255), int(rb * 70), int(rb * 90))
        rr = int((EYE_RX + 4) * (1.0 + t * 1.6))
        _arc(EL, EY, rr, rr, 0, 360, rc, 1, 24)
        _arc(ER, EY, rr, rr, 0, 360, rc, 1, 24)
    _heart(EL, EY, int(32 * sc), WHITE, 4)
    _heart(ER, EY, int(32 * sc), WHITE, 4)
    _arc(CX, MY - 12 + int(math.sin(ms * 0.00314) * 3), 34, 17, 0, 180, WHITE, 5)
    for po, hx0, hy0 in ((0.5, 272, 142), (1.2, 48, 158), (1.8, 284, 108)):
        p = math.fmod(ms * 0.001 + po, 2.5)
        if 0.1 <= p <= 2.4:
            d = p / 2.5
            a = p / 0.4 if p < 0.4 else ((2.5 - p) / 0.5 if p > 2.0 else 1.0)
            av = int(max(0, min(1, a)) * 180)
            _heart(hx0 + int(12 * d), hy0 - int(60 * d), max(4, int(11 * (1.0 - d * 0.4))), _c(av, av // 5, av // 5), 2)

def draw_confused(ms):
    fb.fill(BG)
    to_x = int(math.sin(ms * 0.00157) * 6)
    for ex, cw in ((EL, 1), (ER, -1)):
        base = (ms % 2400) * 0.002618 * cw
        for rad in range(6, EYE_RX + 2, 5):
            frac = rad / (EYE_RX + 2)
            a0 = base + frac * 6.283185
            _arc(ex + to_x, EY, rad, rad, math.degrees(a0), math.degrees(a0 + 5.026548), WHITE, 2, 16)
        _fill_ellipse(ex + to_x, EY, 4, 4, WHITE)
    mw = int(66 * (1.0 - 0.09 * (math.sin(ms * 0.00314) + 1)))
    _line(CX - (mw >> 1) + to_x, MY, CX + (mw >> 1) + to_x, MY, WHITE, 5)
    qy = 42 - int(math.sin(ms * 0.00251) * 12)
    _arc(248, qy + 10, 10, 10, 200, 380, WHITE, 3, 16)
    _fill_ellipse(248, qy + 26, 3, 3, WHITE)
    _fill_ellipse(248, qy + 34, 3, 3, WHITE)

def draw_rizz(ms):
    fb.fill(BG)
    phase = math.fmod(ms * 0.000357, 1.0)
    winking = 0.10 < phase < 0.45
    lo = int(math.sin(ms * 0.00224) * 4)
    _rect(EL + lo, EY, EYE_RX * 2 + 4, 4 if winking else EYE_H, WHITE)
    _arc(ER + lo, EY + 6, EYE_RX + 2, int((EYE_RY - 4) * (0.55 if winking else 1.0)), 180, 360, WHITE, 5)
    _arc(CX + lo + 8, MY - 8, 24, 11, 0, 160, WHITE, 5)
    if 0.50 < phase < 0.80:
        gs = math.sin((phase - 0.50) / 0.30 * 3.141592)
        _sparkle(268, 56, int(11 * gs) + 1, gs, _c(int(255 * gs), int(255 * gs), int(100 * gs)))

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
}

EMOTION_LIST = list(EMOTIONS.keys())

# RP2040 PIO UART Driver (Zero IRQ, Zero Allocations in ISR, Polled FIFO)
if rp2:
    @rp2.asm_pio(autopush=True, push_thresh=8, in_shiftdir=rp2.PIO.SHIFT_RIGHT, fifo_join=rp2.PIO.JOIN_RX)
    def _pio_uart_rx():
        wait(0, pin, 0)
        set(x, 7)                 [10]
        label("bitloop")
        in_(pins, 1)
        jmp(x_dec, "bitloop")     [6]

    @rp2.asm_pio(sideset_init=rp2.PIO.OUT_HIGH, out_init=rp2.PIO.OUT_HIGH, out_shiftdir=rp2.PIO.SHIFT_RIGHT)
    def _pio_uart_tx():
        pull()
        set(x, 7)  .side(0)       [7]
        label("bitloop")
        out(pins, 1)              [6]
        jmp(x_dec, "bitloop")
        nop()      .side(1)       [7]

    class PIOUART:
        """Full-duplex UART using RP2040 PIO on arbitrary GPIO pins (e.g. GP14 TX, GP15 RX)."""
        def __init__(self, tx_pin, rx_pin, baud):
            self._rx_buf = bytearray()
            p_rx = machine.Pin(rx_pin, machine.Pin.IN, machine.Pin.PULL_UP)
            self._sm_rx = rp2.StateMachine(0, _pio_uart_rx, freq=8 * baud, in_base=p_rx)
            self._sm_rx.active(1)
            try:
                p_tx = machine.Pin(tx_pin, machine.Pin.OUT)
                self._sm_tx = rp2.StateMachine(1, _pio_uart_tx, freq=8 * baud, sideset_base=p_tx, out_base=p_tx)
                self._sm_tx.active(1)
            except Exception:
                self._sm_tx = None

        def poll(self):
            while self._sm_rx.rx_fifo() > 0:
                self._rx_buf.append((self._sm_rx.get() >> 24) & 0xFF)
                if len(self._rx_buf) > 128:
                    self._rx_buf = self._rx_buf[-64:]

        def any(self):
            self.poll()
            return len(self._rx_buf)

        def read(self, n=None):
            self.poll()
            if not self._rx_buf: return None
            if n is None or n >= len(self._rx_buf):
                res = bytes(self._rx_buf); self._rx_buf = bytearray(); return res
            res = bytes(self._rx_buf[:n]); self._rx_buf = self._rx_buf[n:]; return res

        def write(self, data):
            if not self._sm_tx: return 0
            if isinstance(data, str): data = data.encode("utf-8")
            for b in data: self._sm_tx.put(b)
            return len(data)

def init_uart(tx_pin=UART_TX_PIN, rx_pin=UART_RX_PIN, baudrate=UART_BAUD):
    for u_id in (0, 1):
        try:
            return machine.UART(u_id, baudrate=baudrate, tx=machine.Pin(tx_pin), rx=machine.Pin(rx_pin))
        except (ValueError, Exception):
            pass
    if rp2:
        try:
            print(f"Pico UART: Using PIO on GP{tx_pin}(TX) & GP{rx_pin}(RX)")
            return PIOUART(tx_pin, rx_pin, baudrate)
        except Exception as e:
            print("PIO UART init error:", e)
    try:
        return machine.UART(0, baudrate=baudrate)
    except Exception:
        return None

# Frame Render Dispatch
def _render_frame(emotion, ms, tft, uart=None):
    tft.show()
    draw_fn = EMOTIONS[emotion]
    for b in range(N_BANDS):
        fb.set_band(b)
        draw_fn(ms)
        tft.show_band(_band_buf)
        if uart and hasattr(uart, "poll"):
            uart.poll()

# Main Loop
def main():
    print("ADAM Pico TFT starting (Band renderer 38.4KB RAM)...")
    tft = ST7789()
    print("ST7789 OK")

    emotion = "idle"

    if TESTING_MODE:
        print("TESTING MODE — cycling emotions every 4 s")
        idx = 0
        last_sw = time.ticks_ms()
        start_ms = time.ticks_ms()

        while True:
            ms = time.ticks_diff(time.ticks_ms(), start_ms)
            if time.ticks_diff(time.ticks_ms(), last_sw) > 4000:
                idx = (idx + 1) % len(EMOTION_LIST)
                emotion = EMOTION_LIST[idx]
                print("→", emotion)
                last_sw = time.ticks_ms()

            _render_frame(emotion, ms, tft)
            time.sleep_ms(33)

    else:
        print(f"LIVE MODE — waiting for UART on GP{UART_RX_PIN} (TX: GP{UART_TX_PIN})")
        uart = init_uart(UART_TX_PIN, UART_RX_PIN, UART_BAUD)
        rxbuf = b""
        start_ms = time.ticks_ms()
        MAX_RXBUF_LEN = 128

        while True:
            ms = time.ticks_diff(time.ticks_ms(), start_ms)

            if uart and uart.any():
                chunk = uart.read(uart.any())
                if chunk:
                    rxbuf += chunk

                if len(rxbuf) > MAX_RXBUF_LEN:
                    rxbuf = b""

                while b"\n" in rxbuf:
                    line, rxbuf = rxbuf.split(b"\n", 1)
                    try:
                        cmd = line.decode("utf-8").strip().lower()
                    except UnicodeError:
                        continue
                    if cmd in EMOTIONS:
                        emotion = cmd
                        print("→", emotion)

            _render_frame(emotion, ms, tft, uart)
            time.sleep_ms(33)

main()
