"""
ADAM v32 — Raspberry Pi Pico Face Renderer  (FINAL)
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
  love | confused | rizz

Set TESTING_MODE = True to auto-cycle all emotions
without the ESP32-CAM connected.
"""

import machine, time, math, framebuf, gc

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
TESTING_MODE = True   # False = live UART from ESP32-CAM

W, H = 320, 240

# ─────────────────────────────────────────────────────────────
# PRE-ALLOCATE FRAME BUFFER  (153.6 KB — done once at boot)
# ─────────────────────────────────────────────────────────────
try:
    _buf = bytearray(W * H * 2)
except MemoryError:
    print("FATAL: cannot allocate framebuffer — reset")
    machine.reset()

fb = framebuf.FrameBuffer(_buf, W, H, framebuf.RGB565)

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

def _grey(b):           # brightness 0.0-1.0  → swapped RGB565
    v = int(max(0, min(1, b)) * 255)
    return _c(v, v, v)

# ─────────────────────────────────────────────────────────────
# FACE GEOMETRY  — scaled up for 2.4" 320×240
# Eyes spread wider, taller — mouth lower — fills the screen
# ─────────────────────────────────────────────────────────────
EL  = 90       # left  eye centre X
ER  = 230      # right eye centre X
EY  = 97       # eye centre Y
EYE_RX = 42    # eye ellipse half-width  (old 27 → +22 %)
EYE_RY = 42   # eye ellipse half-height
MY  = 178      # mouth centre Y  (old 148 → pushed down)
CX  = 160      # horizontal centre

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
        self._cmd(0x36); self._dat([0x60])     # MADCTL: rotated 180°
        self._cmd(0x3A); self._dat([0x55])     # 16-bit colour
        self._cmd(0x20)                        # Inversion OFF  ← hardware fix
        self._cmd(0x13)                        # Normal display on
        self._cmd(0x29); time.sleep_ms(50)     # Display on

    def show(self):
        # Set full-screen window then blast buffer
        self._cmd(0x2A); self._dat([0x00,0x00,(W-1)>>8,(W-1)&0xFF])
        self._cmd(0x2B); self._dat([0x00,0x00,(H-1)>>8,(H-1)&0xFF])
        self._cmd(0x2C)
        self._dc(1); self._cs(0)
        self._spi.write(_buf)
        self._cs(1)


# ─────────────────────────────────────────────────────────────
# DRAWING PRIMITIVES  (operate on global fb)
# ─────────────────────────────────────────────────────────────

def _line(x0, y0, x1, y1, col, t=4):
    """Thick Bresenham line — t=thickness drawn as parallel offsets."""
    dx = abs(x1-x0); dy = abs(y1-y0)
    steep = dy > dx
    h = t >> 1
    for d in range(-h, h+1):
        if steep:
            fb.line(x0+d, y0, x1+d, y1, col)
        else:
            fb.line(x0, y0+d, x1, y1+d, col)

def _arc(cx, cy, rx, ry, a0_deg, a1_deg, col, t=4, steps=48):
    """Thick ellipse arc from a0_deg to a1_deg (degrees, CW like PIL)."""
    prev = None
    for i in range(steps+1):
        a = math.radians(a0_deg + (a1_deg - a0_deg) * i / steps)
        x = int(cx + rx * math.cos(a))
        y = int(cy + ry * math.sin(a))
        if prev:
            _line(prev[0], prev[1], x, y, col, t)
        prev = (x, y)

def _ellipse_outline(cx, cy, rx, ry, col, t=4):
    """Thick ellipse outline — drawn as concentric single-pixel ellipses."""
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
    """Centred filled rounded-ish rectangle (pill shape)."""
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
    """Parametric heart.  sz~28 gives a nice large eye."""
    pts = []
    for i in range(61):
        a = 2*math.pi*i/60
        sc = sz/17.0
        x = int(cx + sc*16*(math.sin(a)**3))
        y = int(cy - sc*(13*math.cos(a)-5*math.cos(2*a)-2*math.cos(3*a)-math.cos(4*a)))
        pts.append((x, y))
    if fill:
        # scanline fill  (simple — draw horizontal spans)
        min_y = max(0, min(p[1] for p in pts))
        max_y = min(H-1, max(p[1] for p in pts))
        for yy in range(min_y, max_y+1):
            xs = [p[0] for p in pts if p[1] == yy]
            if xs:
                fb.fill_rect(min(xs), yy, max(xs)-min(xs)+1, 1, col)
    else:
        for i in range(len(pts)-1):
            _line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], col, t)


# ─────────────────────────────────────────────────────────────
# EMOTION RENDERERS
# Convention (matches PIL adam_tft.py):
#   ^ happy arc  → start=180, end=360  (top half)
#   ∪ sad arc    → start=0,   end=180  (bottom half)
# ─────────────────────────────────────────────────────────────

def draw_idle(ms):
    fb.fill(BG)
    blink   = (ms % 6000) < 140
    breathe = math.sin(ms * 0.001047)
    by      = int(breathe * 2)
    ey      = 3 if blink else 12       # eye pill height
    _rect(EL, EY+by, EYE_RX*2+4, ey, WHITE)
    _rect(ER, EY+by, EYE_RX*2+4, ey, WHITE)
    mw = int(30 * (1.0 + 0.04*breathe))
    _rect(CX, MY+by, mw, 6, DIM)

def draw_speaking(ms):
    fb.fill(BG)
    blink = (ms % 3800) < 100
    ph    = (ms % 420) / 420.0
    mw    = 64 if ph < 0.33 else (42 if ph < 0.66 else 22)
    ey    = 3 if blink else 12
    _rect(EL, EY, EYE_RX*2+4, ey, WHITE)
    _rect(ER, EY, EYE_RX*2+4, ey, WHITE)
    _rect(CX, MY, mw, 10, WHITE)

def draw_happy(ms):
    fb.fill(BG)
    blink  = (ms % 4000) < 120
    bounce = int(math.sin(ms * 0.00286) * 3)
    # sparkles
    for off, sx, sy, arm in [(0,50,55,9),(600,268,58,9),(1100,285,130,6)]:
        ts = (ms+off) % 2200
        if 300 < ts < 1400:
            br = (ts-300)/550.0 if ts < 850 else (1400-ts)/550.0
            _sparkle(sx, sy, int(arm*br+1), br)
    if blink:
        fb.fill_rect(EL-EYE_RX-2, EY, EYE_RX*2+4, 5, WHITE)
        fb.fill_rect(ER-EYE_RX-2, EY, EYE_RX*2+4, 5, WHITE)
    else:
        _arc(EL, EY+6, EYE_RX+2, EYE_RY-4, 180, 360, WHITE, 5)   # ^ left
        _arc(ER, EY+6, EYE_RX+2, EYE_RY-4, 180, 360, WHITE, 5)   # ^ right
    _arc(CX, MY-12+bounce, 34, 17, 0, 180, WHITE, 5)              # smile ∪

def draw_sad(ms):
    fb.fill(BG)
    by = int(math.sin(ms * 0.00157) * 5)
    _arc(EL, EY-6+by, EYE_RX+2, EYE_RY-4, 0, 180, WHITE, 5)     # ∪ droopy
    _arc(ER, EY-6+by, EYE_RX+2, EYE_RY-4, 0, 180, WHITE, 5)
    _arc(CX, MY+10+by, 32, 14, 180, 360, WHITE, 5)               # ^ frown
    # tears
    for po, ex in [(0.2, EL), (1.0, ER)]:
        tp = math.fmod(ms*0.001+po, 1.5)
        if tp < 0.55: continue
        d  = (tp-0.55)/0.95
        ty = EY+36+by+int(d*44)
        av = int(min(1, max(0, 1-(tp-0.55)/0.95)) * 200)
        tc = _c(av//4, av//3, av)
        _fill_ellipse(ex, ty, 6, 9, tc)

def draw_angry(ms):
    fb.fill(BG)
    t = ms % 100
    sx = -2 if t<25 else (2 if t<75 else 0)
    sy =  1 if t<50 else -1
    # slash eyes  \ /
    _line(EL-EYE_RX+sx, EY-EYE_RY+sy, EL+EYE_RX+sx, EY+EYE_RY+sy, WHITE, 6)
    _line(ER+EYE_RX+sx, EY-EYE_RY+sy, ER-EYE_RX+sx, EY+EYE_RY+sy, WHITE, 6)
    # zigzag mouth (scaled up)
    mp = [(100,MY),(115,MY-15),(130,MY),(145,MY+15),(160,MY),
          (175,MY-15),(190,MY),(205,MY+15),(220,MY)]
    _zigzag([(x+sx,y+sy) for x,y in mp], WHITE, 5)
    # veins
    vp = (ms%600)/600.0
    vb = vp*2 if vp<0.5 else (1-vp)*2
    vc = _c(int(200*vb+55), int(50*vb), int(50*vb))
    _line(EL-22, EY-38, EL-30, EY-52, vc, 2)
    _line(EL-28, EY-44, EL-36, EY-40, vc, 2)
    _line(ER+22, EY-38, ER+30, EY-52, vc, 2)
    _line(ER+28, EY-44, ER+36, EY-40, vc, 2)
    # heat shimmer
    ht = ms%1200
    if ht < 960:
        hd = ht/960.0
        hb = min(1.0, (1-abs(hd*2-1)) * 2)
        hc = _c(int(120*hb), int(60*hb), 0)
        hy = EY-6 - int(hd*32)
        fb.line(EL, hy, EL, hy-8, hc)
        fb.line(ER, hy, ER, hy-8, hc)

def draw_panic(ms):
    fb.fill(BG)
    t  = ms % 70
    sx = -3 if t<20 else (3 if t<40 else -1)
    sy =  1 if t<35 else -1
    # stress marks
    sf = (ms%550)/550.0
    sb = sf*2 if sf<0.5 else (1-sf)*2
    sc = _c(int(180*sb), int(210*sb), int(255*sb))
    _line(EL-46+sx, EY-22+sy, EL-56+sx, EY+sy, sc, 3)
    _line(EL-54+sx, EY-10+sy, EL-62+sx, EY-6+sy, sc, 2)
    _line(ER+46+sx, EY-22+sy, ER+56+sx, EY+sy, sc, 3)
    _line(ER+54+sx, EY-10+sy, ER+62+sx, EY-6+sy, sc, 2)
    # wide oval eyes
    _ellipse_outline(EL+sx, EY+sy, EYE_RX+3, EYE_RY+6, WHITE, 5)
    _ellipse_outline(ER+sx, EY+sy, EYE_RX+3, EYE_RY+6, WHITE, 5)
    # darting pupils
    pp = (ms%300)/300.0
    px = 0 if pp<0.45 else (5 if pp<0.55 else (-5 if pp<0.65 else 0))
    py = 0 if pp<0.45 else (-3 if pp<0.55 else (3 if pp<0.65 else 0))
    _fill_ellipse(EL+sx+px, EY+sy+py, 7, 7, WHITE)
    _fill_ellipse(ER+sx+px, EY+sy+py, 7, 7, WHITE)
    # zigzag mouth
    mz = [(108,MY),(122,MY-12),(138,MY),(154,MY+12),(170,MY),
          (186,MY-12),(202,MY),(214,MY+8),(220,MY+4)]
    _zigzag([(x+sx,y+sy) for x,y in mz], WHITE, 5)
    # sweat
    for po, ex in [(0.2, EL), (1.3, ER)]:
        sw = math.fmod(ms*0.001+po, 2.2)
        if sw < 0.35: continue
        d  = (sw-0.35)/1.85
        swy = EY-36 + int(d*55)
        av  = int(min(1.0, max(0.0, 1.0-abs(d-0.5)*2+0.1)) * 200)
        tc  = _c(av//4, av//3, av)
        _fill_ellipse(ex, swy, 5, 8, tc)

def draw_surprised(ms):
    fb.fill(BG)
    phase = math.fmod(ms*0.001, 2.8)
    es    = min(1.0, phase/0.18)
    erx   = int((EYE_RX+4)*es)
    ery   = int((EYE_RY+8)*es)
    # shock lines
    if 0.25 < phase < 1.9:
        lb = min(1.0,(phase-0.25)/0.2) * (((1.9-phase)/0.2) if phase>1.7 else 1.0)
        lc = _grey(lb*0.55)
        _line(EL-25,EY-28, EL-46,EY-42, lc, 2)
        _line(EL-32,EY-12, EL-56,EY-8,  lc, 2)
        _line(EL-26,EY+10, EL-50,EY+16, lc, 2)
        _line(ER+25,EY-28, ER+46,EY-42, lc, 2)
        _line(ER+32,EY-12, ER+56,EY-8,  lc, 2)
        _line(ER+26,EY+10, ER+50,EY+16, lc, 2)
    if erx > 2 and ery > 2:
        _ellipse_outline(EL, EY, erx, ery, WHITE, 5)
        _ellipse_outline(ER, EY, erx, ery, WHITE, 5)
    # O mouth
    if phase > 0.18:
        ms2 = min(1.0,(phase-0.18)/0.2)
        mrx = max(2, int(10*ms2))
        mry = max(2, int(9*ms2))
        _ellipse_outline(CX, MY, mrx, mry, WHITE, 4)

def draw_shy(ms):
    fb.fill(BG)
    blink = (ms%5000) < 120
    bi    = 0.35 + 0.3*(math.sin(ms*0.00251)*0.5+0.5)
    by    = int(math.sin(ms*0.00105)*6)
    # blush
    br = int(bi*200)
    bc = _c(br, br//3, br//2)
    _fill_ellipse(EL-16, EY+32+by, 26, 13, bc)
    _fill_ellipse(ER+16, EY+32+by, 26, 13, bc)
    # eyes
    if blink:
        fb.fill_rect(EL-EYE_RX-2, EY+by, EYE_RX*2+4, 5, WHITE)
        fb.fill_rect(ER-EYE_RX-2, EY+by, EYE_RX*2+4, 5, WHITE)
    else:
        _arc(EL, EY+6+by, EYE_RX+2, EYE_RY-4, 180, 360, WHITE, 5)
        _arc(ER, EY+6+by, EYE_RX+2, EYE_RY-4, 180, 360, WHITE, 5)
    # tiny smile
    _arc(CX, MY-4+by, 14, 11, 0, 180, WHITE, 5)
    # floating heart
    hf = math.fmod(ms*0.001+1.5, 3.0)
    if 0.1 < hf < 2.6:
        d  = hf/2.6
        hx = 278+int(10*d); hy = 88-int(36*d)
        av = int(min(1,max(0, hf/0.4 if hf<0.4 else ((2.6-hf)/0.4 if hf>2.2 else 1.0)))*160)
        hc = _c(av, av//6, av//5)
        _heart(hx, hy, 9, hc, fill=False, t=2)

def draw_sleep(ms):
    fb.fill(BG)
    by = int(math.sin(ms*0.00126)*2)
    _arc(EL, EY-4+by, EYE_RX+2, EYE_RY-4, 0, 180, WHITE, 5)   # closed arcs ∪
    _arc(ER, EY-4+by, EYE_RX+2, EYE_RY-4, 0, 180, WHITE, 5)
    # ZZZ: use framebuf.text() — 8×8 font, scale with fill_rect tricks
    for i, (zx,zy,sc) in enumerate([(258,74,3),(272,56,2),(282,42,1)]):
        zp = math.fmod((ms+i*500)*0.001, 4.0)
        if zp < 0.1 or zp > 3.8: continue
        za = min(1.0, zp/0.25 if zp<0.25 else ((4.0-zp)/0.5 if zp>3.5 else 1.0))
        zy2 = zy - int(zp*10)
        zc  = _grey(za*0.9)
        # Manual Z glyph scaled: 3 lines = top, diagonal, bottom
        sw = 8*sc; sh = 8*sc
        _line(zx,     zy2,    zx+sw, zy2,    zc, sc)
        _line(zx+sw,  zy2,    zx,    zy2+sh, zc, sc)
        _line(zx,     zy2+sh, zx+sw, zy2+sh, zc, sc)

def draw_thinking(ms):
    fb.fill(BG)
    so = int(math.sin(ms*0.00393)*8)
    # scanning wave eyes
    lw = [(EL-EYE_RX+so,EY),(EL-EYE_RX//2+so,EY-EYE_RY//2),
          (EL+so,EY),(EL+EYE_RX//2+so,EY+EYE_RY//2),(EL+EYE_RX+so,EY)]
    rw = [(ER-EYE_RX-so,EY),(ER-EYE_RX//2-so,EY-EYE_RY//2),
          (ER-so,EY),(ER+EYE_RX//2-so,EY+EYE_RY//2),(ER+EYE_RX-so,EY)]
    _zigzag(lw, WHITE, 5)
    _zigzag(rw, WHITE, 5)
    # flat mouth + blinking cursor
    mp = math.sin(ms*0.00314)*0.5+0.5
    mw = 62+int(mp*10)
    _line(CX-mw//2, MY, CX+mw//2, MY, _grey(0.6+mp*0.4), 5)
    if (ms//900)%2 == 0:
        fb.fill_rect(CX+mw//2+4, MY-7, 4, 14, WHITE)

def draw_reconnecting(ms):
    fb.fill(BG)
    angle = (ms%1200)/1200.0*2*math.pi
    # pulsing rings behind eyes
    rp = (ms%1200)/1200.0
    rb = max(0.0, 0.4-rp*0.4)
    if rb > 0.04:
        rc = _c(int(rb*120), int(rb*200), int(rb*255))
        rr = int((EYE_RX+4)*(1.0+rp*1.3))
        _arc(EL, EY, rr, rr, 0, 360, rc, 1, 40)
        _arc(ER, EY, rr, rr, 0, 360, rc, 1, 40)
    # spinning star eyes
    for ex, sign in [(EL, 1), (ER, -1)]:
        for k in range(4):
            a  = angle*sign + k*math.pi/2
            al = EYE_RX+2
            tw = 5 if k%2==0 else 3
            x0 = ex+int(math.cos(a)*al); y0 = EY+int(math.sin(a)*al)
            x1 = ex-int(math.cos(a)*al); y1 = EY-int(math.sin(a)*al)
            _line(x0,y0,x1,y1, WHITE, tw)
        # diagonal arms
        for k in range(4):
            a  = angle*sign + k*math.pi/2 + math.pi/4
            al = int((EYE_RX+2)*0.72)
            x0 = ex+int(math.cos(a)*al); y0 = EY+int(math.sin(a)*al)
            x1 = ex-int(math.cos(a)*al); y1 = EY-int(math.sin(a)*al)
            _line(x0,y0,x1,y1, WHITE, 2)
    # flat mouth
    mp = math.sin(ms*0.00449)*0.5+0.5
    _line(CX-32, MY, CX+32, MY, _grey(0.5+mp*0.5), 5)
    # loading dots
    dp = (ms//470)%3
    for d in range(3):
        dc = WHITE if d==dp else DIM
        _fill_ellipse(CX-10+d*10, MY+20, 5, 5, dc)

def draw_love(ms):
    fb.fill(BG)
    # heartbeat scale
    t = math.fmod(ms*0.000909, 1.0)
    sc = 1.0
    if   t < 0.14: sc = 1.0 + t/0.14*0.25
    elif t < 0.28: sc = 1.25 - (t-0.14)/0.14*0.25
    elif t < 0.42: sc = 1.0  + (t-0.28)/0.14*0.14
    elif t < 0.56: sc = 1.14 - (t-0.42)/0.14*0.14
    # pulse rings
    rp = math.fmod(ms*0.000909, 1.0)
    rb = max(0.0, 0.45-rp*0.45)
    if rb > 0.04:
        rc = _c(int(rb*255), int(rb*70), int(rb*90))
        rr = int((EYE_RX+4)*(1.0+rp*1.6))
        _arc(EL, EY, rr, rr, 0, 360, rc, 1, 40)
        _arc(ER, EY, rr, rr, 0, 360, rc, 1, 40)
    # heart eyes — outline (fill is slow on Pico; outline looks great)
    _heart(EL, EY, int(32*sc), WHITE, fill=False, t=4)
    _heart(ER, EY, int(32*sc), WHITE, fill=False, t=4)
    # smile
    sb = int(math.sin(ms*0.00314)*3)
    _arc(CX, MY-12+sb, 34, 17, 0, 180, WHITE, 5)
    # floating hearts
    for po, hx0, hy0 in [(0.5,272,142),(1.2,48,158),(1.8,284,108)]:
        p = math.fmod(ms*0.001+po, 2.5)
        if p < 0.1 or p > 2.4: continue
        d  = p/2.5
        hx = hx0+int(12*d); hy = hy0-int(60*d)
        a  = p/0.4 if p<0.4 else ((2.5-p)/0.5 if p>2.0 else 1.0)
        av = int(max(0,min(1,a))*180)
        hs = max(4, int(11*(1.0-d*0.4)))
        _heart(hx, hy, hs, _c(av, av//5, av//5), fill=False, t=2)

def draw_confused(ms):
    fb.fill(BG)
    to_x = int(math.sin(ms*0.00157)*6)
    # spiral eyes — concentric arcs rotating
    for ex, cw in [(EL, True), (ER, False)]:
        base = (ms%2400)/2400.0*2*math.pi * (1 if cw else -1)
        for rad in range(6, EYE_RX+2, 5):
            frac = rad/(EYE_RX+2)
            a0 = base + frac*2*math.pi
            a1 = a0 + math.pi*1.6
            _arc(ex+to_x, EY, rad, rad, math.degrees(a0), math.degrees(a1), WHITE, 2, 20)
        _fill_ellipse(ex+to_x, EY, 4, 4, WHITE)
    # flat mouth
    mp = math.sin(ms*0.00314)*0.5+0.5
    mw = int(66*(1.0-0.18*mp))
    _line(CX-mw//2+to_x, MY, CX+mw//2+to_x, MY, WHITE, 5)
    # bouncing ?
    qy = 42 - int(math.sin(ms*0.00251)*12)
    # draw ? manually: arc + dot
    _arc(248, qy+10, 10, 10, 200, 380, WHITE, 3, 20)   # top curve
    _fill_ellipse(248, qy+26, 3, 3, WHITE)              # stem top
    _fill_ellipse(248, qy+34, 3, 3, WHITE)              # dot

def draw_rizz(ms):
    fb.fill(BG)
    phase   = math.fmod(ms*0.000357, 1.0)
    winking = 0.10 < phase < 0.45
    lo      = int(math.sin(ms*0.00224)*4)
    # left eye: flat wink line (always horizontal — the wink IS the eye)
    wh = 4 if winking else 12
    _rect(EL+lo, EY, EYE_RX*2+4, wh, WHITE)
    # right eye: confident ^ arc, squints during wink
    rh_scale = 0.55 if winking else 1.0
    _arc(ER+lo, EY+6, EYE_RX+2, int((EYE_RY-4)*rh_scale), 180, 360, WHITE, 5)
    # smirk — asymmetric
    _arc(CX+lo+8, MY-8, 24, 11, 0, 160, WHITE, 5)
    # gleam
    if 0.50 < phase < 0.80:
        gp = (phase-0.50)/0.30
        gs = math.sin(gp*math.pi)
        _sparkle(268, 56, int(11*gs)+1, gs, _c(int(255*gs), int(255*gs), int(100*gs)))


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
}

EMOTION_LIST = list(EMOTIONS.keys())

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("ADAM Pico TFT starting...")
    tft = ST7789()
    print("ST7789 OK")

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

            EMOTIONS[emotion](ms)
            tft.show()
            gc.collect()
            time.sleep_ms(33)   # ~30 FPS

    else:
        print("LIVE MODE — waiting for UART emotion commands on GP1")
        # UART0 — RX only matters here (Pico never talks back to the
        # ESP32-CAM relay), but MicroPython's UART constructor still
        # wants both tx/rx pins named; GP0 is wired but unused.
        uart     = machine.UART(0, baudrate=115200,
                                tx=machine.Pin(0), rx=machine.Pin(1))
        rxbuf    = b""
        start_ms = time.ticks_ms()

        # ── UART buffer safety limits ──────────────────────────────
        # Without a cap, a corrupted/partial transmission that never
        # contains a newline (e.g. the ESP32-CAM's relay wire glitches
        # mid-send, or gets disconnected/reconnected during a reflash)
        # would make rxbuf grow forever, silently eating RAM until the
        # Pico OOMs. This caps it and drops the buffer if it ever grows
        # unreasonably large without finding a line terminator.
        MAX_RXBUF_LEN = 256   # emotion words are short; way more than enough

        while True:
            ms = time.ticks_diff(time.ticks_ms(), start_ms)

            # ── Non-blocking UART read — hardened against the crash ──
            # ORIGINAL BUG: uart.read(uart.any()) can return None if the
            # UART hardware buffer is drained/cleared between the
            # uart.any() check and the read() call itself (a real race,
            # not hypothetical — happens under load, e.g. right as the
            # ESP32-CAM is also mid-transmission on its own send). The
            # old code did `rxbuf += uart.read(uart.any())` unguarded,
            # and separately, .decode("utf-8", "ignore") in MicroPython
            # does not reliably swallow every malformed byte sequence
            # the way it does in CPython — a corrupted/partial UART
            # read can still raise UnicodeError even with "ignore" set.
            # That crash killed main() entirely, which is why the whole
            # face renderer died instead of just skipping one bad line.
            if uart.any():
                chunk = uart.read(uart.any())
                if chunk:   # guards the None-return race condition
                    rxbuf += chunk

                if len(rxbuf) > MAX_RXBUF_LEN:
                    # No newline ever showed up and the buffer grew
                    # unreasonably large — this is garbage/corrupted
                    # data, not a real emotion command. Drop it and
                    # start clean rather than accumulating forever.
                    print("⚠️  UART rxbuf overflow — discarding", len(rxbuf), "bytes")
                    rxbuf = b""

                while b"\n" in rxbuf:
                    line, rxbuf = rxbuf.split(b"\n", 1)
                    try:
                        cmd = line.decode("utf-8").strip().lower()
                    except UnicodeError:
                        # Corrupted/partial bytes that don't form valid
                        # UTF-8 — skip this one line only, keep running.
                        # This is the actual fix for the crash: one bad
                        # line no longer takes down the whole renderer.
                        print("⚠️  Dropped malformed UART line (non-UTF8 bytes)")
                        continue
                    if cmd in EMOTIONS:
                        emotion = cmd
                        print("→", emotion)
                    elif cmd:
                        # Non-empty but unrecognized — e.g. a truncated
                        # word from a split transmission. Falls back to
                        # whatever emotion is already showing instead
                        # of crashing or silently accepting garbage.
                        print("⚠️  Unknown emotion command:", repr(cmd))

            EMOTIONS[emotion](ms)
            tft.show()
            gc.collect()
            time.sleep_ms(33)

main()