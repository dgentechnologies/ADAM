"""
adam_tft.py  —  ADAM v3 TFT Emotion Renderer
Dgen Technologies Pvt. Ltd.  |  May 2026

Drives an ILI9341 320×240 TFT display directly from Raspberry Pi Zero 2W.
Replaces the ESP32-CAM display role. All drawing is done in Python via PIL,
pushed to the display over hardware SPI using luma.lcd.

Dependencies (install via setup steps in guide):
    luma.lcd, Pillow, RPi.GPIO, spidev

Hardware SPI pins (Pi Zero 2W physical):
    MOSI  → Pin 19  (GPIO10)
    SCLK  → Pin 23  (GPIO11)
    CS    → Pin 24  (GPIO8  / CE0)
    DC    → Pin 18  (GPIO24)
    RST   → Pin 22  (GPIO25)
    VCC   → Pin 17  (3.3V)
    GND   → Pin 20  (GND)
    LED   → Pin 1   (3.3V) or PWM GPIO for brightness control

Integration with adamV25.py (add near top):
    from adam_tft import TFTEmotionRenderer
    tft = TFTEmotionRenderer()
    tft.start()

Then wherever you set ADAM's emotion:
    tft.set_emotion("happy")     # matches emotion name strings exactly

To stop cleanly on exit:
    tft.stop()
"""

import time
import math
import threading
from PIL import Image, ImageDraw, ImageFont

# ── Try to import luma.lcd; fall back to headless preview mode ──────────────
try:
    from luma.core.interface.serial import spi
    from luma.lcd.device import ili9341
    _HARDWARE = True
except ImportError:
    _HARDWARE = False
    print("[adam_tft] luma.lcd not found — running in headless/preview mode")

# ─── Display constants ───────────────────────────────────────────────────────
W, H       = 320, 240
FPS_TARGET = 30
FRAME_MS   = 1.0 / FPS_TARGET

# ─── Color palette (RGB tuples) ─────────────────────────────────────────────
BG         = (0,   0,   0)
WHITE      = (255, 255, 255)
DIM        = (130, 130, 130)
DIM2       = (60,  60,  60)
PINK       = (255, 100, 120)
BLUE       = (80,  140, 255)
RED        = (230, 40,  40)
YELLOW     = (255, 235, 80)
ORANGE     = (255, 140, 0)

# ─── Face geometry ───────────────────────────────────────────────────────────
EYE_L_X = 105
EYE_R_X = 215
EYE_Y   = 88
MOUTH_Y = 148
CX      = 160        # horizontal center

# ─── Emotion names ───────────────────────────────────────────────────────────
EMOTIONS = [
    "idle", "speaking", "happy", "sad", "angry",
    "panic", "surprised", "shy", "sleep", "thinking",
    "reconnecting", "love", "confused", "rizz"
]


# ════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS  (all operate on a PIL ImageDraw object)
# ════════════════════════════════════════════════════════════════════════════

def _lerp(a, b, t):
    return a + (b - a) * t

def _grey(b):
    """brightness 0.0-1.0 → RGB grey tuple"""
    v = int(max(0, min(1, b)) * 255)
    return (v, v, v)

def _alpha_blend(color, alpha, bg=BG):
    """Blend color onto bg with alpha 0.0-1.0"""
    return tuple(int(bg[i] + (color[i] - bg[i]) * alpha) for i in range(3))

def _arc_points(cx, cy, rx, ry, start_deg, end_deg, steps=60):
    """Return list of (x,y) for an ellipse arc, going clockwise."""
    pts = []
    for i in range(steps + 1):
        a = math.radians(start_deg + (end_deg - start_deg) * i / steps)
        pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return pts

def _draw_arc(draw, cx, cy, rx, ry, start_deg, end_deg, color, thick=4):
    """Draw a thick ellipse arc using polyline."""
    pts = _arc_points(cx, cy, rx, ry, start_deg, end_deg)
    if len(pts) >= 2:
        for offset in range(-(thick // 2), thick // 2 + 1):
            shifted = [(x, y + offset) for x, y in pts]
            draw.line(shifted, fill=color, width=1)
            shifted2 = [(x + offset, y) for x, y in pts]
            draw.line(shifted2, fill=color, width=1)

def _draw_line(draw, x0, y0, x1, y1, color, thick=4):
    draw.line([(x0, y0), (x1, y1)], fill=color, width=thick)

def _rect_eye(draw, cx, cy, w, h, color):
    draw.rounded_rectangle(
        [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2],
        radius=h // 2, fill=color
    )

def _sparkle(draw, sx, sy, arm, bright, color=None):
    if bright <= 0.02:
        return
    if color is None:
        color = _grey(bright)
    da = int(arm * 0.7)
    c2 = _alpha_blend(color, 0.55)
    draw.line([(sx, sy - arm), (sx, sy + arm)], fill=color, width=2)
    draw.line([(sx - arm, sy), (sx + arm, sy)], fill=color, width=2)
    draw.line([(sx - da, sy - da), (sx + da, sy + da)], fill=c2, width=1)
    draw.line([(sx + da, sy - da), (sx - da, sy + da)], fill=c2, width=1)

def _heart_fill(draw, cx, cy, size, color):
    """Parametric filled heart."""
    pts = []
    for i in range(60):
        t = 2 * math.pi * i / 60
        sc = size / 17.0
        x = cx + sc * 16 * (math.sin(t) ** 3)
        y = cy - sc * (13 * math.cos(t) - 5 * math.cos(2*t)
                       - 2 * math.cos(3*t) - math.cos(4*t))
        pts.append((x, y))
    draw.polygon(pts, fill=color)

def _heart_outline(draw, cx, cy, size, color, thick=3):
    """Parametric outline heart."""
    pts = []
    for i in range(61):
        t = 2 * math.pi * i / 60
        sc = size / 17.0
        x = cx + sc * 16 * (math.sin(t) ** 3)
        y = cy - sc * (13 * math.cos(t) - 5 * math.cos(2*t)
                       - 2 * math.cos(3*t) - math.cos(4*t))
        pts.append((x, y))
    draw.line(pts, fill=color, width=thick)


# ════════════════════════════════════════════════════════════════════════════
# EMOTION DRAW FUNCTIONS
# Convention: PIL arc angles → 0°=right(east) 90°=down(south) clockwise.
#   Top-half arch (^) : start=180, end=360
#   Bottom-half arch (∪): start=0, end=180
# ════════════════════════════════════════════════════════════════════════════

def _draw_idle(img, ms):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, H], fill=BG)

    blink   = (ms % 6000) < 130
    breathe = math.sin(ms * 0.001047)
    by      = int(breathe * 1.5)

    if blink:
        draw.rectangle([EYE_L_X - 21, EYE_Y + by - 1, EYE_L_X + 21, EYE_Y + by + 2], fill=WHITE)
        draw.rectangle([EYE_R_X - 21, EYE_Y + by - 1, EYE_R_X + 21, EYE_Y + by + 2], fill=WHITE)
    else:
        _rect_eye(draw, EYE_L_X, EYE_Y + by, 42, 10, WHITE)
        _rect_eye(draw, EYE_R_X, EYE_Y + by, 42, 10, WHITE)

    m_scale = 1.0 + 0.04 * breathe
    mw = int(24 * m_scale)
    draw.rounded_rectangle(
        [CX - mw // 2, MOUTH_Y + by, CX + mw // 2, MOUTH_Y + by + 5],
        radius=2, fill=DIM
    )


def _draw_speaking(img, ms):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, H], fill=BG)

    blink = (ms % 3800) < 100
    ph    = (ms % 420) / 420.0
    mw    = 52 if ph < 0.33 else (34 if ph < 0.66 else 18)

    if blink:
        draw.rectangle([EYE_L_X - 21, EYE_Y - 1, EYE_L_X + 21, EYE_Y + 2], fill=WHITE)
        draw.rectangle([EYE_R_X - 21, EYE_Y - 1, EYE_R_X + 21, EYE_Y + 2], fill=WHITE)
    else:
        _rect_eye(draw, EYE_L_X, EYE_Y, 42, 10, WHITE)
        _rect_eye(draw, EYE_R_X, EYE_Y, 42, 10, WHITE)

    draw.rounded_rectangle(
        [CX - mw // 2, MOUTH_Y - 3, CX + mw // 2, MOUTH_Y + 5],
        radius=4, fill=WHITE
    )


def _draw_happy(img, ms):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, H], fill=BG)

    blink  = (ms % 4000) < 110
    bounce = math.sin(ms * 0.002856) * 2.5

    # Sparkles
    for offset, sx, sy, arm in [(0, 58, 55, 7), (600, 262, 60, 7), (1100, 280, 128, 5)]:
        t_s = (ms + offset) % 2200
        if 300 < t_s < 1400:
            bright = (t_s - 300) / 550.0 if t_s < 850 else (1400 - t_s) / 550.0
            c = (int(bright * 255), int(bright * 255), int(bright * 0.6 * 255))
            _sparkle(draw, sx, sy, arm, bright, c)

    if blink:
        draw.rectangle([EYE_L_X - 27, EYE_Y, EYE_L_X + 27, EYE_Y + 4], fill=WHITE)
        draw.rectangle([EYE_R_X - 27, EYE_Y, EYE_R_X + 27, EYE_Y + 4], fill=WHITE)
    else:
        _draw_arc(draw, EYE_L_X, EYE_Y + 5, 27, 22, 180, 360, WHITE, 5)  # ^ arch
        _draw_arc(draw, EYE_R_X, EYE_Y + 5, 27, 22, 180, 360, WHITE, 5)

    my = MOUTH_Y + int(bounce)
    _draw_arc(draw, CX, my - 10, 28, 14, 0, 180, WHITE, 4)          # smile ∪


def _draw_sad(img, ms):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, H], fill=BG)

    sag = math.sin(ms * 0.001571) * 4.0
    by  = int(sag)

    _draw_arc(draw, EYE_L_X, EYE_Y - 5 + by, 27, 22, 0, 180, WHITE, 5)   # ∪ droopy
    _draw_arc(draw, EYE_R_X, EYE_Y - 5 + by, 27, 22, 0, 180, WHITE, 5)
    _draw_arc(draw, CX, MOUTH_Y + 8 + by, 26, 12, 180, 360, WHITE, 4)    # ^ frown

    for phase_offset, ex in [(0.2, EYE_L_X), (1.0, EYE_R_X)]:
        tp = math.fmod(ms * 0.001 + phase_offset, 1.5)
        if tp < 0.55:
            continue
        d  = (tp - 0.55) / 0.95
        ty = EYE_Y + 28 + int(d * 35)
        a  = 1.0 if tp < 1.2 else (1.5 - tp) / 0.3
        a  = max(0, min(1, a))
        av = int(a * 200)
        tc = (av // 3, av // 2, av)                          # blue-tinted tear
        draw.ellipse([ex - 5, ty - 7, ex + 5, ty + 7], fill=tc)
        if d > 0.08:
            draw.line([(ex - 1, ty + 6), (ex - 2, ty + 7 + int(d * 16))],
                      fill=tc, width=2)


def _draw_angry(img, ms):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, H], fill=BG)

    t  = ms % 100
    sx = -2 if t < 25 else (2 if t < 75 else 0)
    sy = 1 if t < 50 else -1

    _draw_line(draw, 82+sx, 72+sy, 128+sx, 96+sy, WHITE, 5)
    _draw_line(draw, 238+sx, 72+sy, 192+sx, 96+sy, WHITE, 5)

    mp = [(108,142),(120,131),(133,142),(146,153),
          (159,142),(172,131),(185,142),(197,153),(208,142)]
    for i in range(8):
        _draw_line(draw, mp[i][0]+sx, mp[i][1]+sy,
                   mp[i+1][0]+sx, mp[i+1][1]+sy, WHITE, 4)

    vp = (ms % 600) / 600.0
    vb = vp * 2 if vp < 0.5 else (1 - vp) * 2
    vc = (int(200*vb+55), int(60*vb), int(60*vb))
    draw.line([(88,58),(82,50),(86,42),(94,44)], fill=vc, width=2)
    draw.line([(232,58),(238,50),(234,42),(226,44)], fill=vc, width=2)

    ht = ms % 1200
    if ht < 960:
        hd = ht / 960.0
        hb = hd/0.3 if hd < 0.3 else ((1-hd)/0.3 if hd > 0.7 else 1.0)
        hb = max(0, min(1, hb))
        hc = (int(160*hb), int(80*hb), 0)
        hy = EYE_Y - 4 - int(hd * 26)
        draw.line([(100, hy),(100, hy-7)], fill=hc, width=1)
        draw.line([(108, hy-3),(107, hy-9)], fill=hc, width=1)
        draw.line([(220, hy),(220, hy-7)], fill=hc, width=1)
        draw.line([(212, hy-3),(213, hy-9)], fill=hc, width=1)


def _draw_panic(img, ms):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, H], fill=BG)

    t  = ms % 70
    sx = -2 if t < 20 else (2 if t < 40 else -1)
    sy = 1 if t < 35 else -1

    sf = (ms % 550) / 550.0
    sb = sf * 2 if sf < 0.5 else (1 - sf) * 2
    sc = (int(200*sb), int(230*sb), int(255*sb))
    draw.line([(64,70),(57,79),(61,88)], fill=sc, width=2)
    draw.line([(56,76),(52,78),(55,83)], fill=sc, width=1)
    draw.line([(256,70),(263,79),(259,88)], fill=sc, width=2)
    draw.line([(262,76),(268,78),(265,83)], fill=sc, width=1)

    for ex in [EYE_L_X, EYE_R_X]:
        for d in range(3):
            draw.ellipse([ex+sx-23+d, EYE_Y+sy-29+d,
                          ex+sx+23-d, EYE_Y+sy+29-d], outline=WHITE, width=1)

    pp = (ms % 300) / 300.0
    px2 = 0 if pp < 0.45 else (4 if pp < 0.55 else (-4 if pp < 0.65 else 0))
    py2 = 0 if pp < 0.45 else (-2 if pp < 0.55 else (2 if pp < 0.65 else 0))
    draw.ellipse([EYE_L_X+sx+px2-5, EYE_Y+sy+py2-5,
                  EYE_L_X+sx+px2+5, EYE_Y+sy+py2+5], fill=WHITE)
    draw.ellipse([EYE_R_X+sx+px2-5, EYE_Y+sy+py2-5,
                  EYE_R_X+sx+px2+5, EYE_Y+sy+py2+5], fill=WHITE)

    mz = [(128,143),(138,135),(148,143),(158,151),
          (168,143),(178,135),(188,143),(196,149),(200,145)]
    for i in range(8):
        _draw_line(draw, mz[i][0]+sx, mz[i][1]+sy,
                   mz[i+1][0]+sx, mz[i+1][1]+sy, WHITE, 4)

    for phase_off, ex in [(0.2, EYE_L_X), (1.3, EYE_R_X)]:
        sw = math.fmod(ms * 0.001 + phase_off, 2.2)
        if sw < 0.35:
            continue
        d = (sw - 0.35) / 1.85
        swy = EYE_Y - 28 + int(d * 44)
        a = min(1.0, (sw-0.35)/0.1) if sw < 0.45 else (max(0,(2.2-sw)/0.2) if sw > 2.0 else 1.0)
        av = int(a * 200)
        tc = (av // 4, av // 3, av)
        draw.ellipse([ex-4, swy-6, ex+4, swy+6], fill=tc)


def _draw_surprised(img, ms):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, H], fill=BG)

    phase = math.fmod(ms * 0.001, 2.8)
    eye_s = min(1.0, phase / 0.18)
    erx   = int(23 * eye_s)
    ery   = int(31 * eye_s)

    if phase > 0.25 and phase < 1.9:
        lb = min(1.0, (phase-0.25)/0.2) * (((1.9-phase)/0.2) if phase > 1.7 else 1.0)
        lc = _grey(lb * 0.55)
        draw.line([(68,52),(48,38)], fill=lc, width=2)
        draw.line([(60,65),(36,60)], fill=lc, width=2)
        draw.line([(72,78),(50,82)], fill=lc, width=2)
        draw.line([(252,52),(272,38)], fill=lc, width=2)
        draw.line([(260,65),(284,60)], fill=lc, width=2)
        draw.line([(248,78),(270,82)], fill=lc, width=2)

    if erx > 2 and ery > 2:
        for d in range(3):
            draw.ellipse([EYE_L_X-erx+d, EYE_Y-ery+d, EYE_L_X+erx-d, EYE_Y+ery-d],
                         outline=WHITE, width=1)
            draw.ellipse([EYE_R_X-erx+d, EYE_Y-ery+d, EYE_R_X+erx-d, EYE_Y+ery-d],
                         outline=WHITE, width=1)
        if phase < 0.22:
            gb = (0.22 - phase) / 0.22 * 0.35
            gc = _grey(gb)
            draw.ellipse([EYE_L_X+2, EYE_Y-16, EYE_L_X+14, EYE_Y-4], fill=gc)
            draw.ellipse([EYE_R_X+2, EYE_Y-16, EYE_R_X+14, EYE_Y-4], fill=gc)

    if phase > 0.18:
        ms2 = min(1.0, (phase - 0.18) / 0.2)
        mry = max(2, int(7 * ms2))
        draw.ellipse([CX-8, MOUTH_Y-3-mry, CX+8, MOUTH_Y-3+mry], outline=WHITE, width=3)


def _draw_shy(img, ms):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, H], fill=BG)

    blink  = (ms % 5000) < 110
    blush_i = 0.35 + 0.3 * (math.sin(ms * 0.002513) * 0.5 + 0.5)
    by     = int(math.sin(ms * 0.001047) * 5.0)

    for r in range(22, 0, -4):
        ratio = r / 22.0
        ri = int(blush_i * ratio * 200)
        bc = _alpha_blend(PINK, ri / 255.0)
        draw.ellipse([82-r, 116+by-r//2, 82+r, 116+by+r//2], fill=bc)
        draw.ellipse([238-r, 116+by-r//2, 238+r, 116+by+r//2], fill=bc)

    if blink:
        draw.rectangle([EYE_L_X-27, EYE_Y+by, EYE_L_X+27, EYE_Y+by+4], fill=WHITE)
        draw.rectangle([EYE_R_X-27, EYE_Y+by, EYE_R_X+27, EYE_Y+by+4], fill=WHITE)
    else:
        _draw_arc(draw, EYE_L_X, EYE_Y+5+by, 27, 22, 180, 360, WHITE, 5)
        _draw_arc(draw, EYE_R_X, EYE_Y+5+by, 27, 22, 180, 360, WHITE, 5)

    _draw_arc(draw, CX, MOUTH_Y-2+by, 10, 8, 0, 180, WHITE, 4)

    hf = math.fmod(ms * 0.001 + 1.5, 3.0)
    if 0.1 < hf < 2.6:
        d  = hf / 2.6
        hx = 272 + int(8*d)
        hy = 78  - int(28*d)
        a  = hf/0.4 if hf < 0.4 else ((2.6-hf)/0.4 if hf > 2.2 else 1.0)
        av = int(max(0, min(1, a)) * 160)
        _heart_outline(draw, hx, hy, 7, (av, av//5, av//4), 2)


def _draw_sleep(img, ms):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, H], fill=BG)

    by = int(math.sin(ms * 0.001257) * 2.0)
    _draw_arc(draw, EYE_L_X, EYE_Y-5+by, 27, 22, 0, 180, WHITE, 5)
    _draw_arc(draw, EYE_R_X, EYE_Y-5+by, 27, 22, 0, 180, WHITE, 5)

    sizes  = [3, 2, 1]
    offsets = [0, 500, 1000]
    try:
        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 28)
        font_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 20)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 14)
        fonts = [font_lg, font_md, font_sm]
    except Exception:
        fonts = [None, None, None]

    for i, (sz, off) in enumerate(zip(sizes, offsets)):
        zp = math.fmod((ms + off) * 0.001, 4.0)
        if zp < 0.1 or zp > 3.8:
            continue
        za = zp/0.25 if zp < 0.25 else ((4.0-zp)/0.5 if zp > 3.5 else 1.0)
        za = max(0, min(1, za))
        zy = 76 - i*18 - int(zp * 9.5)
        zx = 250 + int(zp * 3.5)
        zc = _grey(za * 0.85)
        if fonts[i]:
            draw.text((zx, zy), "z", font=fonts[i], fill=zc)
        else:
            draw.text((zx, zy), "z", fill=zc)


def _draw_thinking(img, ms):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, H], fill=BG)

    so = int(math.sin(ms * 0.003927) * 6.0)

    lw = [(70,88),(80,78),(92,88),(104,98),(116,88),(128,78),(138,88)]
    rw = [(180,88),(192,78),(204,88),(216,98),(228,88),(240,78),(252,88)]
    for i in range(6):
        _draw_line(draw, lw[i][0]+so, lw[i][1], lw[i+1][0]+so, lw[i+1][1], WHITE, 4)
        _draw_line(draw, rw[i][0]-so, rw[i][1], rw[i+1][0]-so, rw[i+1][1], WHITE, 4)

    mp = math.sin(ms * 0.003142) * 0.5 + 0.5
    mw = 50 + int(mp * 8)
    bc = _grey(0.6 + mp * 0.4)
    draw.rounded_rectangle([CX-mw//2, MOUTH_Y-2, CX+mw//2, MOUTH_Y+2], radius=2, fill=bc)

    if (ms // 900) % 2 == 0:
        draw.rectangle([CX+mw//2+3, MOUTH_Y-6, CX+mw//2+6, MOUTH_Y+6], fill=WHITE)


def _draw_reconnecting(img, ms):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, H], fill=BG)

    angle = (ms % 1200) / 1200.0 * 2 * math.pi
    ring_p = (ms % 1200) / 1200.0

    rs = 1.0 + ring_p * 1.3
    rb = max(0.0, 0.4 - ring_p * 0.4)
    if rb > 0.04:
        rc = (int(rb*160), int(rb*230), int(rb*255))
        rr = int(22 * rs)
        draw.ellipse([EYE_L_X-rr, EYE_Y-rr, EYE_L_X+rr, EYE_Y+rr], outline=rc, width=1)
        draw.ellipse([EYE_R_X-rr, EYE_Y-rr, EYE_R_X+rr, EYE_Y+rr], outline=rc, width=1)

    arms = [0, math.pi/2, math.pi, 3*math.pi/2]
    for arm_a in arms:
        al = 22
        for ex, base_a in [(EYE_L_X, angle+arm_a), (EYE_R_X, -angle+arm_a)]:
            x0 = ex + int(math.cos(base_a) * al)
            y0 = EYE_Y + int(math.sin(base_a) * al)
            x1 = ex - int(math.cos(base_a) * al)
            y1 = EYE_Y - int(math.sin(base_a) * al)
            draw.line([(x0,y0),(x1,y1)], fill=WHITE, width=4 if arms.index(arm_a) % 2 == 0 else 2)

    mp = math.sin(ms * 0.004488) * 0.5 + 0.5
    mw = 60
    draw.rounded_rectangle([CX-mw//2, MOUTH_Y-2, CX+mw//2, MOUTH_Y+2], radius=2,
                            fill=_grey(0.5 + mp * 0.5))

    dp = (ms // 470) % 3
    for d in range(3):
        cx_ = CX - 10 + d * 10
        c_  = _grey(1.0 if d == dp else 0.15)
        draw.ellipse([cx_-4, MOUTH_Y+18, cx_+4, MOUTH_Y+26], fill=c_)


def _draw_love(img, ms):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, H], fill=BG)

    t  = math.fmod(ms * 0.000909, 1.0)
    sc = 1.0
    if   t < 0.14: sc = 1.0 + t/0.14 * 0.22
    elif t < 0.28: sc = 1.22 - (t-0.14)/0.14 * 0.22
    elif t < 0.42: sc = 1.0  + (t-0.28)/0.14 * 0.12
    elif t < 0.56: sc = 1.12 - (t-0.42)/0.14 * 0.12

    rp = math.fmod(ms * 0.000909, 1.0)
    rb = max(0.0, 0.45 - rp * 0.45)
    if rb > 0.04:
        rc = (int(rb*255), int(rb*80), int(rb*100))
        rr = int(24 * (1.0 + rp * 1.5))
        draw.ellipse([EYE_L_X-rr, EYE_Y-rr, EYE_L_X+rr, EYE_Y+rr], outline=rc, width=1)
        draw.ellipse([EYE_R_X-rr, EYE_Y-rr, EYE_R_X+rr, EYE_Y+rr], outline=rc, width=1)

    _heart_fill(draw, EYE_L_X, EYE_Y, int(28 * sc), WHITE)
    _heart_fill(draw, EYE_R_X, EYE_Y, int(28 * sc), WHITE)

    sb = math.sin(ms * 0.003142) * 2.0
    _draw_arc(draw, CX, MOUTH_Y-8+int(sb), 30, 12, 0, 180, WHITE, 4)

    for phase_off, sx2, sy in [(0.5, 258, 140), (1.2, 50, 155), (1.8, 278, 102)]:
        p = math.fmod(ms * 0.001 + phase_off, 2.5)
        if p < 0.1 or p > 2.4:
            continue
        d   = p / 2.5
        hx  = sx2 + int(10*d)
        hy  = sy  - int(50*d)
        a   = p/0.4 if p < 0.4 else ((2.5-p)/0.5 if p > 2.0 else 1.0)
        av  = int(max(0, min(1, a)) * 180)
        hs  = max(3, int(9 * (1.0 - d * 0.4)))
        _heart_outline(draw, hx, hy, hs, (av, av//4, av//4), 2)


def _draw_confused(img, ms):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, H], fill=BG)

    to = int(math.sin(ms * 0.001571) * 5.0)

    for cx_, cw in [(EYE_L_X, True), (EYE_R_X, False)]:
        base = (ms % 2400) / 2400.0 * 2 * math.pi * (1 if cw else -1)
        prev_x, prev_y = cx_, EYE_Y
        for step in range(1, 74):
            a = step * 0.1
            r = a / 7.3 * 22.0
            ang = base + a * (1 if cw else -1)
            nx  = cx_ + r * math.cos(ang) + to
            ny  = EYE_Y + r * math.sin(ang)
            if step > 1:
                draw.line([(int(prev_x), int(prev_y)), (int(nx), int(ny))], fill=WHITE, width=2)
            prev_x, prev_y = nx, ny
        draw.ellipse([cx_+to-3, EYE_Y-3, cx_+to+3, EYE_Y+3], fill=WHITE)

    mp = math.sin(ms * 0.003142) * 0.5 + 0.5
    mw = int(54 * (1.0 - 0.18 * mp))
    draw.rounded_rectangle([CX-mw//2+to, MOUTH_Y-2, CX+mw//2+to, MOUTH_Y+3], radius=2, fill=WHITE)

    qy = 46 - int(math.sin(ms * 0.002513) * 10.0)
    try:
        f  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 40)
        draw.text((238, qy), "?", font=f, fill=WHITE)
    except Exception:
        draw.text((238, qy), "?", fill=WHITE)


def _draw_rizz(img, ms):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, H], fill=BG)

    phase   = math.fmod(ms * 0.000357, 1.0)
    winking = 0.10 < phase < 0.45
    lean_o  = int(math.sin(ms * 0.002244) * 3.0)

    lh = 3 if winking else 8
    draw.rounded_rectangle(
        [EYE_L_X+lean_o-25, EYE_Y-lh//2, EYE_L_X+lean_o+25, EYE_Y+lh//2],
        radius=lh//2, fill=WHITE
    )

    rh = 13 if winking else 22
    _draw_arc(draw, EYE_R_X+lean_o, EYE_Y+5, 27, rh, 180, 360, WHITE, 5)

    lo = lean_o
    draw.line([(142+lo, 140),(163+lo, 136)], fill=WHITE, width=4)
    draw.line([(163+lo, 136),(174+lo, 130)], fill=WHITE, width=4)
    draw.line([(174+lo, 130),(183+lo, 136)], fill=WHITE, width=4)

    if 0.50 < phase < 0.80:
        gp = (phase - 0.50) / 0.30
        gs = math.sin(gp * math.pi)
        av = int(gs * 240)
        gc = (av, av, int(av * 0.5))
        _sparkle(draw, 260, 58, 9, gs, gc)


# ─── Draw dispatch ───────────────────────────────────────────────────────────
_DRAW_MAP = {
    "idle":         _draw_idle,
    "speaking":     _draw_speaking,
    "happy":        _draw_happy,
    "sad":          _draw_sad,
    "angry":        _draw_angry,
    "panic":        _draw_panic,
    "surprised":    _draw_surprised,
    "shy":          _draw_shy,
    "sleep":        _draw_sleep,
    "thinking":     _draw_thinking,
    "reconnecting": _draw_reconnecting,
    "love":         _draw_love,
    "confused":     _draw_confused,
    "rizz":         _draw_rizz,
}


# ════════════════════════════════════════════════════════════════════════════
# TFT RENDERER CLASS
# ════════════════════════════════════════════════════════════════════════════

class TFTEmotionRenderer:
    """
    Threaded TFT renderer for ADAM.

    Usage:
        tft = TFTEmotionRenderer()
        tft.start()
        tft.set_emotion("happy")
        ...
        tft.stop()
    """

    def __init__(self,
                 spi_port=0,
                 spi_device=0,
                 gpio_DC=24,
                 gpio_RST=25,
                 spi_speed_hz=32_000_000,
                 brightness_gpio=None):

        self._emotion   = "idle"
        self._lock      = threading.Lock()
        self._running   = False
        self._thread    = None
        self._device    = None

        if _HARDWARE:
            try:
                serial = spi(
                    port=spi_port,
                    device=spi_device,
                    gpio_DC=gpio_DC,
                    gpio_RST=gpio_RST,
                    bus_speed_hz=spi_speed_hz,
                    reset_hold_time=0.2,
                    reset_release_time=0.2,
                )
                self._device = ili9341(serial, width=W, height=H, rotate=0)
                print("[adam_tft] ILI9341 initialised OK")
            except Exception as e:
                print(f"[adam_tft] Hardware init failed: {e}  — headless mode")
        else:
            print("[adam_tft] Headless mode (luma.lcd not available)")

    # ── Public API ───────────────────────────────────────────────────────────

    def set_emotion(self, name: str):
        """Thread-safe emotion setter. Accepts any string in EMOTIONS list."""
        name = name.lower().strip()
        if name not in _DRAW_MAP:
            print(f"[adam_tft] Unknown emotion '{name}', defaulting to idle")
            name = "idle"
        with self._lock:
            self._emotion = name

    def get_emotion(self) -> str:
        with self._lock:
            return self._emotion

    def start(self):
        """Start the background render thread."""
        self._running = True
        self._thread  = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        print("[adam_tft] Render thread started")

    def stop(self):
        """Stop the render thread and clear the display."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._device:
            try:
                blank = Image.new("RGB", (W, H), BG)
                self._device.display(blank)
            except Exception:
                pass
        print("[adam_tft] Render thread stopped")

    # ── Internal render loop ─────────────────────────────────────────────────

    def _render_loop(self):
        img = Image.new("RGB", (W, H), BG)
        start_time = time.monotonic()

        while self._running:
            t0 = time.monotonic()
            ms = int((t0 - start_time) * 1000)

            with self._lock:
                emo = self._emotion

            fn = _DRAW_MAP.get(emo, _draw_idle)
            fn(img, ms)

            if self._device:
                try:
                    self._device.display(img)
                except Exception as e:
                    print(f"[adam_tft] Display error: {e}")

            elapsed = time.monotonic() - t0
            sleep   = max(0.0, FRAME_MS - elapsed)
            time.sleep(sleep)


# ════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST  (run directly on the Pi to test all emotions)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    renderer = TFTEmotionRenderer()
    renderer.start()

    if len(sys.argv) > 1:
        # python3 adam_tft.py happy
        renderer.set_emotion(sys.argv[1])
        print(f"[adam_tft] Holding emotion: {sys.argv[1]}  —  Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        # Cycle through all emotions automatically
        print("[adam_tft] Cycling all emotions (5 s each) — Ctrl+C to stop")
        try:
            for emo in EMOTIONS:
                print(f"  → {emo}")
                renderer.set_emotion(emo)
                time.sleep(5)
        except KeyboardInterrupt:
            pass

    renderer.stop()
