import gc
gc.collect()          # do this FIRST, before anything else touches the heap

import machine, time, math, framebuf

TESTING_MODE = True
W, H = 320, 240

# ─────────────────────────────────────────────────────────────
# PRE-ALLOCATE FRAME BUFFER — retry with gc.collect() between
# attempts instead of resetting on the first failure. A single
# 153,600-byte allocation can fail on a fragmented heap even when
# free() reports enough total RAM; forcing a collect (and giving
# MicroPython's allocator a couple of tries) fixes that without
# needing a full reboot loop.
# ─────────────────────────────────────────────────────────────
_buf = None
for _attempt in range(5):
    gc.collect()
    try:
        _buf = bytearray(W * H * 2)
        break
    except MemoryError:
        time.sleep_ms(50)

if _buf is None:
    print("FATAL: cannot allocate framebuffer after retries — reset")
    machine.reset()

fb = framebuf.FrameBuffer(_buf, W, H, framebuf.RGB565)
gc.collect()