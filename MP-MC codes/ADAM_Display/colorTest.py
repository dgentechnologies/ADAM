"""
ADAM v32 - Hardware Screen Geometry & Color Test
Driver: ST7789 (320x240) - INVERSION FIXED
"""
import machine
import time
import math
import framebuf
import gc

# --- Pre-allocate frame buffer ---
W, H = 320, 240
try:
    _buffer = bytearray(W * H * 2)
except MemoryError:
    print("Memory Error!")
    machine.reset()

fbuf = framebuf.FrameBuffer(_buffer, W, H, framebuf.RGB565)

# --- Hardware Pins ---
SPI_MOSI = 19
SPI_SCK  = 18
PIN_CS   = 17
PIN_DC   = 16
PIN_RST  = 20

# --- Color Palette ---
def color565(r, g, b):
    c = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    return ((c & 0xFF) << 8) | (c >> 8)

BG    = color565(255,   255,   255)
WHITE = color565(255, 255, 255)
BLACK = color565(0,   0,   0)
RED   = color565(255, 0,   0)
GREEN = color565(0,   255, 0)
BLUE  = color565(0,   0,   255)
GREY  = color565(120, 120, 120)

# ============================================================================
# ST7789 DRIVER CLASS
# ============================================================================
class ST7789:
    def __init__(self, spi, cs, dc, rst):
        self.spi = spi
        self.cs = machine.Pin(cs, machine.Pin.OUT)
        self.dc = machine.Pin(dc, machine.Pin.OUT)
        self.rst = machine.Pin(rst, machine.Pin.OUT)
        self.reset()
        self.init_display()

    def write_cmd(self, cmd):
        self.dc.value(0)
        self.cs.value(0)
        self.spi.write(bytearray([cmd]))
        self.cs.value(1)

    def write_data(self, data):
        self.dc.value(1)
        self.cs.value(0)
        self.spi.write(bytearray(data))
        self.cs.value(1)

    def reset(self):
        self.rst.value(1)
        time.sleep_ms(50)
        self.rst.value(0)
        time.sleep_ms(50)
        self.rst.value(1)
        time.sleep_ms(50)

    def init_display(self):
        self.write_cmd(0x11) # Sleep Out
        time.sleep_ms(120)
        self.write_cmd(0x36) # Memory Data Access Control
        self.write_data([0xA0]) # Restored to 0xA0
        self.write_cmd(0x3A) # RGB Format
        self.write_data([0x55]) # 16-bit
        
        # ---> THIS IS THE MAGIC FIX <---
        # Changed 0x21 (Inversion ON) to 0x20 (Inversion OFF)
        self.write_cmd(0x20) 
        
        self.write_cmd(0x13) # Normal Display On
        self.write_cmd(0x29) # Display On
        time.sleep_ms(50)

    def show(self):
        self.write_cmd(0x2A) 
        self.write_data([0x00, 0x00, (W-1)>>8, (W-1)&0xFF])
        self.write_cmd(0x2B) 
        self.write_data([0x00, 0x00, (H-1)>>8, (H-1)&0xFF])
        self.write_cmd(0x2C) 
        self.dc.value(1)
        self.cs.value(0)
        self.spi.write(_buffer)
        self.cs.value(1)

# ============================================================================
# CUSTOM SHAPES
# ============================================================================
def fill_circle(cx, cy, r, color):
    """Draws a solid filled circle."""
    for y in range(-r, r + 1):
        for x in range(-r, r + 1):
            if x*x + y*y <= r*r:
                fbuf.pixel(cx + x, cy + y, color)

def get_dynamic_color(ms):
    """Generates a smooth morphing RGB color using Sine waves."""
    r = int((math.sin(ms * 0.001) + 1) * 127)
    g = int((math.sin(ms * 0.001 + 2) + 1) * 127)
    b = int((math.sin(ms * 0.001 + 4) + 1) * 127)
    return color565(r, g, b)

# ============================================================================
# MAIN LOOP
# ============================================================================
def main():
    print("Initializing ST7789 Color Test...")
    spi = machine.SPI(0, baudrate=40_000_000, polarity=1, phase=1, 
                      sck=machine.Pin(SPI_SCK), mosi=machine.Pin(SPI_MOSI))
    
    tft = ST7789(spi, cs=PIN_CS, dc=PIN_DC, rst=PIN_RST)
    
    print("Running display test...")
    while True:
        ms = time.ticks_ms()
        fbuf.fill(BG) # Clear screen to white
        
        # 1. Draw Squares
        fbuf.fill_rect(35, 20, 50, 50, RED)
        fbuf.fill_rect(135, 20, 50, 50, GREEN)
        fbuf.fill_rect(235, 20, 50, 50, BLUE)
        
        # 2. Draw Text (Centered under squares)
        fbuf.text("RED", 48, 80, WHITE)
        fbuf.text("GREEN", 140, 80, WHITE)
        fbuf.text("BLUE", 245, 80, WHITE)
        
        # 3. Draw Black Circle with Grey Border
        # Trick: Draw a solid grey circle, then a slightly smaller black circle inside it!
        fill_circle(160, 140, 30, GREY)
        fill_circle(160, 140, 26, BLACK)
        
        # 4. Bottom Constantly Changing Color Bar
        dyn_color = get_dynamic_color(ms)
        fbuf.fill_rect(0, 195, 320, 45, dyn_color)
        fbuf.text("DYNAMIC COLOR STRIP", 85, 215, WHITE)
        
        # Push to screen
        tft.show()
        
        gc.collect()
        time.sleep_ms(30)

if __name__ == '__main__':
    main()