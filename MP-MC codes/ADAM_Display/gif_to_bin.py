"""
gif_to_bin.py — Convert a GIF or MP4 into a Pico-playable sprite .bin file
============================================================================
Run this on your PC. Works for both simple flat-color GIFs (like the
existing face animations) and busier GIFs/MP4 clips.

USAGE:
    python3 gif_to_bin.py <input.gif|input.mp4> <emotion_name> [options]

    python3 gif_to_bin.py wink.gif wink
    python3 gif_to_bin.py dance.mp4 confetti --max-frames 30 --colors 32
    python3 gif_to_bin.py detailed.gif surprised --colors 16 --fps 12

OPTIONS:
    --max-frames N     Cap total frames sampled from the source (default 30).
                        Longer/higher-fps sources are downsampled evenly.
    --fps N             Target playback fps to encode into the .bin header
                        (default 24). This is a suggestion the Pico player
                        reads; it does not itself change how many frames
                        are sampled (use --max-frames / --sample-every).
    --sample-every N    Instead of computing sampling from --max-frames,
                        take every Nth source frame directly. Overrides
                        --max-frames if given.
    --colors N          Quantize each frame to N colors before compressing
                        (default 64). Lower = smaller file, blockier look.
                        Flat cartoon GIFs barely need this; use it for
                        detailed/MP4 sources where high color count hurts
                        RLE compression a lot. Set to 0 to disable
                        quantization entirely (keep full 16-bit color).
    --fit MODE          letterbox (default) | crop | stretch
    --out-dir DIR        Output directory (default: alongside this script)

WHY QUANTIZATION MATTERS:
    The RLE format used here compresses RUNS of identical adjacent pixels.
    Flat-color cartoon art (like your existing faces: solid black bg, thin
    white lines) compresses extremely well -- 150KB frames become 1-2KB.
    Photographic or gradient-heavy content has almost no identical runs at
    full 16-bit color, so RLE barely helps and files can balloon to nearly
    the full 153.6KB PER FRAME, which will not fit comfortably in flash for
    more than a few frames. Reducing to a smaller color palette (e.g. 16-32
    colors) creates the flat runs RLE needs, at some visual cost. This
    script warns you if a source is compressing poorly so you're not
    surprised by a huge or failing upload.

OUTPUT: <emotion_name>.bin in --out-dir, ready to copy onto the Pico's
flash filesystem (Thonny drag-drop, or `mpremote cp <file> :`).
Same file format as generate_frames.py -- both feed the same
adam_v32_sprite_player.py on the Pico.
"""

import sys
import os
import struct
import argparse

from PIL import Image, ImageSequence

W, H = 320, 240

# Flash budget guardrail -- warn (not block) if a clip looks too big to be
# a comfortable citizen of the Pico's flash filesystem alongside other
# emotion clips. Adjust if you know your board's actual free flash size.
WARN_TOTAL_BYTES = 300 * 1024   # 300KB
HARD_WARN_TOTAL_BYTES = 800 * 1024  # 800KB -- loud warning


def rgb565_swapped_bytes(img_rgb):
    """Convert a PIL RGB image (W x H) into RGB565 bytes, byte-swapped for
    the ST7789's big-endian SPI expectation. Same bit math as _c() in the
    MicroPython files -- kept consistent so PC-generated and Pico-decoded
    colors match exactly."""
    px = img_rgb.load()
    out = bytearray(W * H * 2)
    idx = 0
    for y in range(H):
        for x in range(W):
            r, g, b = px[x, y]
            v = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
            swapped = ((v & 0xFF) << 8) | (v >> 8)
            out[idx] = swapped & 0xFF
            out[idx + 1] = (swapped >> 8) & 0xFF
            idx += 2
    return bytes(out)


def rle_encode(raw_bytes):
    """RLE over 16-bit pixels: (count:u16, color:u16) pairs. Identical to
    the format used in generate_frames.py and decoded by
    adam_v32_sprite_player.py -- keep these three files in sync if you
    ever change the format."""
    assert len(raw_bytes) % 2 == 0
    n_pixels = len(raw_bytes) // 2
    pixels = struct.unpack("<%dH" % n_pixels, raw_bytes)

    out = bytearray()
    i = 0
    while i < n_pixels:
        color = pixels[i]
        run = 1
        while i + run < n_pixels and pixels[i + run] == color and run < 65535:
            run += 1
        out += struct.pack("<HH", run, color)
        i += run
    return bytes(out)


def fit_frame(img, mode):
    """Resize/fit a source frame (any size) to exactly W x H using the
    requested strategy."""
    img = img.convert("RGB")
    src_w, src_h = img.size
    src_ratio = src_w / src_h
    dst_ratio = W / H

    if mode == "stretch":
        return img.resize((W, H), Image.LANCZOS)

    if mode == "crop":
        # Scale up to cover, then center-crop
        if src_ratio > dst_ratio:
            new_h = H
            new_w = int(H * src_ratio)
        else:
            new_w = W
            new_h = int(W / src_ratio)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - W) // 2
        top = (new_h - H) // 2
        return resized.crop((left, top, left + W, top + H))

    # letterbox (default): scale to fit inside, pad with black
    if src_ratio > dst_ratio:
        new_w = W
        new_h = int(W / src_ratio)
    else:
        new_h = H
        new_w = int(H * src_ratio)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    off_x = (W - new_w) // 2
    off_y = (H - new_h) // 2
    canvas.paste(resized, (off_x, off_y))
    return canvas


def quantize_frame(img, n_colors):
    if n_colors <= 0:
        return img
    # PIL's adaptive palette quantization -- picks the N colors that best
    # represent this frame, which both shrinks the file and (more
    # importantly) creates the flat same-color runs RLE compresses well.
    quantized = img.quantize(colors=n_colors, method=Image.MEDIANCUT)
    return quantized.convert("RGB")


def load_source_frames(path):
    """Returns a list of PIL RGB frames from either a GIF or a video file,
    plus a best-guess source fps (used only for informational output)."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".gif":
        img = Image.open(path)
        frames = []
        durations = []
        for frame in ImageSequence.Iterator(img):
            frames.append(frame.convert("RGB"))
            durations.append(frame.info.get("duration", 100))  # ms
        avg_duration = sum(durations) / len(durations) if durations else 100
        src_fps = 1000.0 / avg_duration if avg_duration > 0 else 10.0
        return frames, src_fps

    else:
        # Assume video (mp4, mov, avi, etc.) -- use OpenCV
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file: %s" % path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frames = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
            frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames, src_fps


def sample_frames(frames, max_frames=None, sample_every=None):
    n = len(frames)
    if sample_every and sample_every > 1:
        return frames[::sample_every]
    if max_frames and n > max_frames:
        # Evenly spaced sampling across the whole clip
        step = n / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        return [frames[i] for i in indices]
    return frames


def build_bin(name, frames, fps, colors, fit_mode, out_dir):
    delay_ms = max(1, int(1000 / fps))
    n = len(frames)

    print(f"Processing '{name}': {n} frames, fit={fit_mode}, "
          f"colors={'full' if colors <= 0 else colors}, target_fps={fps}")

    compressed_frames = []
    total_raw = 0
    total_compressed = 0

    for i, frame in enumerate(frames):
        fitted = fit_frame(frame, fit_mode)
        fitted = quantize_frame(fitted, colors)
        raw = rgb565_swapped_bytes(fitted)
        comp = rle_encode(raw)
        compressed_frames.append(comp)
        total_raw += len(raw)
        total_compressed += len(comp)
        ratio = len(comp) / len(raw) * 100
        print(f"  frame {i+1}/{n}: {len(comp)}B ({ratio:.1f}% of raw)")

    # Compression sanity check / warning
    avg_ratio = total_compressed / total_raw
    if total_compressed > HARD_WARN_TOTAL_BYTES:
        print(f"\n⚠️  WARNING: '{name}.bin' will be ~{total_compressed/1024:.0f}KB.")
        print("   This is large for Pico flash alongside other clips. The source")
        print("   likely has too much fine detail/color variation for RLE to help.")
        print("   Try: --colors 16 (or lower), fewer --max-frames, or a simpler source.")
    elif total_compressed > WARN_TOTAL_BYTES:
        print(f"\n⚠️  Note: '{name}.bin' is ~{total_compressed/1024:.0f}KB. Still fine for")
        print("   flash, but consider --colors or --max-frames if you plan many clips.")

    if avg_ratio > 0.5:
        print(f"\n⚠️  Compression ratio is poor ({avg_ratio*100:.0f}% of raw size) --")
        print("   this source doesn't have many flat same-color runs. Try lowering")
        print("   --colors (e.g. 16 or 32) to improve this significantly.")

    # Pack the .bin file (same format as generate_frames.py)
    index_bytes = bytearray()
    data_bytes = bytearray()
    offset = 0
    for blob in compressed_frames:
        index_bytes += struct.pack("<II", offset, len(blob))
        data_bytes += blob
        offset += len(blob)

    header = struct.pack("<4sBHHHH", b"ADFB", 1, W, H, n, delay_ms)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.bin")
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(index_bytes)
        f.write(data_bytes)

    total = len(header) + len(index_bytes) + len(data_bytes)
    print(f"\n-> wrote {out_path}  ({total} bytes, {total/1024:.1f} KB)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Convert a GIF/MP4 to a Pico sprite .bin file")
    ap.add_argument("input", help="Path to .gif or video file (.mp4, .mov, .avi, ...)")
    ap.add_argument("name", help="Emotion name -- output will be <name>.bin, and this is "
                                   "the UART command word the Pico player matches against")
    ap.add_argument("--max-frames", type=int, default=30,
                     help="Cap sampled frame count (default 30)")
    ap.add_argument("--sample-every", type=int, default=None,
                     help="Take every Nth source frame instead of computing from --max-frames")
    ap.add_argument("--fps", type=float, default=24.0,
                     help="Target playback fps written into the .bin header (default 24)")
    ap.add_argument("--colors", type=int, default=64,
                     help="Quantize to N colors before compressing (default 64; use 0 to "
                          "disable and keep full color -- not recommended for detailed sources)")
    ap.add_argument("--fit", choices=["letterbox", "crop", "stretch"], default="letterbox",
                     help="How to fit non-320x240 source frames (default: letterbox)")
    ap.add_argument("--out-dir", default=None,
                     help="Output directory for the .bin file. Defaults to the same "
                          "folder as the input file. Use '.' for the current working "
                          "directory instead.")
    args = ap.parse_args()

    # FIX: previously defaulted to a hardcoded sandbox path that doesn't
    # exist on your machine. Now defaults to the SAME FOLDER AS THE INPUT
    # FILE (most intuitive -- output lands right next to the source you
    # converted), unless you explicitly pass --out-dir.
    if args.out_dir is None:
        args.out_dir = os.path.dirname(os.path.abspath(args.input)) or "."

    print(f"Loading '{args.input}'...")
    frames, src_fps = load_source_frames(args.input)
    print(f"Loaded {len(frames)} source frames (~{src_fps:.1f} source fps)")

    frames = sample_frames(frames, max_frames=args.max_frames, sample_every=args.sample_every)
    print(f"Sampled down to {len(frames)} frames for encoding")

    build_bin(args.name, frames, args.fps, args.colors, args.fit, args.out_dir)


if __name__ == "__main__":
    main()