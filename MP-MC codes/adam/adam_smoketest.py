#!/usr/bin/env python3
"""ADAM Pi smoke test — run inside the venv on the Pi.

Verifies every module imports, the custom system prompt loads, vosk preloads,
and reports timing + peak memory. Exits non-zero if anything fails.
Usage (on Pi):  GEMINI_API_KEY=dummy ~/adam/venv/bin/python ~/adam/adam_smoketest.py
"""
import os, sys, time, resource, traceback

os.environ.setdefault("GEMINI_API_KEY", "dummy")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

fails = []

# 1) light modules (no vosk)
light = ['config', 'audio_utils', 'hardware', 'memory_store', 'tools_schema',
         'web_search', 'laptop_agent_client', 'song_playback', 'esp32_link',
         'ws_server', 'system_prompt', 'tool_handler']
print("=== module imports (light) ===")
for m in light:
    try:
        __import__(m)
        print(f"  OK   {m}")
    except Exception as e:
        print(f"  FAIL {m} -> {type(e).__name__}: {e}")
        traceback.print_exc()
        fails.append(m)

# 2) config values
import config
print("\n=== config sanity ===")
print(f"  CAPTURE_DEVICE  = {config.CAPTURE_DEVICE}")
print(f"  PLAYBACK_DEVICE = {config.PLAYBACK_DEVICE}")
print(f"  SYSTEM_PROMPT   = {config.SYSTEM_PROMPT_FILE.name} exists={config.SYSTEM_PROMPT_FILE.exists()}")
print(f"  UART            = {config.PI_UART_PORT} @ {config.PI_UART_BAUD}")
print(f"  NECK_GPIO_PIN   = {config.NECK_GPIO_PIN}")
print(f"  VOSK_MODEL_PATH = {config.VOSK_MODEL_PATH}")

# 3) custom system prompt actually loads (not the tiny built-in fallback)
import system_prompt
p = system_prompt.build_system_prompt()
print(f"\n=== system prompt ===\n  len={len(p)}  starts={p[:64]!r}")
if len(p) < 1500:
    print("  WARN: prompt short — custom SystemPrompt.txt may not be loading")

# 4) heavy import: session (preloads vosk model)
print("\n=== import session (preloads vosk) ===")
t = time.time()
try:
    import session  # noqa
    dt = time.time() - t
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  OK   session imported in {dt:.1f}s, peak RSS={rss:.0f} MB")
except Exception as e:
    print(f"  FAIL session -> {type(e).__name__}: {e}")
    traceback.print_exc()
    fails.append('session')

print("\n=== RESULT:", "ALL PASS" if not fails else f"FAILURES: {fails}", "===")
sys.exit(1 if fails else 0)
