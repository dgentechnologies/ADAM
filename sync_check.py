#!/usr/bin/env python3
"""
sync_check.py — verify the Pi and the laptop are running the same code
==============================================================================
Two independent checks, both hash-based:

    python sync_check.py              # shared files: pi tree vs adam-desktop tree
    python sync_check.py --pi         # also: local pi tree vs the real Pi
    python sync_check.py --pi --host pi@adam-pi.local

WHY
---
ADAM is two programs that have to agree. The Pi decides what to send; the
laptop decides what it will accept. Every time those two drifted, the symptom
was never an error — it was a feature that silently did nothing:

  * v40's laptop_agent_client.py believed the laptop had 8 actions when it
    really had 18, so ten capabilities were invisible whenever mDNS discovery
    had not finished yet.
  * v40 coerced every laptop_control value to int, so write_clipboard and
    dispatch_coding_task arrived with their argument replaced by None, and the
    laptop answered "value required" to a request that had looked fine.

Both of those are now impossible to reintroduce quietly, because the protocol
lives in ONE file that both sides import — and this script fails loudly the
moment the two copies stop matching.

Exit code 0 = in sync, 1 = drift (or the Pi could not be reached, with --pi).
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PI_DIR = ROOT / "MP-MC codes" / "pi" / "adam"
PC_DIR = ROOT / "adam-desktop" / "src"

# ═════════════════════════════════════════════════════════════════════════════
# Files that must be BYTE-IDENTICAL in both trees.
#
# Only add a file here if both machines genuinely need the same logic — a shared
# protocol definition, a shared data format. Do NOT add a file just because the
# two sides have similar code: backend.py and session.py are different programs
# and always will be.
# ═════════════════════════════════════════════════════════════════════════════

SHARED_FILES = [
    "laptop_actions.py",
]

# What must exist on the Pi, with the same hash as the local copy, for a
# deployment to count as complete. Runtime state (JSON stores, .mic_floor.json)
# is deliberately excluded: those diverge by design the moment ADAM runs.
PI_DEPLOY_FILES = [
    "main.py", "session.py", "config.py", "tool_handler.py", "tools_schema.py",
    "hardware.py", "audio_utils.py", "memory_store.py", "system_prompt.py",
    "prompt_store.py", "prompts.txt", "prompts_check.py", "laptop_actions.py",
    "laptop_agent_client.py", "ws_server.py", "web_search.py",
    "song_playback.py", "adam_smoketest.py",
]

DEFAULT_PI_HOST = "pi@adam-pi.local"
DEFAULT_PI_PATH = "~/adam"

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_shared() -> int:
    """Compare the files both trees must share."""
    print("Shared protocol files (pi/adam  vs  adam-desktop)")
    print("-" * 70)
    problems = 0
    for name in SHARED_FILES:
        a, b = PI_DIR / name, PC_DIR / name
        if not a.exists():
            print(f"  {RED}MISSING{RESET}  {name}  (not in pi/adam)")
            problems += 1
            continue
        if not b.exists():
            print(f"  {RED}MISSING{RESET}  {name}  (not in adam-desktop) — "
                  f"copy it: cp '{a}' '{b}'")
            problems += 1
            continue
        ha, hb = sha(a), sha(b)
        if ha == hb:
            print(f"  {GREEN}same{RESET}     {name}  {DIM}{ha[:12]}  "
                  f"{a.stat().st_size} bytes{RESET}")
        else:
            print(f"  {RED}DIFFERS{RESET}  {name}")
            print(f"           pi/adam  {ha[:12]}  {a.stat().st_size} bytes")
            print(f"           adam-desktop    {hb[:12]}  {b.stat().st_size} bytes")
            print(f"           {YELLOW}One of these is stale. Decide which is "
                  f"correct, then copy it over the other.{RESET}")
            problems += 1
    return problems


def check_pi(host: str, remote: str) -> int:
    """Compare the local pi tree against what is actually running on the Pi."""
    print()
    print(f"Pi deployment ({host}:{remote})")
    print("-" * 70)

    # One SSH round trip for every hash, rather than one per file: on a Pi Zero
    # over Wi-Fi, 18 separate connections take the better part of a minute.
    files = " ".join(f"'{remote}/{f}'" for f in PI_DEPLOY_FILES)
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
             f"cd {remote} 2>/dev/null && sha256sum {files} 2>&1 || true"],
            capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        print(f"  {RED}Pi did not answer within 90s.{RESET}")
        print("  NOT VERIFIED — do not claim the Pi was updated.")
        return 1
    except FileNotFoundError:
        print(f"  {RED}ssh not found on this machine.{RESET}")
        return 1

    if r.returncode != 0 and not r.stdout.strip():
        print(f"  {RED}Could not reach the Pi.{RESET}  "
              f"{(r.stderr or '').strip().splitlines()[-1] if r.stderr else ''}")
        print("  NOT VERIFIED — do not claim the Pi was updated.")
        return 1

    remote_hashes = {}
    for line in r.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            remote_hashes[Path(parts[1].strip()).name] = parts[0]

    problems = 0
    for name in PI_DEPLOY_FILES:
        local = PI_DIR / name
        if not local.exists():
            print(f"  {DIM}skip{RESET}     {name}  (not in the local tree)")
            continue
        rh = remote_hashes.get(name)
        if rh is None:
            print(f"  {RED}ABSENT{RESET}   {name}  — not on the Pi at all")
            problems += 1
        elif rh == sha(local):
            print(f"  {GREEN}same{RESET}     {name}")
        else:
            print(f"  {RED}STALE{RESET}    {name}  — the Pi is running "
                  f"different code")
            problems += 1
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--pi", action="store_true",
                    help="also compare the local pi tree against the real Pi")
    ap.add_argument("--host", default=DEFAULT_PI_HOST)
    ap.add_argument("--path", default=DEFAULT_PI_PATH)
    args = ap.parse_args()

    problems = check_shared()
    if args.pi:
        problems += check_pi(args.host, args.path)

    print()
    if problems:
        print(f"{RED}OUT OF SYNC — {problems} problem(s).{RESET}")
        return 1
    print(f"{GREEN}IN SYNC.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
