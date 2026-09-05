"""
watchdog.py — Parallel Supervisor & Hardware Watchdog for ADAM
==============================================================
Runs parallel to or as a supervisor for ADAM.
Monitors:
  1. I2S hardware audio lockup (continuous 0.0 RMS while mic should be active)
  2. Process freeze / event loop hang (heartbeat timed out > 15s)
  3. Process crashes or unhandled exceptions
  4. ALSA soundcard device contention / orphaned child processes

When an issue is detected:
  1. Logs the exact failure cause with timestamp
  2. Cleanly terminates ADAM and kills lingering arecord/aplay subprocesses
  3. Releases /dev/snd/* devices and resets ALSA
  4. Automatically restarts ADAM

Usage:
  # Supervisor mode (RECOMMENDED — runs and supervises main.py):
  python watchdog.py

  # Daemon mode (Runs parallel in background, monitoring an existing ADAM instance):
  python watchdog.py --daemon

  # Status check:
  python watchdog.py --status
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

from heartbeat import clear_heartbeat, read_heartbeat

BOOT_GRACE_PERIOD_S = 45.0
HEARTBEAT_TIMEOUT_S = 20.0
MAX_ZERO_RMS_CHUNKS = 240  # ~8.0s of continuous 0.0 RMS while listening
RESTART_BURST_WINDOW_S = 60.0
MAX_RESTARTS_PER_WINDOW = 5
COOLDOWN_AFTER_BURST_S = 30.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(SCRIPT_DIR, "main.py")


def log(msg: str) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] 🛡️  {msg}", flush=True)


def cleanup_audio_hardware() -> None:
    """Kill any orphan arecord or aplay processes holding /dev/snd/*."""
    try:
        subprocess.run(["pkill", "-9", "-f", "arecord"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "aplay"], capture_output=True)
    except Exception:
        pass


def cleanup_stale_adam(exclude_pid: int | None = None) -> None:
    """Kill any existing main.py or orphan audio processes."""
    cleanup_audio_hardware()
    my_pid = os.getpid()
    try:
        p = subprocess.run(["pgrep", "-f", "python.*main\\.py"], capture_output=True, text=True)
        pids = [int(x) for x in p.stdout.strip().split() if x.isdigit()]
        for pid in pids:
            if pid != my_pid and pid != exclude_pid:
                kill_process_tree(pid, timeout=1.0)
    except Exception:
        pass
    cleanup_audio_hardware()



def kill_process_tree(pid: int, timeout: float = 3.0) -> None:
    """Terminate process gracefully with SIGTERM, escalate to SIGKILL."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception as e:
        log(f"SIGTERM error on PID {pid}: {e}")

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
            time.sleep(0.2)
        except ProcessLookupError:
            return

    # Escalate to SIGKILL
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as e:
        log(f"SIGKILL error on PID {pid}: {e}")


def check_status() -> int:
    """Print current health status of ADAM."""
    hb = read_heartbeat()
    if not hb:
        print("❌ No active ADAM heartbeat found. ADAM is not running.")
        return 1

    age = time.time() - hb.get("timestamp", 0)
    pid = hb.get("pid")
    status = hb.get("status", "unknown")
    rms = hb.get("mic_rms", 0.0)
    zero_run = hb.get("zero_run", 0)

    is_alive = False
    if pid:
        try:
            os.kill(pid, 0)
            is_alive = True
        except ProcessLookupError:
            is_alive = False

    print("=" * 50)
    print("  ADAM Runtime Health Status")
    print("=" * 50)
    print(f"  Process PID   : {pid} ({'ALIVE' if is_alive else 'DEAD'})")
    print(f"  Health Status : {status.upper()}")
    print(f"  Heartbeat Age : {age:.1f}s ago")
    print(f"  Mic RMS       : {rms:.1f}")
    print(f"  Zero RMS Run  : {zero_run} chunks")
    print("=" * 50)

    if age > HEARTBEAT_TIMEOUT_S or not is_alive or status == "i2s_dead":
        print("⚠️  ADAM requires attention or restart.")
        return 1
    print("✅ ADAM is healthy.")
    return 0


def run_supervisor() -> None:
    """Run main.py as a managed child process with auto-recovery."""
    log("Starting ADAM Watchdog Supervisor...")
    log(f"Target: {MAIN_SCRIPT} (interpreter: {sys.executable})")

    restart_times: list[float] = []
    child_proc: subprocess.Popen | None = None
    stop_requested = False

    def _sig_handler(sig, frame):
        nonlocal stop_requested
        sig_name = "SIGINT" if sig == signal.SIGINT else "SIGTERM"
        log(f"Received {sig_name} — terminating supervisor and child...")
        stop_requested = True
        if child_proc and child_proc.poll() is None:
            try:
                child_proc.send_signal(signal.SIGINT)
            except Exception:
                pass

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    while not stop_requested:
        # Rate-limiting restart storms
        now = time.time()
        restart_times = [t for t in restart_times if now - t < RESTART_BURST_WINDOW_S]
        if len(restart_times) >= MAX_RESTARTS_PER_WINDOW:
            log(f"⚠️  Too many restarts ({len(restart_times)} in {RESTART_BURST_WINDOW_S}s). "
                f"Cooling down for {COOLDOWN_AFTER_BURST_S}s...")
            time.sleep(COOLDOWN_AFTER_BURST_S)
            restart_times.clear()

        cleanup_stale_adam()
        clear_heartbeat()
        time.sleep(1.0)

        log("Spawning fresh ADAM instance...")
        restart_times.append(time.time())

        try:
            child_proc = subprocess.Popen(
                [sys.executable, "-u", MAIN_SCRIPT],
                cwd=SCRIPT_DIR,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        except Exception as e:
            log(f"❌ Failed to launch ADAM: {e}")
            time.sleep(3.0)
            continue

        boot_deadline = time.time() + BOOT_GRACE_PERIOD_S
        failure_reason: str | None = None

        while not stop_requested:
            ret = child_proc.poll()
            if ret is not None:
                if stop_requested:
                    break
                failure_reason = f"Process exited unexpectedly with code {ret}"
                break

            now_t = time.time()
            hb = read_heartbeat()

            if hb:
                age = now_t - hb.get("timestamp", 0)
                status = hb.get("status", "healthy")
                zero_run = hb.get("zero_run", 0)

                if status == "i2s_dead" or zero_run >= MAX_ZERO_RMS_CHUNKS:
                    failure_reason = (
                        f"I2S audio capture wedged (continuous 0.0 RMS for {zero_run} chunks)"
                    )
                    break

                # Allow extra grace time while process is still booting / importing
                timeout_limit = BOOT_GRACE_PERIOD_S if status in ("booting", "starting") else HEARTBEAT_TIMEOUT_S
                if age > timeout_limit:
                    failure_reason = f"Heartbeat timed out ({age:.1f}s without response — process hung)"
                    break

            elif now_t > boot_deadline:
                # Still no heartbeat after boot window
                failure_reason = f"Process failed to produce initial heartbeat within {BOOT_GRACE_PERIOD_S:.0f}s"
                break

            time.sleep(1.0)

        if stop_requested:
            break

        if failure_reason:
            log(f"🚨 Issue detected: {failure_reason}")
            log("Initiating automatic recovery restart...")
            if child_proc and child_proc.poll() is None:
                kill_process_tree(child_proc.pid)
            cleanup_audio_hardware()
            clear_heartbeat()
            time.sleep(1.5)

    if child_proc and child_proc.poll() is None:
        kill_process_tree(child_proc.pid)
    cleanup_audio_hardware()
    clear_heartbeat()
    log("Supervisor stopped cleanly. Goodbye.")


def run_daemon() -> None:
    """Run as a parallel background daemon monitoring an external ADAM instance."""
    log("Starting ADAM Watchdog in Background Daemon mode...")
    stop_requested = False

    def _sig_handler(sig, frame):
        nonlocal stop_requested
        log("Daemon received shutdown signal.")
        stop_requested = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    while not stop_requested:
        hb = read_heartbeat()
        now_t = time.time()

        if hb:
            pid = hb.get("pid")
            age = now_t - hb.get("timestamp", 0)
            status = hb.get("status", "healthy")
            zero_run = hb.get("zero_run", 0)

            is_alive = False
            if pid:
                try:
                    os.kill(pid, 0)
                    is_alive = True
                except ProcessLookupError:
                    is_alive = False

            issue: str | None = None
            if not is_alive:
                issue = f"ADAM process PID {pid} has died"
            elif status == "i2s_dead" or zero_run >= MAX_ZERO_RMS_CHUNKS:
                issue = f"I2S capture wedged (0.0 RMS for {zero_run} chunks)"
            elif age > HEARTBEAT_TIMEOUT_S:
                issue = f"Heartbeat timed out ({age:.1f}s stale)"

            if issue:
                log(f"🚨 Daemon detected issue: {issue}")
                if pid and is_alive:
                    log(f"Terminating stuck PID {pid}...")
                    kill_process_tree(pid)
                cleanup_audio_hardware()
                clear_heartbeat()

                # Check if managed by systemd
                res = subprocess.run(
                    ["systemctl", "is-active", "adam"],
                    capture_output=True,
                    text=True,
                )
                if res.stdout.strip() == "active":
                    log("Restarting adam.service via systemd...")
                    subprocess.run(["sudo", "systemctl", "restart", "adam"])
                else:
                    log("Restarting ADAM via python main.py in background...")
                    subprocess.Popen(
                        [sys.executable, "-u", MAIN_SCRIPT],
                        cwd=SCRIPT_DIR,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                time.sleep(5.0)

        time.sleep(2.0)

    log("Daemon stopped cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="ADAM Parallel Watchdog & Supervisor")
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as an independent parallel background monitor",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check health status of the running ADAM instance and exit",
    )
    args = parser.parse_args()

    if args.status:
        sys.exit(check_status())
    elif args.daemon:
        run_daemon()
    else:
        run_supervisor()


if __name__ == "__main__":
    main()
