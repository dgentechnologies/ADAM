"""
song_playback.py — ADAM v40 song/concert playback
==============================================================================
_play_song_task() plays a randomly-chosen WAV from SONG_FILE_PATHS by writing
its PCM directly into the SAME aplay process that speaker() keeps open for the
whole session — never a second competing aplay (which reliably hit ALSA
"Device or resource busy" on plughw:0,0). Runs as its own asyncio task so the
camera, servos, and Gemini send/receive keep running; only the mic is muted
for the duration. Stops when the file ends or Touch3 sets song_stop_requested.

Shared state is passed in by the caller (session.run_session) as live objects:
  • song_playing / song_stop_requested — asyncio.Event flags
  • active_speaker_proc — single-element list holding speaker()'s live Popen
  • adam_speaking — asyncio.Event (kept in the signature for parity with the
                    caller's task wiring)
"""

import wave
import random
import asyncio
from pathlib import Path

from config import SONG_FILE_PATHS, PLAYBACK_CHANNELS, PLAYBACK_RATE
from hardware import tft_set


async def _play_song_task(song_playing: asyncio.Event,
                          song_stop_requested: asyncio.Event,
                          active_speaker_proc: list,
                          adam_speaking: asyncio.Event) -> None:
    """
    Plays a randomly-chosen song from SONG_FILE_PATHS by writing its PCM
    audio directly into the SAME aplay process speaker() already has
    open — not a second competing process.

    WHY THIS APPROACH (after the previous spawn-a-second-aplay design
    repeatedly hit "Device or resource busy"): speaker() opens ONE aplay
    process that stays alive for the entire session lifetime, only
    recreated on exception/reconnect — never closed between turns. Any
    second process trying to open the same ALSA device (plughw:0,0) will
    always contend with that permanently-open first one, no matter how
    carefully timed. The only way to truly avoid the collision is to not
    open a second device handle at all — write into the one that's
    already open and working, exactly the same way Gemini's own
    converted audio chunks already do via out_q → proc.stdin.write().

    This means song files must already be in the playback format
    (48kHz stereo s16 by default) — see SONG_FILE_PATHS' comment for the
    ffmpeg conversion command. No resampling is done here; keeping this
    function simple and fast is more important than accepting arbitrary
    input formats, since resampling on a Pi Zero 2W mid-playback is
    itself a source of audible glitches.

    Runs as its own asyncio task so nothing else in the event loop is
    blocked — camera, servos, Gemini send/receive, gestures all keep
    running normally in parallel. Only the mic is muted for the duration.

    Stops on whichever comes first: the file finishing naturally, or
    song_stop_requested being set (Touch3 during playback).
    """
    song_playing.set()
    song_stop_requested.clear()
    wav_file = None
    try:
        song_path = random.choice(SONG_FILE_PATHS)
        if not Path(song_path).exists():
            print(f"  ⚠️  Song file not found: {song_path}")
            return

        print(f"  🎵 Song playback started: {song_path}")
        tft_set("happy")

        wav_file = await asyncio.to_thread(wave.open, song_path, "rb")
        n_channels = wav_file.getnchannels()
        sampwidth  = wav_file.getsampwidth()
        framerate  = wav_file.getframerate()

        if (n_channels != PLAYBACK_CHANNELS or sampwidth != 2
                or framerate != PLAYBACK_RATE):
            print(f"  ⚠️  {Path(song_path).name} is {framerate}Hz "
                  f"{n_channels}ch {sampwidth*8}-bit, but playback expects "
                  f"{PLAYBACK_RATE}Hz {PLAYBACK_CHANNELS}ch 16-bit — "
                  f"convert it first with: ffmpeg -i input.mp3 -ar "
                  f"{PLAYBACK_RATE} -ac {PLAYBACK_CHANNELS} -sample_fmt "
                  f"s16 {Path(song_path).stem}.wav")
            return

        chunk_frames = 4096  # frames per read, matches speaker()'s own
                              # 4096-byte write granularity for out_q chunks
        pending_data = None  # a chunk that failed to write, retried below
        write_fail_streak = 0
        MAX_WRITE_FAIL_STREAK = 50  # ~10s of retries at 0.2s each before giving up

        while True:
            if song_stop_requested.is_set():
                print("  🎵 Song stopped early (Touch3)")
                break

            proc = active_speaker_proc[0]
            if proc is None or proc.poll() is not None:
                # speaker()'s aplay isn't available right now (mid-
                # reconnect, or session tearing down) — wait briefly for
                # it to come back rather than giving up. Any chunk we'd
                # already read but failed to write (pending_data) is kept
                # and retried once a live process shows up again, so the
                # song genuinely resumes from where it left off instead
                # of dropping audio or aborting on a reconnect.
                write_fail_streak += 1
                if write_fail_streak > MAX_WRITE_FAIL_STREAK:
                    print("  ⚠️  Song playback gave up — no speaker "
                          "process available after repeated retries")
                    break
                await asyncio.sleep(0.2)
                continue

            if pending_data is None:
                data = await asyncio.to_thread(wav_file.readframes, chunk_frames)
                if not data:
                    print("  🎵 Song finished playing")
                    break
            else:
                data = pending_data
                pending_data = None

            try:
                if proc.stdin:
                    await asyncio.to_thread(proc.stdin.write, data)
                    await asyncio.to_thread(proc.stdin.flush)
                    write_fail_streak = 0
                else:
                    raise RuntimeError("proc.stdin is None")
            except Exception as e:
                # Don't discard this chunk — the process that just died
                # (e.g. torn down mid-reconnect, confirmed in logs:
                # "Interrupted system call" / "I/O operation on closed
                # file") will be replaced by speaker() shortly. Keep the
                # chunk and retry it once active_speaker_proc[0] points
                # at a live process again, so the song resumes seamlessly
                # instead of stopping on every reconnect.
                pending_data = data
                write_fail_streak += 1
                if write_fail_streak == 1:
                    print(f"  ⚠️  Song playback write interrupted ({e}) — "
                          f"will resume once speaker reconnects")
                if write_fail_streak > MAX_WRITE_FAIL_STREAK:
                    print("  ⚠️  Song playback gave up after repeated "
                          "write failures")
                    break
                await asyncio.sleep(0.2)
                continue

            # Yield control between chunks so this doesn't hog the event
            # loop or the shared aplay stdin — camera/servo/Gemini tasks
            # all get their turn between song chunks too.
            await asyncio.sleep(0)
    except Exception as e:
        print(f"  ⚠️  Song playback error: {e}")
    finally:
        if wav_file is not None:
            try:
                wav_file.close()
            except Exception:
                pass
        song_playing.clear()
        song_stop_requested.clear()
        tft_set("happy")
