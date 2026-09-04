"""
tool_handler.py — ADAM v40 tool-call dispatcher
==============================================================================
handle_tool_call() executes the function calls the Gemini model emits and
returns their results. It also owns a set of module-level "mailbox" flags that
bridge the sync-style tool handler and the async run_session loop.

WHY THE SINGLE-ELEMENT LISTS: several pieces of state have to be shared
between this module-level handler and run_session()'s nested coroutines
(which can't be closed over from here). They're stored as one-element lists
(e.g. `_doa_angle = [0.0]`) so both sides mutate the SAME object in place —
run_session imports these names by reference and reads/writes `[0]`. Never
rebind them (do `_idle_mode_requested[0] = True`, not `= [True]`), or the
two sides would drift onto different objects. This is safe as plain globals
because the codebase only ever runs ONE live session at a time.

  • _last_emotion_set_this_turn  — set_emotion() sets; end_of_turn() clears
  • _face_is_generic_speaking     — is the on-screen face the transient
                                    "speaking" placeholder vs a real emotion
  • _doa_angle / _doa_last_update_t — mirror of listen()'s DOA reading, for
                                    get_sound_direction
  • _idle_mode_requested          — enter_idle_mode() request mailbox
  • _idle_mode_persistent         — idle state that survives reconnects
  • _play_song_requested          — play_song() request mailbox
"""

import time
import asyncio
import datetime
from pathlib import Path

from config import (
    DOA_ANGLE_DEADZONE,
    SONG_FILE_PATHS,
    NECK_TILT_CENTER,
    NECK_PAN_CENTER,
    MEMORY_FILE,
    FACE_MEMORY_FILE,
)
from hardware import servo_pan, servo_tilt, tft_set
from memory_store import memory, faces, save_json
from web_search import web_search
from laptop_agent_client import laptop_control_sync

EMOTION_NOD = {
    "happy": "nod", "excited": "nod", "surprised": "nod", "love": "nod",
    "sad": "none",  "angry": "none",  "thinking": "none", "blush": "none",
    "confused": "none", "smug": "none", "sleep": "none", "rizz": "none",
    "panic": "none", "shy": "none", "reconnecting": "none",
}

# Module-level tracker for the emotion fix: set_emotion() calls update
# this; end_of_turn() in the speaker task checks and clears it. Safe as a
# plain module global since this codebase only ever runs one live session
# at a time (see run_session's single-session design throughout).
_last_emotion_set_this_turn = [False]
# Tracks whether the face CURRENTLY on screen is the transient
# "speaking" placeholder (as opposed to a deliberately-set emotion like
# love/angry/sad). Only this specific case should auto-reset back to a
# resting face when speech ends — a deliberately-set emotion should
# persist naturally. This was missing entirely after the previous fix
# removed the happy-fallback, which fixed "always resets to happy" but
# broke the opposite direction: nothing ever reset "speaking" back to a
# resting face once actual speech ended, so it stayed stuck showing
# "speaking" indefinitely.
_face_is_generic_speaking = [False]

# Module-level mirror of the session's DOA state, for get_sound_direction's
# handler (a module-level function, can't directly close over run_session's
# local doa_angle/doa_last_update_t). Updated from listen() on every fresh
# reading. Safe as a plain global since this codebase runs one live session
# at a time, same reasoning as _last_emotion_set_this_turn above.
_doa_angle = [0.0]
_doa_last_update_t = [0.0]

# Module-level mirror of idle_mode, for enter_idle_mode's handler — same
# reasoning as the DOA mirror above (handle_tool_call is module-level,
# can't directly close over run_session's local idle_mode Event). The
# run_session loop reads this each tick and syncs it to the real
# asyncio.Event, since a plain bool is simpler to touch from a sync-style
# tool handler than exposing the Event object itself across that boundary.
_idle_mode_requested = [False]

# PERSISTENT idle-mode state, surviving across reconnects. The session-
# local `idle_mode` asyncio.Event() inside run_session() is recreated
# fresh on every single call — including every reconnect (GoAway,
# transient 1007, network hiccup). Since conversations routinely span
# multiple sessions, idle mode was silently resetting to "not idle" on
# any reconnect with NO visible log line indicating it happened — the
# bug report showing full responses resuming with no "wake phrase heard"
# line is explained exactly by this: a reconnect happened between turns,
# and the fresh session's idle_mode simply started False again. This
# module-level flag is the source of truth that DOES survive reconnects;
# run_session() syncs its local Event to/from this at session start and
# on every change.
_idle_mode_persistent = [False]

# When the current idle period started (time.time(), 0.0 = not idle). Lives
# here rather than in run_session() so a reconnect cannot restart the clock.
# IDLE_MAX_S is enforced against it in session.py's idle_watcher(): a live log
# showed ADAM overhearing "be quiet" from a phone call on the other side of
# the room, calling enter_idle_mode(), and then staying idle indefinitely —
# every subsequent mic chunk went to the offline Vosk detector and nothing
# reached Gemini ("opens 1 sent 0" in the mic stats), which is what "ADAM is
# not hearing anything I say" looked like from outside. The documented exits
# (hearing "adam" locally, or Touch3) both failed in that room: the noise
# floor sat at the gate threshold, so the small en-us model was fed a
# continuous stream of call audio.
_idle_since = [0.0]

# Module-level mirror for play_song requests — same reasoning as
# _idle_mode_requested above. run_session() reads this each receive-loop
# tick right after tool dispatch and starts actual playback there, since
# that's where it has access to the real session-scoped song_playing/
# song_stop_requested Events and can spawn the background playback task.
_play_song_requested = [False]

async def handle_tool_call(tc, ws_broadcast_fn) -> list:
    responses = []
    for fc in tc.function_calls:
        name    = fc.name
        call_id = fc.id
        args    = dict(fc.args) if fc.args else {}
        try:
            if name == "get_current_datetime":
                now    = datetime.datetime.now()
                result = {
                    "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "date":     now.strftime("%A, %d %B %Y"),
                    "time":     now.strftime("%I:%M %p"),
                }

            elif name == "get_sound_direction":
                age = time.time() - _doa_last_update_t[0]
                if age > 4.0:
                    result = {"available": False,
                              "reason": "No recent enough audio reading to tell."}
                elif abs(_doa_angle[0]) <= DOA_ANGLE_DEADZONE:
                    result = {"available": True, "direction": "center",
                              "detail": "Sounds like you're roughly straight ahead."}
                else:
                    direction = "left" if _doa_angle[0] < 0 else "right"
                    result = {"available": True, "direction": direction,
                              "degrees_off_center": abs(int(_doa_angle[0]))}

            elif name == "enter_idle_mode":
                _idle_mode_requested[0] = True
                print("  🔇 enter_idle_mode called — will go silent")
                result = {"status": "ok",
                          "note": "Going silent now until woken by name."}

            elif name == "move_head_gesture":
                gesture = args.get("gesture", "nod")

                async def _do_gesture():
                    if gesture == "nod":
                        # Quick tilt down-up-down-center — a natural
                        # "yes" nod using the tilt servo.
                        for ang in (NECK_TILT_CENTER + 12,
                                   NECK_TILT_CENTER - 6,
                                   NECK_TILT_CENTER + 8,
                                   NECK_TILT_CENTER):
                            servo_tilt(ang)
                            await asyncio.sleep(0.18)
                    else:  # shake
                        # Quick pan left-right-left-center — a natural
                        # "no" shake using the pan servo.
                        for ang in (NECK_PAN_CENTER - 15,
                                   NECK_PAN_CENTER + 15,
                                   NECK_PAN_CENTER - 8,
                                   NECK_PAN_CENTER):
                            await asyncio.to_thread(servo_pan, ang)
                            await asyncio.sleep(0.18)

                if _idle_mode_persistent[0]:
                    # ADAM is in idle mode (STOP gesture / "stay silent").
                    # The head must stay centered and completely still
                    # until idle exits — suppress the physical nod/shake
                    # even if the model still emits this call off its
                    # still-live video feed. Audio is already hard-gated
                    # in both directions during idle; this closes the same
                    # gap for physical neck motion so the servos can't move
                    # on their own while ADAM is meant to be dormant.
                    print(f"  🤖 Head gesture '{gesture}' suppressed — idle")
                    result = {"status": "ok", "note": "Idle — staying still."}
                else:
                    print(f"  🤖 Head gesture: {gesture}")
                    # Run in the background so the tool response returns
                    # immediately rather than blocking the model's turn on
                    # ~0.7s of servo movement.
                    asyncio.create_task(_do_gesture())
                    result = {"status": "ok"}

            elif name == "play_song":
                if _play_song_requested[0]:
                    # Guard against duplicate tool_call messages in the
                    # same turn (observed in logs — Gemini can emit the
                    # same function call twice) triggering two overlapping
                    # song starts. Second call this turn is a no-op.
                    print("  🎵 play_song called again this turn — ignoring duplicate")
                    result = {"status": "ok", "note": "Already starting."}
                elif not any(Path(p).exists() for p in SONG_FILE_PATHS):
                    print(f"  ⚠️  play_song called but no song files found "
                          f"in: {SONG_FILE_PATHS}")
                    result = {"status": "error",
                              "reason": "No song files found — nothing to play."}
                else:
                    _play_song_requested[0] = True
                    print("  🎵 play_song called — starting playback")
                    result = {"status": "ok",
                              "note": "Playing now. Mic is muted until the "
                                      "song ends or Touch3 stops it."}

            elif name == "set_emotion":
                emotion = args.get("emotion", "happy")
                tft_set(emotion)
                _last_emotion_set_this_turn[0] = True
                _face_is_generic_speaking[0] = False
                await ws_broadcast_fn({"type": "emotion", "emotion": emotion,
                                       "head": EMOTION_NOD.get(emotion, "none")})
                result = {"status": "ok"}

            elif name == "save_memory":
                key = args.get("key", "").strip()
                val = args.get("value", "").strip()
                if key:
                    memory[key] = val
                    save_json(MEMORY_FILE, memory)
                    print(f"  🧠 Memory saved: {key}")
                    result = {"status": "saved"}
                else:
                    result = {"status": "error", "reason": "key empty"}

            elif name == "delete_memory":
                key = args.get("key", "").strip()
                if key in memory:
                    del memory[key]
                    save_json(MEMORY_FILE, memory)
                    result = {"status": "deleted"}
                else:
                    result = {"status": "not_found"}

            elif name == "get_memory":
                key    = args.get("key", "").strip()
                result = {"value": memory.get(key) if key else None, "all": memory}

            elif name == "remember_person":
                pid = args.get("person_id") or f"person_{int(time.time())}"
                faces[pid] = {
                    "name":         args.get("name", "Unknown"),
                    "appearance":   args.get("appearance", ""),
                    "relationship": args.get("relationship", "acquaintance"),
                    "notes":        args.get("notes", ""),
                    "last_seen":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                save_json(FACE_MEMORY_FILE, faces)
                print(f"  👤 Remembered: {args.get('name')} [{pid}]")
                result = {"status": "saved", "person_id": pid}

            elif name == "web_search":
                query = args.get("query", "").strip()
                recent_only = bool(args.get("recent_only", False))
                if query:
                    raw    = await web_search(query, recent_only=recent_only)
                    result = {"results": raw[:600] + ("…" if len(raw) > 600 else "")}
                else:
                    result = {"error": "query empty"}

            elif name == "laptop_control":
                action = args.get("action", "")
                value  = args.get("value")
                if value is not None:
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        value = None
                print(f"  🖥️  laptop_control → action={action} value={value}")
                result = await asyncio.to_thread(laptop_control_sync, action, value)
                if result.get("status") == "ok":
                    print(f"  ✅ laptop_control ok: {result}")
                else:
                    print(f"  ⚠️  laptop_control failed: {result}")

            else:
                result = {"error": f"unknown tool: {name}"}

        except Exception as e:
            result = {"error": str(e)}
            print(f"  ⚠️  Tool {name} error: {e}")

        responses.append({"id": call_id, "name": name, "response": result})
    return responses
