# Desktop touch assignments

The desktop can save touch assignments to the Pi through the existing data API.
This is an additive local source change; it has not been deployed to hardware.

## Physical numbering

The current `esp32_cam/esp32_cam.ino` is authoritative:

| ID | Firmware label | GPIO |
|---|---|---|
| touch1 | Left cheek | 12 |
| touch2 | Right cheek | 14 |
| touch3 | Stop / petting A | 15 |
| touch4 | Petting B | 2 |

Touch3 alone sends STOP. Touch3 + Touch4 sends PETTING. The optional three-pad
firmware configuration omits Touch4. Desktop visual labels are Left side,
Right side, Top of head and Back of head, respectively. Those requested
placements do not change GPIO assignments; confirm wiring on the assembled robot.

## API

`GET /api/ping` advertises `capabilities.touch_assignments`, `touch_events` and
`touch_gesture_policy: 2` when the runtime has started. Desktop requires policy 2
before applying the revised gestures. `GET /api/touch/assignments` returns
`{ok:true, api:1, assignments:{...}, sensors:{...}}`.

`POST` or `PUT /api/touch/assignments`, with the existing `X-ADAM-Token` header,
accepts `{assignments:{touch1:{hold:{action:"volume_up",value:null}},touch3:{double:...,triple:...,hold:...},...}}`.
The success reply echoes the validated assignments only after an atomic save to
`touch_assignments.json`. The desktop compares this acknowledgement with what it
sent before displaying Applied. A bad key returns403; invalid mappings return400;
disk failure returns500 and retains the previously active mapping.

Assignments use the shared `laptop_actions.py` types. Clipboard operations and
coding-task dispatch are excluded. Missing pads retain original firmware
gestures. **Single tap cannot be reassigned on any pad.** Touch1/2/4 allow hold
only; Touch3 allows double, triple and hold. The API rejects other mappings.
An explicit `none` disables that custom gesture, while single-tap reactions
remain built in. An empty object restores all firmware defaults. Version 1
stored mappings migrate in memory by retaining only the configurable gestures;
the next save writes version 2 atomically.

## Runtime and priority

One controller lives outside Gemini reconnects. It consumes the existing raw
four-byte T packets and G gesture packets, so no firmware wire change is needed.
Touch3 waits 300 ms after a release for the next tap. Two taps wait for a
possible third tap, then produce one double event; three taps produce one
triple event, never an additional double or single event. Holding 650 ms
produces one hold and cancels the pending tap sequence. Single taps on configured
pads use the fixed robot reaction. A petting chord retains priority over custom
holds. Each physical
press receives only one custom action; repeated firmware STOP/slap codes are
deduplicated. A pad already held when the runtime starts cannot trigger a shortcut.

Alarm dismiss/snooze retains priority on Touch1–3. Touch3 retains song-stop and
idle wake-up priority. The protected state is latched for that press so dismissing
an alarm cannot also run a laptop shortcut when the same finger is released.
The original session gesture branches handle these local behaviors. Default
session reactions still require an active session, as before; mapped laptop
shortcuts continue during a Gemini reconnect.

Laptop actions use the existing authenticated client on a separate worker, with
a bounded16-item queue. Telemetry cannot block raw touch handling. Changing the
mapping discards queued actions from the old mapping. Shutdown cancels the
controller and discards queued work; an HTTP request already sent can complete
within the client's existing timeout.

WebSocket telemetry is `{type:"touch",id,sensor,event,action,handled_by:"pi",status}`.
Custom `event` is double/triple/hold; status is ok/error/ignored. Values and credentials are
excluded. **The PC must not execute this event again:** the Pi already dispatched
the action over the authenticated laptop command channel.

## Verification

Run `python -m unittest discover -s "MP-MC codes/pi/tests" -p test_touch_controls.py -v`
from the repository root. Tests cover classification, deduplication, protected
presses, default fallbacks, persisted acknowledgement, auth rejection, async
dispatch and stale queue invalidation using simulated touches and an isolated
HTTP server. Physical pad timing and the optional three-pad board remain hardware
verification steps.
