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
firmware configuration omits Touch4. There is no confirmed top/back orientation
in this protocol; the desktop uses the firmware labels.

## API

`GET /api/ping` advertises `capabilities.touch_assignments` and `touch_events`
only when the runtime has started. `GET /api/touch/assignments` returns
`{ok:true, api:1, assignments:{...}, sensors:{...}}`.

`POST` or `PUT /api/touch/assignments`, with the existing `X-ADAM-Token` header,
accepts `{assignments:{touch1:{tap:{action:"volume_up",value:null},double:...,hold:...},...}}`.
The success reply echoes the validated assignments only after an atomic save to
`touch_assignments.json`. The desktop compares this acknowledgement with what it
sent before displaying Applied. A bad key returns403; invalid mappings return400;
disk failure returns500 and retains the previously active mapping.

Assignments use the shared `laptop_actions.py` types. Clipboard operations and
coding-task dispatch are excluded. Missing pads retain original firmware
gestures. On a configured pad, an omitted gesture falls back to that pad's default
reaction; explicit `none` means no action. An empty object restores all defaults.

## Runtime and priority

One controller lives outside Gemini reconnects. It consumes the existing raw
four-byte T packets and G gesture packets, so no firmware wire change is needed.
A tap waits300ms for a second tap. Two releases within that window produce one
double tap. Holding650ms produces one hold and no trailing tap. Each physical
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
`event` is tap/double/hold; status is ok/error/ignored. Values and credentials are
excluded. **The PC must not execute this event again:** the Pi already dispatched
the action over the authenticated laptop command channel.

## Verification

Run `python -m unittest discover -s "MP-MC codes/pi/tests" -p test_touch_controls.py -v`
from the repository root. Tests cover classification, deduplication, protected
presses, default fallbacks, persisted acknowledgement, auth rejection, async
dispatch and stale queue invalidation using simulated touches and an isolated
HTTP server. Physical pad timing and the optional three-pad board remain hardware
verification steps.
