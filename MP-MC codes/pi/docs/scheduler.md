# The scheduler — alarms, timers, reminders and todos

`scheduler.py` gives ADAM a memory of *time*. It runs entirely on the Pi,
entirely offline, and survives a reboot. The PC app is a live **view** of this
data, never a second copy of it.

## 1. Why the Pi owns the time

ADAM can set an alarm by voice with no laptop anywhere in the room. If the
laptop held the only copy, an alarm set while the companion app was closed would
simply not exist. So the Pi is the source of truth and the app caches what it
reads; when the Pi is unreachable the app's Clock tab shows its age rather than
pretending to be live.

The same reasoning applies to the speaking path: the ticker that fires alarms
lives on the Pi, so an alarm rings whether or not any other machine is awake.

## 2. Storage

One JSON file, `config.SCHEDULE_FILE` (`~/adam/adam_schedules.json`):

```json
{"version": 1,
 "schedules": [ {…}, … ],
 "todos":     [ {…}, … ]}
```

Written through the same atomic `save_json()` the memory store uses, so a power
cut mid-write cannot leave a truncated file. `_sanitise()` runs on load and
drops entries that fail validation (`_valid_schedule`, `_valid_todo`) rather
than letting one corrupt entry break every alarm on the device.

## 3. The three kinds

| Kind | Fires | Notes |
|---|---|---|
| `alarm` | at a clock time, optionally repeating on chosen weekdays | the wake-up case |
| `reminder` | same behaviour as an alarm | differs in *how* it is delivered and which prompt section it fires into |
| `timer` | once, N seconds from when it was set | a countdown, not a clock time |

`reminder` and `alarm` are identical mechanically. Keeping them separate is
about delivery: a reminder names a task, an alarm is a wake-up, and the prompt
that carries them into the conversation differs.

## 4. The time model — the two decisions that matter

These are the difference between an alarm that works and one that misbehaves in
ways nobody can reproduce.

### Everything is local wall-clock time

Next-fire times are computed with `datetime.now()`, not UTC and not
`time.monotonic()`. An alarm set for "07:00" rings at 07:00 on the wall clock
the user is looking at, **including the morning after a DST change** — which is
what a person means by "7 am".

The trade-off is deliberate: a 24-hour *timer* spanning a DST shift runs for 24
hours of wall time rather than 86,400 SI seconds. That is the right trade for a
bedside alarm and the wrong one for a stopwatch, and this is a bedside alarm.

### The clock can jump, and the scheduler notices

The Pi has no RTC. It boots with a wrong clock and corrects itself over NTP — a
jump of minutes or hours, usually within the first minute of uptime.

A naive `now >= due` loop would fire **every stale schedule at once** the moment
that correction lands. So the ticker compares the wall clock against a monotonic
reference and treats a disagreement larger than `CLOCK_JUMP_S` as a resync: it
recomputes next-fire times from the new clock and **fires nothing**. Missed-fire
handling then decides, on its own, whether anything was genuinely missed.

This is the bug most likely to be reintroduced by anyone simplifying the ticker,
so the check has its own constant and its own comment rather than being folded
into the comparison.

## 5. Missed fires

A schedule whose time passed while ADAM was off is delivered on boot **if it was
due within `MISSED_GRACE_S`**, and silently marked missed if it was due longer
ago than that:

- A reminder from eleven minutes ago is still useful.
- A wake-up alarm from eleven hours ago is not.

Either way a one-shot is never left pending to surprise the user days later, and
a repeating alarm resumes at its **next** occurrence rather than replaying the
ones it slept through.

## 6. Duplicate fires

Each schedule records `last_fired` as a local-time string. Firing is conditional
on that not already matching the occurrence being fired, so a restart between
"decide to fire" and "speak it" cannot double-deliver the same alarm.

Without this, a crash in the wrong millisecond means the alarm rings twice —
which reads to the user as a bug in the whole feature.

## 7. The API

`scheduler.py` exports (`__all__`):

**Setting**
- `set_alarm(label, when="", repeat=None)`
- `set_reminder(label, when="", repeat=None)`
- `set_timer(seconds=None, label="", minutes=None, hours=None)`
- `add_todo(text, due="")`

**Reading**
- `list_schedules(include_past=False)`
- `list_todos(include_done=False)`
- `snapshot()` — everything, for the PC app's single call
- `render_for_prompt()` — upcoming items, formatted to inject into the prompt

**Changing**
- `cancel_schedule(target="")`
- `complete_todo(target="", text="")`
- `delete_todo(target="", text="")`
- `replace_all(schedules=None, todos=None)` — wholesale replacement, used by the sync API

**Firing**
- `drain_pending_fires()` — what is due now
- `alarm_ringing()`
- `snooze_current(minutes=5)`
- `dismiss_current()`

**Parsing** — `parse_time_of_day()`, `parse_when()`, `parse_weekdays()`,
`next_due()`. These turn "7 am", "in twenty minutes", "weekdays" into times, and
they are where an ambiguous request is caught. A time already in the past is
refused with wording ADAM can say rather than being silently rolled to tomorrow.

## 8. The envelope

Every setter returns `{"ok": bool, "reason": str, "data": {…}}` via `_ok()` /
`_err()`, and **nothing raises**. The refusals are written as things ADAM can
say — "I couldn't tell which alarm you meant" — because that string travels back
to the model and is spoken.

`tool_handler._handle_scheduler` passes that wording through verbatim instead of
flattening it to a generic failure, so the useful part survives.

## 9. The tool layer

Nine tools are declared: `set_alarm`, `set_reminder`, `set_timer`,
`list_schedules`, `cancel_schedule`, `add_todo`, `list_todos`, `complete_todo`,
`delete_todo`.

**`scheduler` is imported lazily, inside the tool branch** — not at module load
in `tool_handler.py`. This is the pattern that makes a new module safe to
deploy: a Pi that has not had `scheduler.py` copied yet loses this one feature
and still answers every other tool call. The failure mode is "that isn't
available", not "the robot is broken".

Confirmations come from the `confirmation` pools in `prompts.txt`, and the
result carries a note telling the model the line is a *suggestion, not a
script* — otherwise the model reads the line out verbatim every time and the
anti-repetition work is wasted.

## 10. Concurrency — why there are no locks

`asyncio.start_server` (the sync API), `main.py` and the scheduler tick all run
in the **same event loop**. A request handler and a scheduler tick therefore
cannot run at the same instant, and `replace_all()`'s read-modify-write is
atomic with respect to the ticker *by construction* rather than by locking.

**Adding a thread pool later would silently break this** and would need a real
lock. There is a comment on the read path in `sync_api.py` saying so.

## 11. The Clock tab

The PC app's Clock tab renders schedules, todos and memories fetched through the
app's own `/pi/snapshot` proxy (the browser does not call the Pi directly). It
polls only while the tab is visible, and every write is **relayed to the Pi and
reflected back from its reply** — never optimistically.

That ordering is the point: the Pi is the source of truth, so a failed write
must not leave the UI claiming the change landed. Ticking a to-do sends a
single-item verb (`{toggle: id}`) rather than rewriting the whole list, so
fields the display snapshot omits — a recurring alarm's stored time, a snooze
count — can never be lost in transit.

See [`pc_app_integration.md`](pc_app_integration.md) for the endpoints.

## 12. Configuration

| Setting | Default | Meaning |
|---|---|---|
| `SCHEDULE_FILE` | `~/adam/adam_schedules.json` | the store |
| `CLOCK_JUMP_S` | — | wall-vs-monotonic disagreement that counts as a resync |
| `MISSED_GRACE_S` | — | how late a missed schedule may still be delivered |

## 13. Testing

```bash
cd ~/adam && ./venv/bin/python adam_smoketest.py
```

The scheduler is not yet covered by its own smoke-test group — the `imports`
group verifies it imports cleanly, and the tool wiring is checked from the
`router` group's generation-map assertions. A `scheduler` group covering
`parse_when()` edge cases, the clock-jump resync and duplicate-fire suppression
is the obvious next test to write; until it exists, those three behaviours are
verified only by reading the code.

## 14. Files

| File | Role |
|---|---|
| `adam/scheduler.py` | the store, the time model, the ticker, the API |
| `adam/tool_handler.py` | `_handle_scheduler()` — envelope flattening, confirmation pools |
| `adam/tools_schema.py` | the nine declarations |
| `adam/sync_api.py` | HTTP read/write for the companion app |
| `adam/prompts.txt` | `confirmation` pools; the reminder injection section |
| `pcAPP/static/js/clock.js` | the Clock tab |
| `adam/config.py` | `SCHEDULE_FILE`, `CLOCK_JUMP_S`, `MISSED_GRACE_S` |
