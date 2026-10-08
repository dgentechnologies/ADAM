# ADAM Needle 2 + Gemini Live Hybrid Test Plan

## Purpose

This document is a **test-first integration plan** for ADAM v32 on the Raspberry Pi Zero 2W.

The goal is to experimentally validate this architecture:

```text
                         USER SPEECH
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
            LOCAL ASR                 GEMINI LIVE
                 │                  (already connected)
                 ▼                         │
             NEEDLE 2                      │
                 │                         │
                 ▼                         │
            LOCAL TOOL                     │
                 │                         │
                 ▼                         │
           TOOL RESULT ───────────────────►│
                                           ▼
                                  ADAM'S GEMINI VOICE
                                           │
                                           ▼
                                         SPEAKER
```

Example:

> "ADAM, increase the volume."

Expected behavior:

1. The microphone captures the speech.
2. Local ASR produces text.
3. Needle 2 determines that this is a local action.
4. Needle calls the existing laptop-control capability.
5. `laptop_agent.py` changes the laptop volume.
6. The laptop agent returns the actual resulting state, e.g. `80%`.
7. The existing Gemini Live session remains connected.
8. The result is injected into that existing Gemini session.
9. Gemini produces the confirmation using ADAM's **existing Gemini Live voice**.
10. The speaker plays the Gemini-generated audio exactly as it does today.

### Critical constraint

**Do NOT replace ADAM's Gemini voice with local TTS.**

Gemini Live remains the authoritative voice/output layer.

Needle 2 is being tested as a **local action brain**, not as ADAM's voice, personality, general conversation engine, or TTS engine.

---

# 1. Source-of-truth constraints

The current ADAM v32 implementation already uses:

- Raspberry Pi Zero 2W as the main brain.
- Gemini Live as the live voice session.
- Gemini voice `Charon`.
- `arecord` for microphone capture.
- `aplay` for speaker output.
- An asyncio-based concurrent architecture.
- Existing Gemini Live tool calling.
- Existing laptop-control tool.
- Existing modular laptop action registry.
- Existing `laptop_agent.py`.
- Existing persistent Gemini Live WebSocket/session.
- Existing Gemini `send_tool_response()` flow.

Do not unnecessarily rewrite the existing working audio pipeline.

The current documentation explicitly describes the Pi as running the Gemini Live session, audio pipeline, camera/gesture ingestion, servo control, TFT/emotion dispatch, memory, web search, and laptop control concurrently.

The current implementation already keeps one Gemini Live WebSocket session and has separate `listen`, `send`, `receive`, `speaker`, camera, gesture, wake-word, idle, and laptop-health tasks.

The current Gemini configuration uses:

- Live model: `gemini-3.1-flash-live-preview`
- Voice: `Charon`
- Audio capture: 48 kHz stereo → 16 kHz mono for Gemini
- Gemini output: 24 kHz mono → 48 kHz stereo playback

**Do not change these values merely to test Needle.**

---

# 2. First principle: this is an experiment, not a rewrite

Before modifying `ADAM_PI.py`, create an isolated test implementation.

The first objective is to answer:

> Can Needle 2 actually run reliably on THIS Raspberry Pi Zero 2W?

The second objective is:

> Can Needle reliably produce the intended local tool calls?

The third objective is:

> Can a Needle-generated action result be handed to the already-running Gemini Live session and cause ADAM to speak the result using the existing Gemini voice?

Only after all three work independently should they be combined.

---

# 3. Safety requirements

### DO NOT

- Replace Gemini Live.
- Replace Gemini's voice.
- Replace the existing speaker pipeline.
- Remove existing Gemini tools.
- Remove existing laptop-agent functionality.
- Rewrite the whole ADAM application.
- Change the working microphone configuration.
- Change the working speaker configuration.
- Disable the existing Gemini path.
- Make Needle responsible for general conversation.
- Make Needle responsible for TTS.
- Assume Needle works on Pi Zero 2W without testing.
- Delete or overwrite the working ADAM source.

### DO

- Create a separate branch or backup.
- Create isolated Needle test files.
- Keep the original ADAM application runnable.
- Log timings for every stage.
- Add feature flags for Needle.
- Allow immediate fallback to Gemini-only behavior.
- Make the experiment reversible.

---

# 4. Phase 0 — inspect the existing project

Before editing anything:

1. Identify the current ADAM main Python file.
2. Identify the current Gemini Live session implementation.
3. Identify the exact `handle_tool_call()` implementation.
4. Identify how Gemini tool responses are returned using `session.send_tool_response()`.
5. Identify the exact `laptop_control` implementation.
6. Identify the laptop agent's `/actions` and `/control` protocol.
7. Identify the current ASR/Vosk implementation.
8. Identify the current microphone queue.
9. Identify the current speaker/audio queue.
10. Identify all environment variables.
11. Identify the Python version.
12. Identify the Pi architecture.
13. Record the installed versions of the existing dependencies.

Do not make changes during this inspection phase.

Produce:

```text
docs/needle_test/00_environment.md
```

containing:

- Pi model
- OS
- kernel
- Python version
- architecture
- CPU information
- RAM
- available storage
- existing ADAM dependencies
- current Gemini configuration
- current audio configuration
- current laptop-agent configuration

---

# 5. Phase 1 — Needle compatibility test

Install/test the **latest available Needle 2 release**, not an old tutorial version.

The test must explicitly verify:

```bash
uname -m
python3 --version
python3 -c "import platform; print(platform.machine())"
```

Then inspect the installed Needle package and its native library.

Do not blindly trust that an `aarch64` wheel means that every ARM CPU feature required by the binary is supported by the Pi Zero 2W.

There has previously been a reported Pi Zero 2W `Illegal instruction` problem with Needle, so this must be treated as a hard compatibility test.

## Required result

Create:

```text
docs/needle_test/01_needle_compatibility.md
```

Record:

- installed Needle version
- wheel/package used
- architecture
- import result
- model loading result
- inference result
- whether native library loads
- whether any `Illegal instruction`, SIGILL, segmentation fault, or illegal CPU instruction occurs
- RAM usage
- CPU usage
- load time
- inference time

If Needle crashes with SIGILL:

**STOP integration work.**

Do not modify ADAM to work around a native binary crash until the cause is understood.

Try the latest official supported ARM build/package first.

---

# 6. Phase 2 — standalone Needle inference

Create:

```text
tools/needle_test/
    needle_smoke_test.py
```

The script must:

1. Load Needle 2.
2. Define a minimal set of ADAM-like tools.
3. Send fixed text commands.
4. Print the structured output.
5. Measure latency.

Start with only these conceptual tools:

```text
volume_up
volume_down
volume_set
volume_mute
volume_unmute
brightness_up
brightness_down
brightness_set
```

Do not connect these tools to the real laptop yet.

Use fake functions.

Example input:

```text
increase the volume
```

Expected conceptual result:

```json
{
  "tool": "volume_up",
  "arguments": {}
}
```

Another:

```text
set laptop volume to 80 percent
```

Expected:

```json
{
  "tool": "volume_set",
  "arguments": {
    "value": 80
  }
}
```

The exact Needle API syntax must be determined from the installed/current official package documentation. Do not invent an API.

---

# 7. Phase 3 — benchmark Needle locally

Benchmark at least:

### Commands

```text
increase the volume
decrease the volume
set the volume to 80 percent
mute the laptop
unmute the laptop
increase brightness
set brightness to 60 percent
turn the volume down
make the screen brighter
```

Also test natural variations:

```text
can you make my laptop louder
make it a little louder
turn the sound up
could you increase my laptop volume
set my laptop volume at eighty percent
make the screen brighter
```

For every request record:

```text
input
model load time
inference start
inference end
inference latency
generated tool
arguments
confidence if available
valid/invalid
```

Create:

```text
docs/needle_test/02_needle_benchmark.md
```

Calculate:

- average latency
- median latency
- p95 latency
- successful tool-call percentage
- malformed output percentage
- wrong-tool percentage
- wrong-argument percentage

---

# 8. Phase 4 — test the existing laptop agent independently

Do not involve Gemini or Needle yet.

Use the existing `laptop_agent.py`.

Verify:

```text
GET /actions
POST /control
```

Confirm that the action manifest exposes the current actions.

Confirm that the laptop agent returns a useful result after execution.

For example:

```json
{
  "status": "ok",
  "volume": 80
}
```

If the current laptop agent does NOT return the resulting volume/brightness state, improve the laptop agent so that it returns the actual state.

This is important.

ADAM must never say:

> "The volume is 80%."

unless the tool actually reports `80`.

Needle should decide **what action to perform**.

The tool should be the **source of truth for what actually happened**.

Create:

```text
docs/needle_test/03_laptop_agent_result_contract.md
```

Document the response schema.

---

# 9. Phase 5 — Needle → real laptop agent

Now connect Needle to the laptop agent.

Architecture:

```text
Text
 ↓
Needle
 ↓
structured tool call
 ↓
ADAM tool adapter
 ↓
laptop_agent.py
 ↓
real laptop action
 ↓
structured result
```

Example:

```text
"increase the volume"
        ↓
Needle
        ↓
volume_up()
        ↓
laptop_agent.py
        ↓
OS volume changes
        ↓
{
  "status": "ok",
  "action": "volume_up",
  "volume": 80
}
```

Do not involve Gemini yet.

Test the entire path repeatedly.

---

# 10. Phase 6 — local ASR → Needle

The current ADAM project already has an optional Vosk-based offline wake-word/ASR path.

Inspect the current implementation first.

Do not automatically replace it.

For the experiment, create a separate pipeline:

```text
Microphone
 ↓
existing local ASR/Vosk
 ↓
recognized text
 ↓
Needle
 ↓
tool
```

If the current local ASR is unsuitable for continuous command recognition, document that fact rather than silently replacing it.

The goal here is to establish whether:

```text
spoken command → local text → Needle
```

is fast and reliable enough.

Measure:

```text
speech end
→ ASR final text
→ Needle start
→ Needle tool call
→ laptop action complete
```

---

# 11. Phase 7 — KEEP GEMINI LIVE CONNECTED

This is the most important experiment.

Do not create a new Gemini session after Needle executes an action.

The existing Gemini Live session must remain connected.

Current ADAM already has:

```python
async with client.aio.live.connect(...) as session:
```

and the receive loop already processes:

```python
msg.tool_call
```

and returns:

```python
await session.send_tool_response(...)
```

The experiment should reuse this existing session architecture.

---

# 12. Proposed dual-path pipeline

Implement a feature flag such as:

```text
NEEDLE_EXPERIMENT=true
```

When disabled:

```text
MIC
 ↓
Gemini Live
 ↓
existing ADAM behavior
```

When enabled:

```text
MIC
 │
 ├──────────────► Gemini Live
 │
 ▼
Local ASR
 │
 ▼
Needle
 │
 ├── local action ──► Tool
 │                       │
 │                       ▼
 │                   result
 │                       │
 │                       ▼
 └────────────────► Gemini Live
                         │
                         ▼
                    Gemini audio
                         │
                         ▼
                       Speaker
```

However, do NOT blindly send duplicate user audio/text into Gemini and Needle without thinking about duplicate responses.

The experiment must include a routing/coordination layer.

---

# 13. Duplicate-response problem

This is a critical issue.

If the same user command goes simultaneously to:

- Needle
- Gemini Live

then Gemini may independently decide to execute the same tool.

That could cause:

```text
volume +10
```

to happen twice.

Or:

```text
Needle → volume_up()
Gemini → laptop_control(volume_up)
```

Therefore, the first implementation must prevent duplicate execution.

Possible approaches:

### Approach A — Needle owns local commands

Local ASR classifies/handles the command.

If Needle produces a valid local tool call:

```text
Needle → execute
```

then Gemini receives only the **result/event**, not the original command as a fresh user request.

### Approach B — Gemini receives the original audio but local Needle is advisory

Needle predicts the local action.

Gemini remains the authority.

This is safer but doesn't give the lowest possible latency.

### Approach C — parallel speculative routing

Needle executes local actions immediately while Gemini is simultaneously listening.

This is the architecture we want to experimentally evaluate, but it requires a **deduplication/ownership mechanism**.

Do not deploy speculative parallel execution until duplicate tool execution is proven impossible.

---

# 14. Recommended first implementation

Use this sequence:

```text
USER SPEECH
    │
    ├──────────────► Gemini Live
    │
    ▼
Local ASR
    │
    ▼
Needle
    │
    ▼
Local intent/action
    │
    ▼
execute
    │
    ▼
tool result
    │
    ▼
inject result into existing Gemini session
```

But add a temporary experiment mode where Gemini is instructed not to execute the same local action again.

The exact mechanism must be determined from the currently supported Gemini Live API behavior and the existing ADAM code.

Do not fabricate undocumented API calls.

---

# 15. Gemini handoff contract

Create a single internal function such as:

```python
async def handoff_local_action_result(
    session,
    user_text: str,
    tool_name: str,
    tool_result: dict,
) -> None:
    ...
```

Its job is ONLY to communicate the already-completed local action to the existing Gemini session.

Example internal event:

```json
{
  "type": "local_action_completed",
  "user_request": "increase the volume",
  "action": "volume_up",
  "result": {
    "status": "ok",
    "volume": 80
  }
}
```

Gemini should then produce a short natural confirmation in ADAM's existing voice.

For example:

> "Yeah, the volume is set to 80%."

Do not ask Gemini to perform the action again.

---

# 16. Preserve ADAM's personality

The existing `system_prompt.txt` and Gemini system instructions remain authoritative.

The handoff should preserve:

- ADAM personality
- ADAM's speaking style
- ADAM's emotional behavior
- ADAM's existing voice
- existing `Charon` voice configuration
- existing conversation context

Do not create a separate Gemini session for the confirmation.

---

# 17. Gemini audio path must remain unchanged

The existing path is:

```text
Gemini Live
 ↓
24 kHz mono audio
 ↓
existing speaker queue
 ↓
48 kHz stereo
 ↓
aplay
 ↓
MAX98357A
 ↓
speaker
```

Do not replace this.

The experiment is about **who decides the action**, not about replacing ADAM's voice system.

---

# 18. Timing instrumentation

Add high-resolution timestamps around every stage.

Use `time.perf_counter()`.

Record at minimum:

```text
T0 = speech starts
T1 = speech ends / ASR final
T2 = Needle inference starts
T3 = Needle tool call produced
T4 = local tool execution starts
T5 = local tool execution complete
T6 = result handed to Gemini
T7 = first Gemini audio byte/chunk received
T8 = first audio reaches speaker queue
T9 = speaker playback begins
```

Calculate:

```text
ASR latency      = T1 - T0
Needle latency   = T3 - T2
Tool latency     = T5 - T4
Gemini reaction  = T7 - T6
Audio startup    = T9 - T7
Total perceived  = T9 - T0
```

Also measure the existing Gemini-only architecture:

```text
T0 → Gemini first audio
```

This gives us a real comparison rather than guessing.

---

# 19. Required benchmark

Run at least 20 trials for each mode.

## Mode A — current Gemini-only

```text
speech
→ Gemini Live
→ Gemini tool call
→ laptop action
→ Gemini response
→ ADAM voice
```

## Mode B — Needle local action + Gemini response

```text
speech
→ local ASR
→ Needle
→ laptop action
→ existing Gemini session
→ ADAM voice
```

## Mode C — experimental parallel architecture

```text
speech
→ local ASR + Gemini Live simultaneously
→ Needle action
→ result handoff
→ Gemini voice
```

Do NOT allow duplicate actions.

If Mode C cannot guarantee single execution, mark it as unsafe and do not continue with real laptop actions.

---

# 20. Test commands

At minimum:

### Volume

```text
increase the volume
decrease the volume
set the volume to 80 percent
turn the volume up
make my laptop louder
mute the laptop
unmute the laptop
```

### Brightness

```text
increase brightness
decrease brightness
set brightness to 60 percent
make the screen brighter
```

### Non-local conversation

```text
who are you
tell me a joke
what is the capital of Japan
why is the sky blue
tell me something interesting
```

The non-local commands are important.

Needle must NOT hijack general conversation.

---

# 21. Expected behavior

### Local action

User:

> "ADAM, increase the volume."

Expected:

```text
Needle
 ↓
volume_up
 ↓
laptop changes volume
 ↓
actual result = 80%
 ↓
Gemini Live
 ↓
ADAM voice:
"Yeah, the volume is set to 80%."
```

### General conversation

User:

> "ADAM, why is the sky blue?"

Expected:

```text
Needle
 ↓
no local action
 ↓
Gemini Live
 ↓
normal ADAM response
```

### Failed action

If the laptop is unavailable:

```text
Needle
 ↓
volume_up
 ↓
laptop agent unavailable
 ↓
{
  "status": "error",
  ...
}
 ↓
Gemini
 ↓
ADAM voice:
"I couldn't reach your laptop."
```

Never claim success when the tool failed.

---

# 22. Confidence handling

Needle 2 may expose confidence information.

If available, log it.

Do not immediately trust a confidence threshold without measuring it.

Start with:

```text
confidence → log only
```

Then evaluate:

- false positives
- false negatives
- ambiguous commands

Only after collecting results should a threshold be introduced.

A false local action is more dangerous than sending a request to Gemini.

---

# 23. Ambiguous commands

Test:

```text
make it louder
do that again
a little more
turn it up
that's too loud
make the screen better
```

Determine whether Needle has enough context.

Do NOT assume the 256-token sliding context is sufficient for ADAM's full conversational context.

If the command requires conversation context, it may need Gemini.

---

# 24. Resource monitoring

The Pi Zero 2W is a constrained 2-core device.

During testing monitor:

```bash
top
htop
free -h
vmstat 1
```

Record:

- CPU %
- RAM %
- swap
- load average
- temperature
- Needle memory
- Gemini process memory
- audio glitches
- dropped audio chunks
- ASR delays

Needle must not starve:

- audio capture
- Gemini networking
- speaker playback
- UART/ESP32 communication
- TFT/emotion control

---

# 25. Audio starvation test

This is especially important.

The current ADAM architecture has many concurrent tasks.

Needle inference must not block the asyncio event loop.

If the Needle Python API is synchronous:

```python
result = needle(...)
```

do NOT call it directly inside a latency-sensitive asyncio task if it blocks.

Use an appropriate worker/thread/process strategy after measuring its CPU behavior.

The existing audio pipeline must continue streaming.

Test:

- no audio dropout
- no robotic/stuttering Gemini voice
- no microphone queue overflow
- no speaker queue starvation
- no UART starvation

---

# 26. Failure/fallback architecture

The final experimental implementation must support:

```text
Needle unavailable
      ↓
Gemini-only
```

```text
Needle crashes
      ↓
Gemini-only
```

```text
Needle confidence low
      ↓
Gemini
```

```text
local tool fails
      ↓
Gemini explains failure
```

```text
Gemini unavailable
      ↓
local action may still work
```

Do not make ADAM completely dependent on Needle.

---

# 27. Files to create

Prefer isolated files first:

```text
tools/needle_test/
    needle_smoke_test.py
    needle_benchmark.py
    needle_laptop_test.py
    needle_gemini_handoff_test.py

docs/needle_test/
    00_environment.md
    01_needle_compatibility.md
    02_needle_benchmark.md
    03_laptop_agent_result_contract.md
    04_asr_to_needle.md
    05_gemini_handoff.md
    06_latency_comparison.md
    07_final_findings.md
```

Do not replace the main ADAM file until the isolated tests pass.

---

# 28. Git safety

Before modifying ADAM:

```bash
git status
git branch
git add .
git commit -m "checkpoint before Needle 2 experiment"
```

Then create:

```bash
git checkout -b experiment/needle2-gemini-hybrid
```

If the repository already has a different branching convention, follow it instead.

---

# 29. Success criteria

The experiment is considered successful only if ALL of the following are true:

### Compatibility

- Needle runs on Pi Zero 2W.
- No SIGILL.
- No native crashes.
- Stable for extended testing.

### Tool calling

- Local commands are recognized reliably.
- Correct tool is selected.
- Arguments are correct.
- False activations are acceptably low.

### Laptop control

- Existing laptop agent remains functional.
- Actual state is returned.
- Success/failure is correctly reported.

### Gemini

- Existing Gemini Live session remains connected.
- Existing ADAM voice remains unchanged.
- Existing personality remains unchanged.
- Gemini can receive the local action result.
- Gemini produces the confirmation naturally.

### Audio

- No noticeable audio stuttering.
- No dropped chunks caused by Needle.
- Existing speaker path remains intact.

### Latency

The Needle architecture must be compared against the existing Gemini-only path using real measurements.

Do not claim "zero latency."

The target is:

> **minimize the time between the user's speech ending, local action completion, and the first byte of ADAM's Gemini-generated confirmation audio.**

---

# 30. Final report

At the end create:

```text
docs/needle_test/07_final_findings.md
```

Include:

## A. Does Needle 2 run on Pi Zero 2W?

YES / NO / UNSTABLE

## B. Needle average inference latency

Measured value.

## C. Needle p95 latency

Measured value.

## D. Local tool accuracy

Measured percentage.

## E. Gemini-only latency

Measured values.

## F. Needle + Gemini latency

Measured values.

## G. Parallel architecture latency

Measured values, only if safe.

## H. CPU/RAM impact

Measured values.

## I. Audio quality

Pass / Fail.

## J. Duplicate-action safety

Pass / Fail.

## K. Recommendation

Choose one:

```text
KEEP GEMINI ONLY
```

or

```text
USE NEEDLE FOR SELECTED LOCAL ACTIONS
```

or

```text
ADOPT NEEDLE + GEMINI HYBRID
```

or

```text
DO NOT USE NEEDLE ON PI ZERO 2W
```

Explain the decision using measured results.

---

# 31. Important implementation philosophy

Do not optimize for the smallest code change.

Optimize for:

1. correctness
2. reversibility
3. measurable latency
4. no duplicate actions
5. no audio degradation
6. preserving ADAM's existing voice/personality
7. graceful fallback

The objective is not merely:

> "Make Needle work."

The objective is:

> **Determine whether Needle can become ADAM's local reflex/action brain while Gemini Live remains ADAM's conversational intelligence and voice.**

---

# 32. Final target architecture

If the experiment succeeds, the target architecture is:

```text
                         ┌─────────────────────┐
                         │      ADAM Pi        │
                         │    Raspberry Pi     │
                         │      Zero 2 W       │
                         └──────────┬──────────┘
                                    │
                                  MIC
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
              LOCAL ASR                        GEMINI LIVE
                    │                        persistent session
                    ▼                               │
                NEEDLE 2                            │
             local action brain                     │
                    │                               │
                    ▼                               │
               TOOL BUS                             │
             ┌──────┼─────────┐                     │
             │      │         │                     │
             ▼      ▼         ▼                     │
          Laptop  GPIO      Memory                  │
          Agent   /Servo    /etc.                   │
             │      │         │                     │
             └──────┴─────────┘                     │
                    │                               │
                    ▼                               │
              ACTUAL RESULT                         │
                    │                               │
                    └──────────────►────────────────┘
                                            │
                                            ▼
                                     Gemini response
                                            │
                                            ▼
                                      Charon voice
                                            │
                                            ▼
                                         Speaker
```

**Do not implement the final architecture blindly.**

Use this document as a controlled experiment.

The first deliverable is proof that Needle 2 is stable and useful on the actual Pi Zero 2W.

The second deliverable is proof that local actions can be completed faster than the current Gemini-only path.

The third deliverable is proof that the completed local action can be handed to the existing Gemini Live session without changing ADAM's voice or personality and without causing duplicate tool execution.

Only then integrate it into the main ADAM application.
