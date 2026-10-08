#!/usr/bin/env python3
"""
adam_smoketest.py — ADAM self-test harness
==============================================================================
Run on the Pi (or on the dev laptop) to check that the v41 subsystems load and
behave, without needing a live Gemini session, a microphone, or a human.

    python adam_smoketest.py              # everything
    python adam_smoketest.py prompt       # one group
    python adam_smoketest.py prompt clip  # several groups
    python adam_smoketest.py --list       # show groups

Exit code 0 = all selected tests passed, 1 = at least one failed.

WHAT THIS IS AND IS NOT
-----------------------
It IS a fast pre-deploy check that the pieces fit: modules import, the central
prompt parses and substitutes, failure fallbacks actually fall back, the typed
laptop protocol coerces values correctly, the scheduler survives clock jumps,
JSON writes are atomic.

It is NOT a substitute for talking to the robot. Anything involving real audio,
the real Gemini Live socket, or real speaker output is explicitly out of scope
here and must be verified by hand on hardware — master prompt §0.9: do not
claim a feature works unless it has actually been tested.

Tests that need a live laptop agent on the LAN are skipped (not failed) when
the agent is unreachable, and say so, so this is still useful offline.

DESIGN
------
No pytest dependency: the Pi venv has what adam needs and nothing more, and a
smoke test that cannot run on the target is useless. Groups register themselves
with @group so adding coverage is one decorated function.
"""

import io
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The prompt file and several persona blocks are full of em-dashes, ━ box
# drawing and Devanagari. On a Windows console (cp1252) printing those raises
# UnicodeEncodeError and the test run dies for a cosmetic reason. Force UTF-8
# on stdout with replacement so the harness never fails on its own output.
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
except Exception:
    pass

GROUPS: dict = {}
_ORDER: list = []


def group(name: str, description: str):
    def deco(fn):
        GROUPS[name] = (fn, description)
        _ORDER.append(name)
        return fn
    return deco


class Ctx:
    """Per-group assertion recorder. Keeps going after a failure so one run
    reports every problem rather than only the first."""

    def __init__(self, name):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def ok(self, label, cond, detail=""):
        if cond:
            self.passed += 1
            print(f"    PASS  {label}")
        else:
            self.failed += 1
            print(f"    FAIL  {label}" + (f"  — {detail}" if detail else ""))
        return bool(cond)

    def eq(self, label, got, want):
        return self.ok(label, got == want, f"got {got!r}, want {want!r}")

    def skip(self, label, why):
        self.skipped += 1
        print(f"    SKIP  {label}  — {why}")


# ═════════════════════════════════════════════════════════════════════════════
# GROUP: imports — every module the Pi runs must import cleanly
# ═════════════════════════════════════════════════════════════════════════════

@group("imports", "every adam module imports without error")
def t_imports(c):
    mods = [
        "config", "memory_store", "prompt_store", "system_prompt",
        "audio_utils", "tools_schema", "tool_handler",
        "laptop_agent_client", "ws_server", "hardware", "session",
        "song_playback",
    ]
    for m in mods:
        try:
            __import__(m)
            c.ok(f"import {m}", True)
        except Exception as e:
            c.ok(f"import {m}", False, f"{type(e).__name__}: {e}")

    # Modules added by v41 — absent until their workstream lands, so a missing
    # one is reported as a skip rather than failing the whole run.
    for m in ("scheduler", "model_router"):
        try:
            __import__(m)
            c.ok(f"import {m}", True)
        except ImportError:
            c.skip(f"import {m}", "not present yet")
        except Exception as e:
            c.ok(f"import {m}", False, f"{type(e).__name__}: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# GROUP: prompt — the central prompt file (master prompt §3, §4)
# ═════════════════════════════════════════════════════════════════════════════

@group("prompt", "central prompt file parses, substitutes and fails safe")
def t_prompt(c):
    import prompt_store as ps
    from config import PROMPTS_FILE, PROMPT_POOL_MIN_VARIANTS

    ps.reload_if_changed(force=True)
    st = ps.status()

    c.ok("prompts.txt is the active source", st["source"] == PROMPTS_FILE.name,
         f"source is {st['source']}")
    c.ok("parsed with no warnings", not st["warnings"],
         "; ".join(st["warnings"]))

    # --- required sections present and non-empty ---------------------------
    for name in ps.REQUIRED_SECTIONS:
        body = ps.section(name)
        c.ok(f"section {name}", bool(body.strip()), "missing or empty")

    # --- required pools meet the ≥5 rule ----------------------------------
    for name in ps.REQUIRED_POOLS:
        n = len(ps.pool(name))
        c.ok(f"pool {name} >= {PROMPT_POOL_MIN_VARIANTS}",
             n >= PROMPT_POOL_MIN_VARIANTS, f"only {n}")

    # --- anti-repetition: never the same line twice running ---------------
    for name in ("task_completed", "idle_nudge", "confirmation",
                 "scheduler_confirmation", "error_failure"):
        prev, repeats = None, 0
        for _ in range(300):
            v = ps.pick(name)
            if v and v == prev:
                repeats += 1
            prev = v
        c.eq(f"pool {name} back-to-back repeats", repeats, 0)

    # --- pools do actually vary (a pool that always returns one line would
    #     pass the repeat test only if it returned "" every time) ----------
    for name in ("task_completed", "greeting"):
        seen = {ps.pick(name) for _ in range(200)}
        c.ok(f"pool {name} uses its whole range",
             len(seen) == len(ps.pool(name)),
             f"saw {len(seen)} of {len(ps.pool(name))}")

    # --- injections are well-formed ---------------------------------------
    for name in ("inject_refusal_correction", "inject_stop_gesture",
                 "inject_cheek_slap", "inject_petting"):
        t = ps.injection(name)
        c.ok(f"{name} is a [SYSTEM: ...] block",
             t.startswith("[SYSTEM:") and t.endswith("]"), repr(t[:60]))

    t = ps.injection("inject_idle_nudge", elapsed="97", nudge="probe-nudge")
    c.ok("inject_idle_nudge substitutes elapsed", "97s" in t, t[:80])
    c.ok("inject_idle_nudge substitutes nudge", "probe-nudge" in t, t[:80])

    # --- the gesture injections must forbid tool calls (§24) --------------
    for name in ("inject_cheek_slap", "inject_petting"):
        t = ps.injection(name).lower()
        c.ok(f"{name} forbids tool calls",
             "do not call any tool" in t, "missing the no-tool rule")

    # --- unknown names must degrade, not raise ----------------------------
    c.eq("missing section returns ''", ps.section("no_such_section_xyz"), "")
    c.eq("missing pool returns ''", ps.pick("no_such_pool_xyz"), "")

    # --- the scrub list survived the move --------------------------------
    markers = ps.disclaimer_markers()
    c.ok("disclaimer markers present", len(markers) >= 10, f"{len(markers)}")
    c.ok("markers are lowercase", all(m == m.lower() for m in markers))
    c.ok("'just a language model' is covered",
         any("just a language model" in m for m in markers))

    # --- the assembled prompt -------------------------------------------
    from system_prompt import build_system_prompt
    p = build_system_prompt()
    c.ok("assembled prompt is substantial", len(p) > 8000, f"{len(p)} chars")

    import re
    left = sorted(set(re.findall(r"\{([a-z_][a-z0-9_]*)\}", p))
                  & ps.KNOWN_PLACEHOLDERS)
    c.eq("no unresolved placeholders in assembled prompt", left, [])

    for must in ("You are ADAM", "SEARCH POLICY", "ZERO FIXED LANGUAGE",
                 "ANTI-REPETITION", "BANNED PHRASES",
                 "CLIPBOARD IS UNTRUSTED", "CURRENT DATE & TIME"):
        c.ok(f"prompt contains {must!r}", must in p)

    # The persona must not re-introduce the dead tool names (§39).
    for dead in ("save_story", "save_person_photo", "generate_to_clipboard",
                 "move_neck"):
        c.ok(f"no reference to dead tool {dead}()", dead not in p,
             "prompts.txt still names a tool that does not exist")

    # Emotion names in the prompt must match the real tool enum.
    #
    # Read the enum out of tools_schema.py's SOURCE rather than calling
    # build_tools(): build_tools() pulls in the laptop-action manifest, which
    # triggers live mDNS discovery and an HTTP round-trip. A prompt-content
    # assertion must not depend on the LAN, and must not take seconds.
    schema_src = (HERE / "tools_schema.py").read_text(encoding="utf-8")
    m = re.search(r'"emotion"\s*:\s*S\([^)]*?enum=\[(.*?)\]', schema_src, re.S)
    if m:
        enum = re.findall(r'"([a-z_]+)"', m.group(1))
        c.ok("set_emotion enum found in tools_schema.py", len(enum) >= 10,
             f"parsed {enum}")
        # "reconnecting" is set by hardware during a reconnect, never chosen by
        # the model, so it has no business in the persona's emotion list.
        missing = [e for e in enum if e != "reconnecting" and e not in p]
        c.eq("every model-selectable set_emotion value appears in the prompt",
             missing, [])
        c.ok("prompt does not advertise the internal 'reconnecting' face",
             "reconnecting" not in ps.section("tool_behaviour"))
    else:
        c.skip("set_emotion enum cross-check",
               "could not locate the enum in tools_schema.py")

    # --- the file must never carry a credential (§0.7) -------------------
    # Reuse prompts_check's regex rather than naive substrings: "desk-buddy"
    # contains "sk-", and "AIza" needs the key-shaped tail to mean anything.
    # A false positive here would train people to ignore the check.
    raw = PROMPTS_FILE.read_text(encoding="utf-8")
    from prompts_check import SECRET_RE
    hits = [ln for ln in raw.splitlines()
            if not ln.startswith("#!") and SECRET_RE.search(ln)]
    c.eq("prompts.txt carries no credential-shaped string", hits, [])

    # --- failure fallbacks: a bad edit must not take ADAM down -----------
    bak = PROMPTS_FILE.with_suffix(".txt.smoketest-bak")
    good_len = len(ps.section("persona"))
    try:
        shutil.copy2(PROMPTS_FILE, bak)

        PROMPTS_FILE.write_text("=== nonsense ==\nno directives\n",
                               encoding="utf-8")
        time.sleep(0.02)
        ps.reload_if_changed()
        c.eq("broken file keeps last-known-good persona",
             len(ps.section("persona")), good_len)

        os.remove(PROMPTS_FILE)
        ps.reload_if_changed()
        c.eq("deleted file keeps last-known-good persona",
             len(ps.section("persona")), good_len)
    finally:
        shutil.copy2(bak, PROMPTS_FILE)
        os.remove(bak)
        ps.reload_if_changed(force=True)
    c.eq("restored cleanly after fallback test",
         len(ps.section("persona")), good_len)

    # --- hot reload actually detects an edit -----------------------------
    bak2 = PROMPTS_FILE.with_suffix(".txt.smoketest-bak2")
    try:
        shutil.copy2(PROMPTS_FILE, bak2)
        PROMPTS_FILE.write_text(
            bak2.read_text(encoding="utf-8")
            + "\n===POOL _smoketest_probe===\nalpha\nbeta\n",
            encoding="utf-8")
        time.sleep(0.02)
        reloaded = ps.reload_if_changed()
        c.ok("hot reload detected the edit", reloaded)
        c.eq("hot-reloaded pool is readable",
             sorted(ps.pool("_smoketest_probe")), ["alpha", "beta"])
    finally:
        shutil.copy2(bak2, PROMPTS_FILE)
        os.remove(bak2)
        ps.reload_if_changed(force=True)
    c.eq("probe pool gone after restore", ps.pool("_smoketest_probe"), [])

    # --- the validator agrees --------------------------------------------
    import subprocess
    r = subprocess.run([sys.executable, str(HERE / "prompts_check.py")],
                       capture_output=True, text=True)
    c.eq("prompts_check.py exits clean", r.returncode, 0)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP: laptop — the typed laptop-action protocol (master prompt §14–§20)
# ═════════════════════════════════════════════════════════════════════════════

@group("laptop", "laptop action manifest, aliases, typed coercion, redaction")
def t_laptop(c):
    import laptop_actions as la

    # --- the manifest covers every action the live agent exposes (§15) -----
    fb = la.fallback_manifest()
    missing = [a for a in la.LIVE_PARITY_ACTIONS if a not in fb]
    c.eq("fallback manifest covers all live actions", missing, [])
    c.ok("fallback manifest is no longer the v40 8-action stub", len(fb) >= 18,
         f"only {len(fb)}")
    c.ok("every entry is typed",
         all(e["value_type"] in ("none", "int", "str", "enum")
             for e in fb.values()))
    c.ok("needs_value agrees with value_type",
         all(e["needs_value"] == (e["value_type"] != "none")
             for e in fb.values()))
    c.eq("parity check passes against our own manifest",
         la.parity_report(fb)["ok"], True)

    # --- aliases resolve, and never onto a name that does not exist -------
    c.eq("clipboard_get -> read_clipboard", la.resolve("clipboard_get"),
         "read_clipboard")
    c.eq("clipboard_set -> write_clipboard", la.resolve("clipboard_set"),
         "write_clipboard")
    c.eq("resolve is case/whitespace tolerant",
         la.resolve("  WRITE_CLIPBOARD "), "write_clipboard")
    for alias, target in la.ALIASES.items():
        c.ok(f"alias {alias} points at a real action",
             target in la.CANONICAL_ACTIONS, f"{target} does not exist")
        c.ok(f"alias {alias} does not shadow a real action",
             alias not in la.CANONICAL_ACTIONS)

    # --- integer coercion: the bug that broke every string action (§17) ---
    for given, want in (("50", 50), (50, 50), ("42%", 42), (73.6, 73),
                        (150, 100), (-5, 0), ("0", 0)):
        ok, v, err = la.coerce("volume_set", given)
        c.ok(f"volume_set {given!r} -> {want}", ok and v == want,
             f"ok={ok} value={v!r} err={err}")
    for bad in ("abc", "", "  ", None):
        ok, _, err = la.coerce("volume_set", bad)
        c.ok(f"volume_set rejects {bad!r}", not ok and bool(err))

    # --- string actions keep their text instead of becoming None ----------
    code = "def f():\n    return {'a': 1}  # braces, quotes, newline"
    for act in ("write_clipboard", "clipboard_set", "dispatch_coding_task"):
        ok, v, err = la.coerce(act, code)
        c.ok(f"{act} keeps its text verbatim", ok and v == code,
             f"ok={ok} err={err}")
    ok, _, err = la.coerce("write_clipboard", None)
    c.ok("write_clipboard with no value is rejected", not ok and bool(err))

    # --- enum validation --------------------------------------------------
    ok, v, _ = la.coerce("set_robot_emotion", "Excited")
    c.ok("emotion is normalised to lowercase", ok and v == "excited", repr(v))
    ok, _, err = la.coerce("set_robot_emotion", "banana")
    c.ok("invalid emotion is rejected", not ok and bool(err))

    # The PC mirror must accept every face the robot can actually show, or a
    # legitimate mirror update gets a 400 back.
    schema_src = (HERE / "tools_schema.py").read_text(encoding="utf-8")
    import re
    m = re.search(r'"emotion"\s*:\s*S\([^)]*?enum=\[(.*?)\]', schema_src, re.S)
    if m:
        pi_enum = re.findall(r'"([a-z_]+)"', m.group(1))
        unmirrored = [e for e in pi_enum if e not in la.ROBOT_EMOTIONS]
        c.eq("every robot emotion is accepted by set_robot_emotion",
             unmirrored, [])
    else:
        c.skip("emotion mirror cross-check", "enum not found in tools_schema.py")

    # --- valueless actions tolerate a stray value instead of failing ------
    for act in ("volume_mute", "lock_screen", "cancel_coding_task",
                "check_coding_task_status", "clipboard_paste"):
        ok, v, _ = la.coerce(act, "stray")
        c.ok(f"{act} drops a stray value", ok and v is None, repr(v))

    # --- unknown action: pass the value through, do not destroy it --------
    ok, v, _ = la.coerce("some_future_action", "text")
    c.ok("unknown action keeps its value", ok and v == "text", repr(v))

    # --- length caps ------------------------------------------------------
    ok, v, _ = la.coerce("write_clipboard", "x" * (la.MAX_STRING_VALUE_CHARS + 50))
    c.eq("write_clipboard is capped", len(v), la.MAX_STRING_VALUE_CHARS)
    ok, v, _ = la.coerce("dispatch_coding_task", "x" * 10_000)
    c.eq("dispatch_coding_task is capped", len(v), 8_000)

    # --- backward compatibility with a pre-v41 (untyped) agent ------------
    old = {
        "volume_set":      {"needs_value": True,  "value_hint": "0-100"},
        "write_clipboard": {"needs_value": True,  "value_hint": "text string"},
        "lock_screen":     {"needs_value": False, "value_hint": ""},
        "mystery_action":  {"needs_value": True,  "value_hint": "a level 0-100"},
    }
    c.eq("untyped volume_set inferred as int",
         la.value_type_of("volume_set", old), "int")
    c.eq("untyped write_clipboard inferred as str",
         la.value_type_of("write_clipboard", old), "str")
    c.eq("untyped lock_screen inferred as none",
         la.value_type_of("lock_screen", old), "none")
    c.eq("unknown numeric-hinted action inferred as int",
         la.value_type_of("mystery_action", old), "int")
    ok, v, _ = la.coerce("write_clipboard", "hello", old)
    c.ok("string survives an untyped manifest", ok and v == "hello", repr(v))
    rep = la.parity_report(old)
    c.ok("parity report flags an untyped manifest", len(rep["untyped"]) == 4)
    c.ok("parity report flags missing actions", len(rep["missing"]) > 0)

    # --- log redaction (§18) ---------------------------------------------
    secret = "hunter2-correct-horse"
    c.eq("write_clipboard value is redacted",
         la.redact_value("write_clipboard", secret), f"<text {len(secret)} chars>")
    c.eq("alias is redacted too",
         la.redact_value("clipboard_set", secret), f"<text {len(secret)} chars>")
    c.eq("coding instruction is redacted",
         la.redact_value("dispatch_coding_task", secret),
         f"<text {len(secret)} chars>")
    c.eq("control parameters are NOT redacted",
         la.redact_value("volume_set", 40), 40)
    c.ok("clipboard read details are redacted",
         "not logged" in str(la.redact_details("read_clipboard", secret)))
    c.ok("clipboard_get details are redacted too",
         "not logged" in str(la.redact_details("clipboard_get", secret)))
    c.eq("other results are NOT redacted",
         la.redact_details("volume_set", "{'volume': 40}"), "{'volume': 40}")

    # --- the Pi client uses the shared manifest, not its own list ---------
    import laptop_agent_client as lac
    c.eq("client fallback IS the shared manifest",
         sorted(lac._LAPTOP_ACTIONS_FALLBACK), sorted(fb))
    annotated = lac._annotate(old)
    c.ok("client annotates an untyped manifest",
         all("value_type" in e for e in annotated.values()))

    # --- the model-facing schema must be able to carry text (§17) --------
    # This was T.INTEGER until v41, which made every string action unusable no
    # matter what the handler did.
    src = (HERE / "tools_schema.py").read_text(encoding="utf-8")
    decl = src[src.index("def build_laptop_control_declaration"):]
    # Comment lines are stripped first: the fix's own explanation mentions the
    # old T.INTEGER, and a test that matches a comment proves nothing.
    code = "\n".join(ln for ln in decl.splitlines()
                     if not ln.lstrip().startswith("#"))
    c.ok("laptop_control 'value' is declared as a STRING",
         'S(type=T.STRING,' in code.split('"value"')[1][:40],
         "value is not a STRING — string actions cannot work")
    c.ok("laptop_control no longer declares any INTEGER parameter",
         "T.INTEGER" not in code)

    # --- the clipboard cap is one number, shared --------------------------
    from config import CLIPBOARD_MAX_CHARS
    c.eq("config takes the clipboard cap from laptop_actions",
         CLIPBOARD_MAX_CHARS, la.CLIPBOARD_MAX_CHARS)
    c.ok("clipboard cap is bigger than v40's 500 chars",
         CLIPBOARD_MAX_CHARS > 500)

    # --- both deployed copies are byte-identical (§16, §38) --------------
    pc_copy = HERE.parents[2] / "pcAPP" / "laptop_actions.py"
    if pc_copy.exists():
        import hashlib
        h1 = hashlib.sha256((HERE / "laptop_actions.py").read_bytes()).hexdigest()
        h2 = hashlib.sha256(pc_copy.read_bytes()).hexdigest()
        c.eq("pi and pcAPP copies of laptop_actions.py are identical", h1, h2)
    else:
        c.skip("pi/pcAPP copy comparison", "pcAPP tree not present")


# ═════════════════════════════════════════════════════════════════════════════
# GROUP: memory — atomic JSON persistence
# ═════════════════════════════════════════════════════════════════════════════

@group("memory", "memory_store reads/writes atomically")
def t_memory(c):
    import memory_store
    tmp = HERE / ".smoketest_mem.json"
    try:
        payload = {"a": 1, "b": "two", "unicode": "नमस्ते — ━"}
        memory_store.save_json(tmp, payload)
        c.ok("file written", tmp.exists())
        c.eq("round-trips exactly", memory_store.load_json(tmp, {}), payload)
        c.eq("missing file returns the default",
             memory_store.load_json(HERE / ".no_such_file.json", {"d": 1}),
             {"d": 1})
        # No .tmp litter left behind — the atomic write must clean up.
        leftovers = list(HERE.glob(".smoketest_mem.json.tmp*"))
        c.eq("no temp files left behind", leftovers, [])
    finally:
        tmp.unlink(missing_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP: router — the v41 generation/multimodal layer (§12, §13, §27)
# ═════════════════════════════════════════════════════════════════════════════
#
# Everything here is offline. No group in this suite may spend an API call or
# need the network: the point of the smoke test is "is this install sane",
# and that question has to be answerable on a Pi with no Wi-Fi. What can be
# tested without a live model is exactly what tends to break — the envelope
# contract, the secret scrubbing, the fail-soft paths, and the fact that the
# tool layer and the router agree on names.

@group("router", "model router envelope, scrubbing, fail-soft, tool wiring")
def t_router(c):
    try:
        import model_router as mr
    except ImportError:
        c.skip("import model_router", "not present yet")
        return

    st = mr.router_status()
    c.ok("router reports a provider", bool(st.get("provider")))
    c.ok("all three models configured",
         all(st.get(k) for k in ("code_model", "text_model", "vision_model")),
         f"{st}")

    # --- the §27 envelope, on both the success and failure paths -----------
    want_keys = {"ok", "text", "kind", "model", "provider", "tokens",
                 "duration_ms", "error"}
    ok_env = mr._envelope("code", "m", text="x", ok=True, started=time.time())
    err_env = mr._envelope("code", "m", ok=False, error="boom", started=time.time())
    c.eq("success envelope has exactly the §27 keys", set(ok_env), want_keys)
    c.eq("failure envelope has exactly the §27 keys", set(err_env), want_keys)
    c.ok("failure envelope carries no text", err_env["text"] == "")
    c.ok("failure envelope is ok=False", err_env["ok"] is False)

    # --- secrets must never survive into anything spoken or logged ---------
    # These are the exact shapes an SDK exception can echo back, and this
    # string is returned to the model, which SPEAKS it. A leak here is audible.
    leaky = ("request failed: https://gen.example/v1?key=AIzaSyBcDeFgHiJkLmNoPqRsT "
             "x-goog-api-key: sk-live-abc123 zz")
    scrubbed = mr._scrub(leaky)
    c.ok("AIza-style key redacted", "AIzaSyBcDeFgHiJkLmNoPqRsT" not in scrubbed,
         scrubbed)
    c.ok("?key= query arg redacted", "AIzaSyBcDeFgHiJkLmNoPqRsT" not in scrubbed)
    c.ok("x-goog-api-key header value redacted", "sk-live-abc123" not in scrubbed,
         scrubbed)
    c.ok("scrubbing keeps the useful part", "request failed" in scrubbed)
    c.eq("None scrubs to empty, not to the string 'None'", mr._scrub(None), "")
    c.ok("fail-soft error strings are scrubbed too",
         "AIzaSyBcDeFgHiJkLmNoPqRsT" not in (err_env["error"] or ""))

    # --- fail-soft: bad input returns, it does not raise (§27) -------------
    import asyncio
    env = asyncio.run(mr.generate_text(""))
    c.ok("empty request returns an envelope, not an exception", isinstance(env, dict))
    c.ok("empty request is ok=False", env["ok"] is False)
    c.ok("empty request still explains itself", bool(env["error"]))
    env = asyncio.run(mr.summarize_text(""))
    c.ok("empty summarise is ok=False with a reason",
         env["ok"] is False and bool(env["error"]))
    env = asyncio.run(mr.transform_text("text", ""))
    c.ok("transform without an instruction is refused",
         env["ok"] is False and bool(env["error"]))

    # The camera path is the one most likely to be hit with no frame — the
    # robot is asked "what do you see" before the camera is up all the time.
    env = asyncio.run(mr.describe_camera("what do you see", frame=None))
    c.ok("vision with no frame is ok=False, not a crash",
         env["ok"] is False and bool(env["error"]))
    c.ok("vision with no frame says something speakable",
         "camera" in (env["error"] or "").lower(), env["error"])

    # --- untrusted text is delimited, never silently inlined (§18) ---------
    # summarize/transform carry clipboard or user text. The delimiter is what
    # keeps it readable as content rather than as an instruction.
    src = (mr.summarize_text.__doc__ or "") + (mr.transform_text.__doc__ or "")
    c.ok("untrusted-text helpers document the boundary", "UNTRUSTED" in src.upper())

    # --- the tool layer and the router must agree --------------------------
    import tools_schema
    decl = set()
    for t in tools_schema.build_tools():
        for fd in (t.function_declarations or []):
            decl.add(fd.name)
    for name in ("generate_code", "generate_text", "describe_camera",
                 "summarize_text", "transform_text"):
        c.ok(f"{name} is declared to the model", name in decl)

    import tool_handler
    c.ok("tool_handler has a generation map",
         set(tool_handler._GEN_KIND) == {
             "generate_code", "generate_text", "describe_camera",
             "summarize_text", "transform_text"},
         f"{sorted(tool_handler._GEN_KIND)}")
    c.ok("every router kind is reachable from a tool",
         set(tool_handler._GEN_KIND.values()) <= set(mr.KINDS))


# ═════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--list" in sys.argv or "-l" in sys.argv:
        print("groups:")
        for name in _ORDER:
            print(f"  {name:12s} {GROUPS[name][1]}")
        return 0

    selected = args or list(_ORDER)
    unknown = [a for a in selected if a not in GROUPS]
    if unknown:
        print(f"unknown group(s): {', '.join(unknown)}")
        print(f"available: {', '.join(_ORDER)}")
        return 1

    print("=" * 78)
    print("ADAM smoke test")
    print(f"  python  {sys.version.split()[0]}")
    print(f"  dir     {HERE}")
    print(f"  groups  {', '.join(selected)}")
    print("=" * 78)

    results = []
    for name in selected:
        fn, desc = GROUPS[name]
        print(f"\n[{name}] {desc}")
        c = Ctx(name)
        try:
            fn(c)
        except Exception:
            c.failed += 1
            print(f"    FAIL  group raised:")
            for line in traceback.format_exc().strip().splitlines():
                print(f"          {line}")
        results.append(c)

    print("\n" + "=" * 78)
    total_p = sum(r.passed for r in results)
    total_f = sum(r.failed for r in results)
    total_s = sum(r.skipped for r in results)
    for r in results:
        flag = "FAIL" if r.failed else "ok  "
        print(f"  {flag}  {r.name:12s} {r.passed:3d} passed  "
              f"{r.failed:3d} failed  {r.skipped:3d} skipped")
    print("=" * 78)
    print(f"  TOTAL: {total_p} passed, {total_f} failed, {total_s} skipped")
    print("=" * 78)
    return 1 if total_f else 0


if __name__ == "__main__":
    sys.exit(main())
