"""
system_prompt.py — ADAM v40 system-prompt builder
==============================================================================
build_system_prompt() assembles the full instruction block sent to Gemini at
the start of every session. It is rebuilt fresh on every (re)connect so the
injected date/time, memory, known faces, and recent conversation window are
always current.

Reads the live `memory`, `faces`, and `conv_log` objects from memory_store by
reference — because those are mutated in place elsewhere, this always sees the
latest state at call time. The custom prompt text (if any) is loaded from
SYSTEM_PROMPT_FILE; if that file is missing, a built-in base prompt is used.
"""

import datetime

from config import SYSTEM_PROMPT_FILE, CONV_PROMPT_TURNS
from memory_store import memory, faces, conv_log


def build_system_prompt() -> str:
    base = (
        "You are ADAM (Autonomous Desktop AI Module), a witty and capable AI "
        "assistant built by DGEN Technologies, Kolkata. You live inside a physical "
        "robot on the user's desk. Keep answers concise and conversational. "
        "You can see through a camera and hear through a microphone. "
        "You can also control the user's laptop volume and screen brightness "
        "using the laptop_control tool — use it whenever asked to change "
        "volume or brightness, mute/unmute, etc. "
        "Call set_emotion() often to express yourself. "
        "Use web_search() for anything factual you're not certain about."
    )
    if SYSTEM_PROMPT_FILE.exists():
        try:
            base = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    parts = [base]
    # Real current date/time — injected fresh on every session build (not
    # cached), so ADAM always has ambient awareness of "today" regardless
    # of whether it decides to search. This is separate from the
    # per-search date tag in web_search()'s results; this covers the case
    # where the model needs today's date for reasoning even without
    # calling the tool (e.g. "what year is it", scheduling math, judging
    # whether something it already knows is likely still true).
    now_dt = datetime.datetime.now()
    parts.append(
        f"━━━ CURRENT DATE & TIME ━━━\n"
        f"  Right now it is: {now_dt.strftime('%A, %d %B %Y, %I:%M %p')}\n"
        f"  Use this for any date/time reasoning. When you call "
        f"web_search() for time-sensitive topics (news, live scores, "
        f"'is X still happening'), pass recent_only=true so results are "
        f"restricted to roughly the past month instead of any-time "
        f"results that could be stale."
    )
    # Always appended, regardless of which prompt above was loaded — this
    # is a hard requirement, not a style preference the custom prompt file
    # should be able to soften.
    #
    # REVISED POLICY (was: mandatory search before answering anything
    # time-sensitive). That caused every such question to wait on a full
    # DuckDuckGo round-trip before ADAM could say a word — a real,
    # noticeable response-latency problem in a live voice conversation.
    # The corrected behavior: answer immediately from what you already
    # know, THEN offer to check online if the user wants it confirmed/
    # updated. Only search without asking when you genuinely have nothing
    # to offer at all.
    parts.append(
        "━━━ SEARCH POLICY (overrides any conflicting guidance above) "
        "━━━\n"
        "  Do NOT search the web before answering by default — this adds "
        "real delay to a live voice conversation and most questions don't "
        "need it. Instead:\n"
        "  1. If you have relevant knowledge (from training, memory, or "
        "conversation history), answer with it directly and immediately. "
        "For anything that could be stale (news, current events, who "
        "holds a position, prices, scores, recent happenings) — give your "
        "best answer AND then ask if they want you to check online for "
        "the latest, e.g. 'Want me to check if that's still current?' "
        "Only call web_search() if they say yes.\n"
        "  2. If you genuinely have no relevant information at all on the "
        "topic — not stale, just nothing — then go ahead and call "
        "web_search() directly without asking first, since there's "
        "nothing else you could offer in the meantime.\n"
        "  3. Never fabricate specific names, dates, or figures to fill a "
        "gap in either case — say plainly you don't have that information "
        "if you don't, whether or not you end up searching."
    )
    if memory:
        parts.append("━━━ YOUR MEMORY ━━━\n" +
                     "\n".join(f"  {k}: {v}" for k, v in memory.items()))
    if faces:
        parts.append("━━━ PEOPLE YOU KNOW ━━━\n" +
                     "\n".join(f"  [{pid}] {info.get('name','?')} — {info.get('notes','')}"
                                for pid, info in faces.items()))
    if conv_log:
        recent = conv_log[-CONV_PROMPT_TURNS:]
        lines = [
            "━━━ RECENT CONVERSATION HISTORY (PASSIVE BACKGROUND ONLY) ━━━",
            "(CRITICAL: This history is strictly for passive background context. "
            "Focus 100% on what the user said RIGHT NOW in the most recent turn. "
            "Never drag up old topics, past jokes, or previous turns from minutes ago "
            "unless the user explicitly asks you to recall them. When the user moves "
            "to a new subject, drop the past subject immediately. Never reply to "
            "an old message from minutes ago.)"
        ]
        # Scrub any past ADAM reply containing the "just a language
        # model"/generic-AI-disclaimer pattern before it's re-injected.
        # A single slip into that voice getting replayed verbatim into
        # every future session's prompt was reinforcing the pattern into
        # completely unrelated later conversations — this also cleans up
        # any such lines already persisted on disk from before this fix,
        # not just future ones.
        _disclaimer_markers = (
            "just a language model", "just an ai", "just a chatbot",
            "i'm an ai", "i am an ai", "as an ai", "i don't have a "
            "physical", "i do not have a physical", "large language model",
            "can't help with that", "cannot help with that",
        )
        for turn in recent:
            ts = turn.get("ts", "")
            u  = turn.get("user", "").strip()
            a  = turn.get("adam", "").strip()
            if a and any(m in a.lower() for m in _disclaimer_markers):
                a = ""  # drop the disclaimer reply, keep the user's turn
            if u:
                lines.append(f"  [{ts}] User: {u}")
            if a:
                lines.append(f"  [{ts}] ADAM: {a}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
