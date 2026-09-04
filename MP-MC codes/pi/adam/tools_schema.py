"""
tools_schema.py — ADAM v40 Gemini tool/function declarations
==============================================================================
build_tools() returns the full list of function declarations exposed to the
Gemini Live model (get_current_datetime, get_sound_direction, enter_idle_mode,
move_head_gesture, play_song, set_emotion, save_memory, delete_memory,
get_memory, remember_person, web_search, and laptop_control).

The laptop_control declaration is built dynamically from the laptop agent's
live /actions manifest (via laptop_agent_client.get_laptop_actions()), so the
laptop itself decides which actions ADAM can offer, with a hard-coded fallback
when the manifest can't be fetched.
"""

from google.genai import types

from laptop_agent_client import get_laptop_actions


def build_tools() -> list:
    S, T = types.Schema, types.Type
    return [types.Tool(function_declarations=[

        types.FunctionDeclaration(
            name="get_current_datetime",
            description="Returns the current local date and time.",
            parameters=S(type=T.OBJECT, properties={})),

        types.FunctionDeclaration(
            name="get_sound_direction",
            description=(
                "Returns which direction the most recent speech came from "
                "(left/right/center, using the two onboard microphones). "
                "ONLY call this if the user EXPLICITLY asks something like "
                "'which direction am I talking from', 'can you tell where "
                "I am', or similar. Never call this proactively or mention "
                "direction unprompted — it's for direct questions only."
            ),
            parameters=S(type=T.OBJECT, properties={})),

        types.FunctionDeclaration(
            name="enter_idle_mode",
            description=(
                "Puts ADAM into a persistent silent/idle state — call this "
                "IMMEDIATELY when the user explicitly asks you to 'stay "
                "silent', 'stay mute', 'be quiet', 'stop talking', or "
                "similar. Once called, you will not speak or respond to "
                "anything — including scheduled idle nudges — until the "
                "user says your name again to wake you up. Do NOT call "
                "this for a normal request to pause mid-sentence; it's "
                "specifically for an extended silent mode."
            ),
            parameters=S(type=T.OBJECT, properties={})),

        types.FunctionDeclaration(
            name="move_head_gesture",
            description=(
                "Makes ADAM's neck perform a quick, human-like physical "
                "gesture. Use 'nod' for agreement/yes, 'shake' for "
                "disagreement/no, or when it adds natural physical "
                "expression to what you're saying (emphasis, reacting to "
                "something surprising, etc.). Don't overuse it — only "
                "when it genuinely fits the moment, not on every reply."
            ),
            parameters=S(type=T.OBJECT, properties={
                "gesture": S(type=T.STRING, enum=["nod", "shake"]),
            }, required=["gesture"])),

        types.FunctionDeclaration(
            name="play_song",
            description=(
                "Plays a song/audio track out loud through ADAM's speaker "
                "— call this when the user asks you to sing, perform, "
                "start a concert, or play music. One of several available "
                "songs is picked at random each time — you don't choose "
                "which. The mic is muted while the song plays (so it "
                "doesn't pick up the song itself), but everything else "
                "keeps running normally in parallel — camera, servos, "
                "conversation state are all unaffected. Playback runs "
                "until the song ends naturally OR the user taps Touch3 to "
                "stop it early. Say something short in character right "
                "before calling this (e.g. 'Alright, here we go!') since "
                "you'll go quiet once the song starts."
            ),
            parameters=S(type=T.OBJECT, properties={})),

        types.FunctionDeclaration(
            name="set_emotion",
            description=(
                "Display an emotion on ADAM's face. Call frequently to express reactions."
            ),
            parameters=S(type=T.OBJECT, properties={
                "emotion": S(type=T.STRING,
                             enum=["happy", "sad", "surprised", "angry",
                                   "thinking", "excited", "love", "blush",
                                   "confused", "smug", "sleep", "rizz",
                                   "panic", "shy", "reconnecting"])
            }, required=["emotion"])),

        types.FunctionDeclaration(
            name="save_memory",
            description="Permanently save a key-value fact.",
            parameters=S(type=T.OBJECT, properties={
                "key":   S(type=T.STRING),
                "value": S(type=T.STRING),
            }, required=["key", "value"])),

        types.FunctionDeclaration(
            name="delete_memory",
            description="Delete a saved memory entry by key.",
            parameters=S(type=T.OBJECT, properties={
                "key": S(type=T.STRING),
            }, required=["key"])),

        types.FunctionDeclaration(
            name="get_memory",
            description="Retrieve a specific memory entry or all entries.",
            parameters=S(type=T.OBJECT, properties={
                "key": S(type=T.STRING, description="Omit to get all entries"),
            })),

        types.FunctionDeclaration(
            name="remember_person",
            description="Save a person to permanent visual memory.",
            parameters=S(type=T.OBJECT, properties={
                "person_id":    S(type=T.STRING),
                "name":         S(type=T.STRING),
                "appearance":   S(type=T.STRING),
                "relationship": S(type=T.STRING),
                "notes":        S(type=T.STRING),
            }, required=["person_id", "name"])),

        types.FunctionDeclaration(
            name="web_search",
            description=(
                "Search the internet via DuckDuckGo for real-time information. "
                "Results are automatically tagged with today's actual date so "
                "you can judge whether they're current. "
                "DO NOT call this before every answer — that adds real delay "
                "to a live voice conversation. Correct usage: (1) answer "
                "first from what you already know, then ask the user if "
                "they want you to check online for the latest info — only "
                "call this tool if they confirm yes; OR (2) call it directly "
                "without asking ONLY when you have genuinely no relevant "
                "information at all to offer. If web_search returns nothing "
                "useful, say plainly that you couldn't find a reliable "
                "answer instead of inventing plausible-sounding details, "
                "names, or dates."
            ),
            parameters=S(type=T.OBJECT, properties={
                "query": S(type=T.STRING),
                "recent_only": S(
                    type=T.BOOLEAN,
                    description=(
                        "Set true for genuinely time-sensitive queries "
                        "(live scores, breaking news, 'is X still "
                        "happening') to restrict results to roughly the "
                        "past month instead of any-time results. Leave "
                        "false/omit for general facts that don't need "
                        "that restriction."
                    )),
            }, required=["query"])),

        build_laptop_control_declaration(),

    ])]


def build_laptop_control_declaration() -> types.FunctionDeclaration:
    S, T = types.Schema, types.Type
    actions = get_laptop_actions()
    action_names = list(actions.keys())

    lines = []
    for name, spec in actions.items():
        if spec.get("needs_value"):
            lines.append(f"  - {name} (needs value, {spec.get('value_hint','')}): "
                         f"{spec.get('description','')}")
        else:
            lines.append(f"  - {name}: {spec.get('description','')}")
    action_doc = "\n".join(lines) if lines else "  (no actions currently available)"

    return types.FunctionDeclaration(
        name="laptop_control",
        description=(
            "Control the user's laptop via laptop_agent.py, found automatically "
            "on the network — no manual setup needed. Available actions:\n"
            + action_doc + "\nOnly pass 'value' for actions that need it. "
            "ONLY call this when the user EXPLICITLY asks you to change "
            "volume/brightness or mute/unmute (e.g. 'turn up the volume', "
            "'make it brighter'). Do NOT call this as a dramatic flourish, "
            "joke, or emotional reaction (e.g. to express anger, "
            "excitement, or affection) — a touch gesture, emotion, or "
            "sarcastic remark is never itself a request to control the "
            "laptop."
        ),
        parameters=S(type=T.OBJECT, properties={
            "action": S(type=T.STRING, enum=action_names or ["volume_up"]),
            "value": S(type=T.INTEGER,
                       description="Required only for *_set actions (0-100)."),
        }, required=["action"]))
