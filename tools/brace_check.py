import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()

PAIRS = {"{": "}", "(": ")", "[": "]"}
CLOSERS = {v: k for k, v in PAIRS.items()}

stack = []          # (char, line)
line = 1
instr = None        # quote char when inside a string
esc = False
incom = None        # 'line' | 'block'

# Track whether a '/' starts a regex literal or is division. A regex can only
# start where a value is expected, i.e. right after an operator/open bracket.
prev_sig = ""       # last significant (non-space) char

i = 0
n = len(src)
while i < n:
    ch = src[i]
    nxt = src[i + 1] if i + 1 < n else ""

    if ch == "\n":
        line += 1
        if incom == "line":
            incom = None
        i += 1
        continue

    if incom == "line":
        i += 1
        continue
    if incom == "block":
        if ch == "*" and nxt == "/":
            incom = None
            i += 2
            continue
        i += 1
        continue

    if instr:
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == instr:
            instr = None
        i += 1
        continue

    if ch == "/" and nxt == "/":
        incom = "line"
        i += 2
        continue
    if ch == "/" and nxt == "*":
        incom = "block"
        i += 2
        continue

    if ch in "\"'`":
        instr = ch
        i += 1
        continue

    # regex literal
    if ch == "/" and prev_sig in ("", "(", ",", "=", ":", "[", "!", "&", "|",
                                  "?", "{", "}", ";", "+", "-", "*", "%",
                                  "<", ">", "~", "^"):
        j = i + 1
        cls = False
        while j < n:
            c = src[j]
            if c == "\\":
                j += 2
                continue
            if c == "[":
                cls = True
            elif c == "]":
                cls = False
            elif c == "/" and not cls:
                break
            elif c == "\n":
                break
            j += 1
        i = j + 1
        prev_sig = "/"
        continue

    if ch in PAIRS:
        stack.append((ch, line))
    elif ch in CLOSERS:
        if not stack:
            print(f"EXTRA '{ch}' at line {line}")
            sys.exit(1)
        op, ln = stack.pop()
        if op != CLOSERS[ch]:
            print(f"MISMATCH: '{op}' opened line {ln}, closed by '{ch}' line {line}")
            sys.exit(1)

    if not ch.isspace():
        prev_sig = ch
    i += 1

if instr:
    print(f"UNTERMINATED string ({instr}) at EOF")
elif stack:
    print("UNCLOSED:")
    for op, ln in stack:
        print(f"   '{op}' opened at line {ln}")
else:
    print("balanced")
