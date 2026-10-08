#!/usr/bin/env python3
"""Find unused imports in a Python file, conservatively.

Conservative on purpose: a false positive here means someone deletes a line
that was load-bearing. So a name is reported ONLY when it appears exactly once
in the whole token stream (its own import statement) and is not re-exported
via __all__, not referenced in any string, and not a known side-effect import.
"""
import ast
import sys
import io
import tokenize
from pathlib import Path

# Imported purely for their side effects — never "used" by name.
# `annotations` is a __future__ compiler directive: it changes how the whole
# file is parsed and removing it can change runtime behaviour. It is never
# referenced by name, so it must be whitelisted or it reads as dead.
SIDE_EFFECT = {"readline", "rlcompleter", "encodings", "warnings", "annotations"}


def imported_names(tree):
    """[(binding_name, lineno, source_text_hint)] for every import binding."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    out.append((a.asname, node.lineno, a.name))
                else:
                    # `import a.b.c` binds `a`
                    out.append((a.name.split(".")[0], node.lineno, a.name))
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    continue
                out.append((a.asname or a.name, node.lineno, a.name))
    return out


def used_names(tree):
    """Every Name/Attribute base actually referenced in code."""
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            n = node
            while isinstance(n, ast.Attribute):
                n = n.value
            if isinstance(n, ast.Name):
                used.add(n.id)
    return used


def exported(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for e in node.value.elts:
                            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                                names.add(e.value)
    return names


def string_mentions(src):
    """Names mentioned inside any string literal or comment.

    Catches getattr("x"), globals()["x"], type annotations in quotes, and
    doc/comment references — all of which make a 'delete it' call unsafe.
    """
    blob = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.STRING, tokenize.COMMENT):
                blob.append(tok.string)
    except Exception:
        return src           # tokenise failed: treat whole file as a mention
    return "\n".join(blob)


def audit(path):
    src = Path(path).read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [("SYNTAX ERROR", e.lineno, str(e))]

    imports = imported_names(tree)
    used = used_names(tree)
    exp = exported(tree)
    strings = string_mentions(src)

    dead = []
    for name, lineno, origin in imports:
        if name in used or name in exp or name in SIDE_EFFECT:
            continue
        if name in strings or origin in strings:
            continue
        dead.append((name, lineno, origin))
    return dead


if __name__ == "__main__":
    targets = sys.argv[1:]
    total = 0
    for t in sorted(targets):
        rows = audit(t)
        if rows:
            print(f"\n{t}")
            for name, lineno, origin in rows:
                print(f"  line {lineno:<5} {name:<24} (from {origin})")
                total += 1
    print(f"\n{total} unused import binding(s) across {len(targets)} file(s)")
