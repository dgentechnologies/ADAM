"""Remove the manual IP + access-token form from the desktop Devices tab.

Connecting is now: account -> cloud device list -> match on the LAN -> connect,
with a plain network search as the fallback. Nothing is typed, so the host,
token and port fields have no remaining purpose and their presence invited
exactly the manual setup the discovery work removed.
"""
from pathlib import Path

p = Path(r"D:\Dgen Technologies Pvt. Ltd\ADAM\adam-desktop\resources\static\index.html")
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)

start = next((i for i, l in enumerate(lines) if 'id="connectionForm"' in l), None)
if start is None:
    raise SystemExit("connectionForm not found - aborting with no changes")
end = next((i for i, l in enumerate(lines[start:], start) if l.strip() == "</form>"), None)
if end is None:
    raise SystemExit("closing </form> not found - aborting with no changes")

# Sanity-check the block really is the manual form before deleting 30 lines.
block = "".join(lines[start:end + 1])
for needle in ("connectionHost", "connectionToken", "connectionSyncPort"):
    if needle not in block:
        raise SystemExit(f"block does not look like the manual form ({needle} absent) - aborting")

REPLACEMENT = """<p class="help">ADAM announces itself on your network. Use <strong>Find ADAM on this network</strong> above, or sign in and pick the ADAM linked to your account \u2014 the connection key is exchanged for you.</p>
"""

lines[start:end + 1] = [REPLACEMENT]
p.write_text("".join(lines), encoding="utf-8")
print(f"removed the manual connection form ({end - start + 1} lines)")
