#!/usr/bin/env python3
"""One agent's tile for the 20-pane wall: header + latest memo excerpt."""
import re, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orchestrator import SEATS                          # noqa: E402

seat = sys.argv[1]
cfg = SEATS[seat]
import time as _t
lives = sorted(HERE.glob(f"floor/*/*/{seat}.live"), key=lambda f: f.stat().st_mtime)
if lives:
    lv = lives[-1]
    age = _t.time() - lv.stat().st_mtime
    txt = lv.read_text()
    if age < 20 and txt.strip():          # actively being written RIGHT NOW
        print(f"== {seat}  \u270d WRITING LIVE \u270d ==")
        for line in txt.splitlines()[-9:]:
            print(line[:120])
        raise SystemExit
memos = sorted(HERE.glob(f"floor/*/*/{seat}.md"), key=lambda f: f.stat().st_mtime)
if not memos:
    print(f"== {seat} ({cfg['model']}, t{cfg['tier']}) ==\n(no memo yet)")
    sys.exit()
m = memos[-1]
age = (time.time() - m.stat().st_mtime) / 3600
body = m.read_text()
body = re.sub(r"```(?:json)?[\s\S]*?```", "", body).strip()   # hide json tail
flag = ""
low = body.lower()
if "all clear" in low or "all_clear" in low:
    flag = " ✓"
print(f"== {seat}{flag}  ({cfg['model']}, t{cfg['tier']}, {age:.1f}h ago) ==")
for line in body.splitlines():
    if line.strip():
        print(line[:120])
