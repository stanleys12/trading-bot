#!/usr/bin/env python3
"""Compact live roster: all 20 seats in two columns for the dashboard."""
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orchestrator import SEATS                          # noqa: E402

def status():
    scores = {}
    sf = HERE / "evals" / "scores.jsonl"
    if sf.exists():
        for line in sf.read_text().splitlines():
            r = json.loads(line)
            scores.setdefault(r["seat"], []).append(r["correct"])
    memos = {}
    for f in HERE.glob("floor/*/*/*.md"):
        if f.stem in SEATS:
            memos[f.stem] = max(memos.get(f.stem, 0), f.stat().st_mtime)
    rows = []
    for s, c in SEATS.items():
        age = time.time() - memos.get(s, 0)
        dot = "●" if age < 7200 else ("○" if age < 86400 else "·")
        sc = scores.get(s)
        ev = f"{sum(sc)}/{len(sc)}" if sc else "  -"
        rows.append(f"{dot} {s[:18]:18} t{c['tier']} {ev:>5}")
    half = (len(rows) + 1) // 2
    w = {"desk": 12, "research": 5, "governance": 3}
    print(f"AGENTS: {len(rows)} seats  (desk {w['desk']} / research {w['research']} / gov {w['governance']})   "
          "●<2h ○today ·idle   t=tier  evals")
    for a, b in zip(rows[:half], rows[half:] + [""]):
        print(f"{a}   {b}")

if __name__ == "__main__":
    status()
