#!/usr/bin/env python3
"""Talk to any seat on the desk, with its full working context.

  talk.py                     roster: every seat, model, tier, last activity
  talk.py risk_officer        interactive chat with that seat
  talk.py risk_officer "why is gross 1.37x?"     one-shot question

The seat answers in character with its playbook, the CURRENT desk briefing,
its most recent memo, and your running conversation (kept per-seat in
fund/chats/, so it remembers what you discussed last time).
"""
import json, subprocess, sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from orchestrator import SEATS, claude_bin, playbook_for   # noqa: E402

CHATS = HERE / "chats"; CHATS.mkdir(exist_ok=True)


def last_memo(seat):
    memos = sorted(HERE.glob(f"floor/*/*/{seat}.md"),
                   key=lambda f: f.stat().st_mtime)
    return memos[-1] if memos else None


def roster():
    scores = {}
    sf = HERE / "evals" / "scores.jsonl"
    if sf.exists():
        for line in sf.read_text().splitlines():
            r = json.loads(line)
            scores.setdefault(r["seat"], []).append(r["correct"])
    print(f"{'SEAT':22}{'WING':12}{'MODEL':8}{'TIER':5}{'EVALS':>10}  LAST MEMO")
    for s, c in SEATS.items():
        sc = scores.get(s)
        ev = f"{sum(sc)}/{len(sc)}" if sc else "-"
        m = last_memo(s)
        when = datetime.fromtimestamp(m.stat().st_mtime).strftime("%m-%d %H:%M") if m else "-"
        print(f"{s:22}{c['wing']:12}{c['model']:8}{c['tier']:<5}{ev:>10}  {when}")


def ask(seat, msg, transcript):
    briefing = ""
    bfile = sorted(HERE.glob("floor/*/*/briefing.md"),
                   key=lambda f: f.stat().st_mtime)
    if bfile:
        briefing = bfile[-1].read_text()
    memo = last_memo(seat)
    prompt = (playbook_for(seat)
              + "\n\nYou are in DIRECT CONVERSATION with the fund's human "
                "principal — answer their question as this seat. Be direct and "
                "concrete; cite only numbers from the briefing/memo below. No "
                "json block needed in conversation.\n"
              + f"\n--- CURRENT BRIEFING ---\n{briefing[:6000]}\n"
              + (f"\n--- YOUR LATEST MEMO ---\n{memo.read_text()[:3000]}\n" if memo else "")
              + (f"\n--- CONVERSATION SO FAR ---\n{transcript[-6000:]}\n" if transcript else "")
              + f"\nPRINCIPAL: {msg}\n{seat.upper()}:")
    r = subprocess.run([claude_bin(), "-p", prompt, "--model",
                        SEATS[seat]["model"]],
                       capture_output=True, text=True, timeout=300)
    return r.stdout.strip() or f"(seat failed: {r.stderr[:200]})"


def main():
    if len(sys.argv) < 2:
        roster(); return
    seat = sys.argv[1]
    if seat not in SEATS:
        print(f"unknown seat '{seat}'. Seats:"); roster(); return
    log = CHATS / f"{seat}.md"
    transcript = log.read_text() if log.exists() else ""
    if len(sys.argv) > 2:                                  # one-shot
        msg = " ".join(sys.argv[2:])
        out = ask(seat, msg, transcript)
        print(out)
        log.write_text(transcript + f"\nPRINCIPAL: {msg}\n{seat}: {out}\n")
        return
    print(f"— chatting with {seat} (playbook: {SEATS[seat]['playbook']}). "
          f"Ctrl-C or empty line to leave. —")
    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not msg:
            break
        out = ask(seat, msg, transcript)
        print(f"\n{seat}> {out}")
        transcript += f"\nPRINCIPAL: {msg}\n{seat}: {out}\n"
        log.write_text(transcript)


if __name__ == "__main__":
    main()
