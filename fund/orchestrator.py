#!/usr/bin/env python3
"""Fund orchestrator: decides which crew sits this cycle, briefs them,
runs each seat headless, synthesizes, journals, and alerts.

Design constraints (deliberate):
- Agents are TOOL-LESS: briefing in, memo out. No credentials, no
  filesystem, no orders. All order flow stays with the validated runners.
- Cadence: market hours = every cycle (light crew hourly, full desk at the
  hours in roster.schedule); off-hours = skeleton crew every 4h; research
  wing daily after the close; evals nightly; coach weekly.
- Every memo must end in a machine-readable json block; unparseable memos
  are logged as seat failures (they count against the seat in evals).
"""
import argparse, glob, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

ROSTER = json.loads((HERE / "roster.json").read_text())
SEATS, SCHED = ROSTER["seats"], ROSTER["schedule"]
STATE_F = HERE / "state.json"
TIMEOUT = 300
SPEC_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def claude_bin() -> str:
    hits = sorted(glob.glob(str(Path.home() /
        ".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude")))
    if not hits:
        raise RuntimeError("claude binary not found under ~/.vscode/extensions")
    return hits[-1]


def feed(msg):
    """Append one line to the live floor feed (the wall's ticker tile)."""
    line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
    with (HERE / "reports" / "floor_feed.log").open("a") as fh:
        fh.write(line + "\n")


def notify(title, msg):
    try:
        subprocess.run(["osascript", "-e",
                        f'display notification "{msg[:120]}" with title "{title}"'],
                       timeout=10, capture_output=True)
    except Exception:
        pass


def load_state():
    try:
        return json.loads(STATE_F.read_text())
    except Exception:
        return {}


def playbook_for(seat: str) -> str:
    cfg = SEATS[seat]
    text = (HERE / "playbooks" / cfg["playbook"]).read_text()
    if cfg.get("param"):
        text = text.replace("{PARAM}", cfg["param"])
    return (HERE / "playbooks" / "_common.md").read_text() + "\n" + text


def _extract_tail(memo: str):
    """Pull the seat's json block: prefer the last ```json fence, else
    brace-balance from the last '{"seat"' occurrence."""
    fences = re.findall(r"```(?:json)?\s*([\s\S]*?)```", memo)
    cands = [f.strip() for f in fences if '"seat"' in f]
    start = memo.rfind('{"seat"')
    if start != -1:
        depth = 0
        for i in range(start, len(memo)):
            if memo[i] == "{":
                depth += 1
            elif memo[i] == "}":
                depth -= 1
                if depth == 0:
                    cands.append(memo[start:i + 1])
                    break
    for c in reversed(cands):
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def queue_proposal(memo: str):
    """Drop the first spec fence in a research_lead memo into the sprint
    proposals queue. The name becomes a filename, so it must be a plain
    token: agent output never picks a path outside proposals/."""
    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", memo):
        try:
            cand = json.loads(block)
        except Exception:
            continue
        if isinstance(cand, dict) and "signal" in cand and "name" in cand:
            name = cand["name"]
            if not (isinstance(name, str) and SPEC_NAME.fullmatch(name)):
                feed(f"research_lead spec name {name!r} refused (unsafe filename)")
                return None
            (ROOT / "qr/sprint/proposals" / f"{name}.json"
             ).write_text(json.dumps(cand, indent=1))
            return name
    return None


def run_seat(seat: str, briefing: str, extra: str = "", model=None,
             live_file=None) -> dict:
    cfg = SEATS[seat]
    prompt = (f"{playbook_for(seat)}\n\n--- BRIEFING ---\n{briefing}\n"
              + (f"\n--- PEER MEMOS ---\n{extra}\n" if extra else "")
              + f"\nWrite your memo now as seat '{seat}'.")
    t0 = time.time()
    try:
        if live_file:
            # STREAMING: write each token to live_file as it is generated,
            # so the dashboard tile shows the agent literally typing.
            Path(live_file).write_text("")
            proc = subprocess.Popen(
                [claude_bin(), "-p", prompt, "--model", model or cfg["model"],
                 "--output-format", "stream-json",
                 "--include-partial-messages", "--verbose"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1)
            parts, result_text = [], None
            with open(live_file, "a") as lf:
                for line in proc.stdout:
                    if time.time() - t0 > TIMEOUT:
                        proc.kill()
                        break
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    if j.get("type") == "stream_event":
                        d = (j.get("event") or {}).get("delta") or {}
                        if d.get("type") == "text_delta":
                            parts.append(d["text"])
                            lf.write(d["text"]); lf.flush()
                    elif j.get("type") == "result":
                        result_text = j.get("result")
            proc.wait(timeout=10)
            memo = (result_text or "".join(parts)).strip()
            ok = len(memo) > 40
        else:
            r = subprocess.run([claude_bin(), "-p", prompt, "--model",
                                model or cfg["model"]],
                               capture_output=True, text=True, timeout=TIMEOUT)
            memo = r.stdout.strip()
            ok = r.returncode == 0 and len(memo) > 40
    except subprocess.TimeoutExpired:
        memo, ok = "(timeout)", False
    tail = _extract_tail(memo)
    return {"seat": seat, "ok": ok and tail is not None, "memo": memo,
            "tail": tail or {"seat": seat, "alerts": [], "proposals": [],
                             "all_clear": None, "parse_failed": True},
            "secs": round(time.time() - t0, 1)}


def decide_crew(now, market_open, state, force=None):
    """Return (label, crew list) for this cycle, or (None, []) to skip."""
    if force:
        return "forced", force
    hr, dow = now.hour, now.strftime("%A")
    crews, label = [], []
    if market_open:
        if hr in SCHED["market_full_hours_pt"]:
            return "market_full", SCHED["market_full_crew"]
        return "market_light", SCHED["market_light_crew"]
    # research / governance windows (run once per day each)
    today = now.strftime("%Y-%m-%d")
    if hr >= SCHED["research_hour_pt"] and state.get("research_day") != today \
            and now.weekday() < 5:
        rot = SCHED["research_crew_rotation"]
        idx = state.get("research_rot", 0) % len(rot)
        state["research_day"], state["research_rot"] = today, idx + 1
        crew = rot[idx] + (["coach"] if dow == SCHED["coach_day"] else [])
        return "research", crew
    if hr >= SCHED["scribe_hour_pt"] and state.get("scribe_day") != today \
            and now.weekday() < 5:
        state["scribe_day"] = today
        return "scribe", ["scribe"]
    # Saturday blitz: the FULL roster against the full fault suite
    # (principal-approved 2026-08-27). Detached so the heartbeat never blocks.
    week = now.strftime("%G-W%V")
    if now.weekday() == 5 and hr >= 10 and state.get("blitz_week") != week:
        state["blitz_week"] = week
        import subprocess as sp
        sp.Popen(["/bin/zsh", str(HERE / "evals" / "blitz.sh")],
                 start_new_session=True,
                 stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        feed("EVAL BLITZ launched: all 20 seats, full suite (runs ~2h in background)")
        return None, []

    # nightly evals: benchmark ONE rotating seat at its live model
    if hr >= SCHED.get("evals_hour_pt", 20) and \
            state.get("evals_day") != today:
        state["evals_day"] = today
        seats = list(SEATS)
        seat = seats[state.get("evals_rot", 0) % len(seats)]
        state["evals_rot"] = state.get("evals_rot", 0) + 1
        import subprocess as sp
        feed(f"EVALS: benchmarking {seat} on the seeded-fault suite")
        r = sp.run(["/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
                    "-W", "ignore", str(HERE.parent / "fund/evals/run_evals.py"),
                    "--seats", seat], capture_output=True, text=True, timeout=3000)
        tailline = (r.stdout.strip().splitlines() or ["no output"])[-1]
        feed(f"EVALS: {seat} -> {tailline}")
        return None, []
    # off-hours skeleton every N hours
    last = state.get("last_skeleton", 0)
    if time.time() - last >= SCHED["offhours_interval_hours"] * 3600 - 120:
        state["last_skeleton"] = time.time()
        return "skeleton", SCHED["offhours_crew"]
    return None, []


PMS = ["equity_pm", "crypto_pm", "options_pm"]


def due_stagger(now, state):
    """Seats due in this 5-minute slot (fires each slot at most once)."""
    slot = now.minute - now.minute % 5
    key = f"{now:%Y-%m-%d %H}:{slot:02d}"
    if state.get("last_slot") == key:
        return []
    state["last_slot"] = key
    seats = SCHED.get("stagger_market", {}).get(str(slot), [])
    return [PMS[now.hour % 3] if s == "PM_ROTATE" else s for s in seats]


def run_solo(seat, briefing, now):
    """One seat takes the desk alone: streamed memo, journal, debate on HIGH."""
    floor = HERE / "floor" / now.strftime("%Y-%m-%d") / f"{now:%H%M}_solo"
    floor.mkdir(parents=True, exist_ok=True)
    (floor / "briefing.md").write_text(briefing)
    extra = ""
    if seat == "chief_of_desk":       # synthesize the trailing hour's memos
        import time as _t
        peers = []
        for f in sorted((HERE / "floor" / now.strftime("%Y-%m-%d")).glob("*/*.md")):
            if f.stem in ("briefing", "chief_of_desk"):
                continue
            if _t.time() - f.stat().st_mtime < 3900:
                peers.append(f"### memo from {f.stem}\n{f.read_text()}")
        extra = "\n\n".join(peers[-14:])
    feed(f"→ {seat} takes the desk")
    r = run_seat(seat, briefing, extra=extra, live_file=floor / f"{seat}.live")
    (floor / f"{seat}.md").write_text(r["memo"])
    als = r["tail"].get("alerts") or []
    head = (f"{len(als)} alert(s): " + als[0].get("msg", "")[:80]) if als \
        else ("all clear" if r["tail"].get("all_clear") else "filed")
    feed(f"{seat} filed ({r['secs']}s) — {head}")
    if seat == "research_lead" and r["ok"]:
        name = queue_proposal(r["memo"])
        if name:
            feed(f"research_lead proposed spec '{name}' -> queued")
    if seat == "chief_of_desk" and r["ok"]:
        (HERE / "reports" / "desk_note_latest.md").write_text(
            f"({now:%Y-%m-%d %H:%M} staggered)\n\n{r['memo']}")
    results = [r]
    for al in als:
        if al.get("severity") != "high":
            continue
        reviewer = "risk_officer" if seat != "risk_officer" else "execution_verifier"
        feed(f"DEBATE: {reviewer} cross-examining {seat}'s HIGH: "
             f"\"{al.get('msg','')[:70]}\"")
        q = (f"PEER REVIEW REQUEST. Seat '{seat}' raised this HIGH alert:\n"
             f"  \"{al.get('msg','')}\"\nTheir memo:\n{r['memo'][:2500]}\n\n"
             f"CONFIRM or REBUT with evidence. Start with CONFIRMED or REBUTTED.")
        d = run_seat(reviewer, briefing, extra=q,
                     live_file=floor / f"{reviewer}.live")
        stance = "CONFIRMED" if "confirmed" in d["memo"][:200].lower() else \
            ("REBUTTED" if "rebutted" in d["memo"][:200].lower() else "unclear")
        (floor / f"{reviewer}__re__{seat}.md").write_text(d["memo"])
        feed(f"DEBATE RESULT: {reviewer} -> {stance}")
        d["seat"] = f"{reviewer} (re: {seat})"
        results.append(d)
        break                                   # one debate per solo filing
    with (HERE / "journal" / "journal.jsonl").open("a") as fh:
        fh.write(json.dumps({
            "ts": now.isoformat(timespec="seconds"), "label": "solo",
            "crew": [seat],
            "alerts": [{**a, "seat": seat} for a in als],
            "proposals": [{**pr, "seat": seat,
                           "tier": SEATS.get(seat, {}).get("tier", 0)}
                          for pr in (r["tail"].get("proposals") or [])],
            "seat_ok": {x["seat"]: x["ok"] for x in results},
            "floor": str(floor)}) + "\n")
    if any(a.get("severity") == "high" for a in als):
        notify("Fund desk: HIGH alert", als[0].get("msg", "")[:90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crew", help="comma list: run exactly these seats once")
    ap.add_argument("--label", default="manual")
    a = ap.parse_args()

    now = datetime.now().astimezone()
    from fund.briefing import build          # late import: needs repo path
    try:
        briefing = build()
    except Exception as e:
        notify("Fund desk DOWN", f"briefing failed: {e}")
        (HERE / "reports" / "last_error.txt").write_text(f"{now} {e}")
        return 1
    market_open = "market_open=True" in briefing.splitlines()[1]

    state = load_state()
    if a.crew:
        label, crew = a.label, [s.strip() for s in a.crew.split(",")]
    elif market_open:
        if SCHED.get("market_mode") == "event":
            return 0                    # dispatcher daemon owns market hours
        if SCHED.get("market_mode") == "full_waves":
            period = int(SCHED.get("wave_minutes", 20)) * 60
            if time.time() - state.get("last_wave", 0) < period - 90:
                STATE_F.write_text(json.dumps(state))
                return 0
            state["last_wave"] = time.time()
            STATE_F.write_text(json.dumps(state))
            label, crew = "wave", list(SEATS)
        else:
            seats = due_stagger(now, state)
            STATE_F.write_text(json.dumps(state))
            for s in seats:
                run_solo(s, briefing, now)
            return 0
    else:
        label, crew = decide_crew(now, market_open, state)
    STATE_F.write_text(json.dumps(state))
    if not crew:
        return 0

    if label == "research":
        # deterministic sprint steps BEFORE agents convene, so their briefing
        # shows tonight's fresh results: intake proposals, run pending specs
        try:
            import subprocess as sp
            PY = "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
            reg = sp.run([PY, "-W", "ignore", str(ROOT / "qr/sprint/registrar.py")],
                         capture_output=True, text=True, timeout=120)
            sprint_log = [reg.stdout.strip()] if reg.stdout.strip() else []
            specs = (ROOT / "qr/sprint/specs").glob("*.json")
            arts = {f.stem for f in (ROOT / "qr/sprint/artifacts").glob("*.json")}
            for s in specs:
                if s.stem not in arts and not s.stem.startswith("C"):
                    r = sp.run([PY, "-W", "ignore",
                                str(ROOT / "qr/sprint/factory.py"), s.stem],
                               capture_output=True, text=True, timeout=1800)
                    sprint_log.append(f"ran {s.stem}: rc={r.returncode}")
            if sprint_log:
                (HERE / "reports" / "sprint_nightly.log").open("a").write(
                    f"{now.isoformat()} {sprint_log}\n")
            briefing = build()          # rebuild: include fresh artifacts
        except Exception as e:
            briefing += f"\n(sprint nightly step failed: {e})"

    floor = HERE / "floor" / now.strftime("%Y-%m-%d") / f"{now:%H%M}_{label}"
    floor.mkdir(parents=True, exist_ok=True)
    (floor / "briefing.md").write_text(briefing)

    workers = [s for s in crew if s != "chief_of_desk"]
    feed(f"── CYCLE {label}: convening {len(crew)} seat(s): {', '.join(crew)}")
    results = []
    with ThreadPoolExecutor(
            max_workers=int(os.environ.get("FUND_WORKERS", "3"))) as ex:
        futs = {ex.submit(run_seat, s, briefing,
                          live_file=floor / f"{s}.live"): s for s in workers}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            # write the memo THE MOMENT the seat finishes -> tiles update live
            (floor / f"{r['seat']}.md").write_text(r["memo"])
            als = r["tail"].get("alerts") or []
            ac = r["tail"].get("all_clear")
            head = (f"{len(als)} alert(s): " + als[0].get("msg", "")[:80]
                    ) if als else ("all clear" if ac else "filed")
            feed(f"{r['seat']} filed ({r['secs']}s) — {head}")

    # ---- debate round: HIGH alerts get a peer cross-examination ----
    debates = 0
    for r in list(results):
        if debates >= 2:
            break
        for al in (r["tail"].get("alerts") or []):
            if al.get("severity") != "high" or debates >= 2:
                continue
            reviewer = "risk_officer" if r["seat"] != "risk_officer" \
                else "execution_verifier"
            if reviewer not in SEATS or reviewer == r["seat"]:
                continue
            feed(f"DEBATE: {reviewer} cross-examining {r['seat']}'s HIGH: "
                 f"\"{al.get('msg','')[:70]}\"")
            q = (f"PEER REVIEW REQUEST. Seat '{r['seat']}' raised this HIGH "
                 f"alert:\n  \"{al.get('msg','')}\"\nTheir full memo:\n"
                 f"{r['memo'][:2500]}\n\nAgainst the briefing, CONFIRM or "
                 f"REBUT this specific claim with evidence. Start your memo "
                 f"with CONFIRMED or REBUTTED.")
            d = run_seat(reviewer, briefing, extra=q,
                         live_file=floor / f"{reviewer}.live")
            stance = "CONFIRMED" if "confirmed" in d["memo"][:200].lower() \
                else ("REBUTTED" if "rebutted" in d["memo"][:200].lower()
                      else "unclear")
            (floor / f"{reviewer}__re__{r['seat']}.md").write_text(d["memo"])
            feed(f"DEBATE RESULT: {reviewer} -> {stance} on {r['seat']}'s alert")
            d["seat"] = f"{reviewer} (re: {r['seat']})"
            results.append(d)
            debates += 1

    for r in results:
        if r["seat"] == "research_lead" and r["ok"]:
            queue_proposal(r["memo"])

    alerts, proposals = [], []
    for r in results:
        for al in r["tail"].get("alerts", []) or []:
            alerts.append({**al, "seat": r["seat"]})
        for pr in r["tail"].get("proposals", []) or []:
            base = r["seat"].split(" ")[0]        # debate seats: "x (re: y)"
            proposals.append({**pr, "seat": r["seat"],
                              "tier": SEATS.get(base, {}).get("tier", 0)})

    note = None
    if "chief_of_desk" in crew:
        peers = "\n\n".join(f"### memo from {r['seat']}\n{r['memo']}"
                            for r in results)
        chief = run_seat("chief_of_desk", briefing, extra=peers,
                         live_file=floor / "chief_of_desk.live")
        results.append(chief)
        note = chief["memo"]
        (floor / "chief_of_desk.md").write_text(note)
        (HERE / "reports" / "desk_note_latest.md").write_text(
            f"({now:%Y-%m-%d %H:%M} {label})\n\n{note}")

    with (HERE / "journal" / "journal.jsonl").open("a") as fh:
        fh.write(json.dumps({
            "ts": now.isoformat(timespec="seconds"), "label": label,
            "crew": crew, "alerts": alerts, "proposals": proposals,
            "seat_ok": {r["seat"]: r["ok"] for r in results},
            "floor": str(floor)}) + "\n")
    if proposals:
        with (HERE / "journal" / "proposals.jsonl").open("a") as fh:
            for p in proposals:
                fh.write(json.dumps({"ts": now.isoformat(timespec="seconds"),
                                     **p, "status": "advisory"}) + "\n")

    if note:
        feed(f"chief_of_desk synthesized {len(results)} memo(s) -> desk note updated")
    high = [al for al in alerts if al.get("severity") == "high"]
    feed(f"── CYCLE {label} complete: {len(alerts)} alert(s), {len(high)} high, "
         f"{len(proposals)} proposal(s)")
    if high:
        notify("Fund desk: HIGH alert",
               f"{len(high)} high: " + high[0].get("msg", "")[:90])
    failed = [r["seat"] for r in results if not r["ok"]]
    print(json.dumps({"label": label, "crew": crew,
                      "alerts": len(alerts), "high": len(high),
                      "proposals": len(proposals), "failed_seats": failed}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
