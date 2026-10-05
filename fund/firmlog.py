#!/usr/bin/env python3
"""THE FIRM LOG: one merged, timestamped stream of everything happening.

Sources (tailed live):        Rendered as:
  fund/reports/floor_feed.log   [AGENTS]  shifts, debates, evals, blitzes
  qr/reports/live/cron.log      [EQUITY]  stock book runs
  .../crypto_cron.log           [CRYPTO]  crypto book runs
  .../options_cron.log          [OPTIONS] covered-call book runs
  fund/reports/fund_cron.log    [ORCH]    orchestrator cycles
  fund/journal/journal.jsonl    [JOURNAL] every filed decision (compact)
  fund/evals/scores.jsonl       [EVALS]   each eval result
  qr/reports/experiment_registry.jsonl [SPRINT] registrations & verdicts
Plus an active broker watcher (every 60s):
  [ORDERS]  order-status changes (fills, expiries) as they happen
  [BOOK]    MISSING-names count whenever it changes

Everything is echoed to stdout AND appended to fund/reports/firm.log.
"""
import json, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "fund/reports/firm.log"

FILES = {
    "AGENTS":  ROOT / "fund/reports/floor_feed.log",
    "EQUITY":  ROOT / "qr/reports/live/cron.log",
    "CRYPTO":  ROOT / "qr/reports/live/crypto_cron.log",
    "OPTIONS": ROOT / "qr/reports/live/options_cron.log",
    "ORCH":    ROOT / "fund/reports/fund_cron.log",
    "JOURNAL": ROOT / "fund/journal/journal.jsonl",
    "EVALS":   ROOT / "fund/evals/scores.jsonl",
    "SPRINT":  ROOT / "qr/reports/experiment_registry.jsonl",
}


def emit(src, text):
    line = f"{datetime.now().strftime('%H:%M:%S')} [{src:7}] {text}"
    print(line, flush=True)
    with OUT.open("a") as fh:
        fh.write(line + "\n")


def render(src, raw):
    raw = raw.rstrip()
    if not raw:
        return None
    if src == "JOURNAL":
        try:
            j = json.loads(raw)
            return (f"{j['label']}: {','.join(j['crew'])} -> "
                    f"{len(j.get('alerts', []))} alerts, "
                    f"{len(j.get('proposals', []))} proposals")
        except Exception:
            return None
    if src == "EVALS":
        try:
            j = json.loads(raw)
            return f"{j['seat']} {j['scenario']} -> {'PASS' if j['correct'] else 'MISS'}"
        except Exception:
            return None
    if src == "SPRINT":
        try:
            j = json.loads(raw)
            return f"{j['name']} [{j.get('family','')}] -> {j.get('outcome','?')}"
        except Exception:
            return None
    return raw


def broker_state():
    from qr.execution.live_paper import _headers, PAPER_BASE, positions, latest_targets
    import requests, collections
    h = _headers()
    o = requests.get(f"{PAPER_BASE}/orders", headers=h,
                     params={"status": "all", "limit": 500,
                             "after": datetime.now().strftime("%Y-%m-%dT07:00:00Z")},
                     timeout=20).json()
    st = dict(collections.Counter(x["status"] for x in o))
    pos = positions()
    t = latest_targets()
    want = {r.symbol for r in t.itertuples() if r.target_shares > 0}
    missing = len(want - set(pos.symbol)) if len(pos) else len(want)
    return st, missing


def main():
    offs = {}
    for src, f in FILES.items():
        offs[src] = f.stat().st_size if f.exists() else 0
    emit("FIRMLOG", "started — merging 8 sources + broker watcher (60s)")
    last_orders, last_missing, last_broker = {}, None, 0.0
    while True:
        for src, f in FILES.items():
            if not f.exists():
                continue
            size = f.stat().st_size
            if size < offs.get(src, 0):
                offs[src] = 0                       # rotated/truncated
            if size > offs.get(src, 0):
                with f.open() as fh:
                    fh.seek(offs[src])
                    for raw in fh.read().splitlines():
                        r = render(src, raw)
                        if r:
                            emit(src, r)
                offs[src] = size
        if time.time() - last_broker >= 60:
            last_broker = time.time()
            try:
                st, missing = broker_state()
                if st != last_orders:
                    diff = {k: f"{v}({v - last_orders.get(k, 0):+d})"
                            for k, v in st.items() if v != last_orders.get(k)}
                    emit("ORDERS", f"status change: {diff}")
                    last_orders = st
                if missing != last_missing:
                    emit("BOOK", f"MISSING names: {last_missing} -> {missing}")
                    last_missing = missing
            except Exception as e:
                emit("ORDERS", f"(watcher error: {str(e)[:80]})")
        time.sleep(1)


if __name__ == "__main__":
    main()
