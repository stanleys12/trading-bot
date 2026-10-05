#!/usr/bin/env python3
"""THE ACTIONS LOG: plain-English record of what the firm actually DOES.

Companion to firm.log (the everything-firehose). This one answers only:
what are we buying, selling, holding — and what materially changed?
Sources are deterministic (broker state + book logs + registry), never
agent opinions. Writes fund/reports/actions.log.
"""
import json, re, time
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qr.execution.live_paper import (_headers, PAPER_BASE, positions,
                                     latest_targets)               # noqa: E402
import requests                                                    # noqa: E402

OUT = ROOT / "fund/reports/actions.log"
STATE = ROOT / "fund/reports/actions_state.json"
LOGS = {
    "STOCKS":  ROOT / "qr/reports/live/cron.log",
    "CRYPTO":  ROOT / "qr/reports/live/crypto_cron.log",
    "OPTIONS": ROOT / "qr/reports/live/options_cron.log",
    "RESEARCH": ROOT / "qr/reports/experiment_registry.jsonl",
}

# book-log lines worth translating into plain English
PATTERNS = [
    ("STOCKS", r"holding -- quarterly schedule",
     "no trades scheduled today (quarterly plan)"),
    ("STOCKS", r'drift repair: .*"n_orders": (\d+)',
     "placed {0} catch-up orders to finish building the portfolio (fill at the 1pm close)"),
    ("STOCKS", r"rebalanced: .*n_orders.: (\d+)",
     "quarterly reshuffle: placed {0} orders"),
    ("CRYPTO", r"done: 0 order\(s\), 2 in-band hold\(s\)",
     "holding Bitcoin + Ethereum — trend still up, no change"),
    ("CRYPTO", r"done: (\d+) order\(s\)",
     "adjusted crypto positions: {0} order(s)"),
    ("OPTIONS", r"actions: \['hold'\]",
     "holding 100 IWM + the short call — no change"),
    ("OPTIONS", r"actions: \[(?!'hold').+\]",
     "options position changed (roll/expiry action)"),
]


def emit(tag, msg):
    line = f"{datetime.now().strftime('%m-%d %H:%M')}  {tag:8} {msg}"
    print(line, flush=True)
    with OUT.open("a") as fh:
        fh.write(line + "\n")


def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def snapshot(st, first=False):
    h = _headers()
    o = requests.get(f"{PAPER_BASE}/orders", headers=h,
                     params={"status": "all", "limit": 500,
                             "after": datetime.now().strftime("%Y-%m-%dT07:00:00Z")},
                     timeout=20).json()
    known = st.get("orders", {})
    placed, filled_b, filled_s, dead = 0, [], [], 0
    for x in o:
        oid, s = x["id"], x["status"]
        if oid not in known:
            placed += 1
        elif known[oid] != s:
            if s == "filled":
                notional = float(x["filled_qty"] or 0) * float(x["filled_avg_price"] or 0)
                (filled_b if x["side"] == "buy" else filled_s).append(notional)
            elif s in ("expired", "canceled", "rejected"):
                dead += 1
        known[oid] = s
    st["orders"] = known
    if placed and not first:
        emit("ORDERS", f"placed {placed} new order(s)")
    if filled_b or filled_s:
        msg = []
        if filled_b:
            msg.append(f"BOUGHT {len(filled_b)} position(s) for ${sum(filled_b):,.0f}")
        if filled_s:
            msg.append(f"SOLD {len(filled_s)} position(s) for ${sum(filled_s):,.0f}")
        emit("FILLS", " ; ".join(msg))
    if dead and not first:
        emit("ORDERS", f"{dead} order(s) expired/cancelled unfilled")

    pos = positions()
    tgt = latest_targets()
    want = {r.symbol for r in tgt.itertuples() if r.target_shares > 0}
    n_now = len(set(pos.symbol) & want) if len(pos) else 0
    key = f"{n_now}/{len(want)}"
    if st.get("completion") != key:
        emit("BOOK", f"stock portfolio completion: {key} target names held"
             + ("" if first else f" (was {st.get('completion')})"))
        st["completion"] = key

    a = requests.get(f"{PAPER_BASE}/account", headers=h, timeout=20).json()
    eq, cash = float(a["equity"]), float(a["cash"])
    if abs(eq - st.get("equity", eq)) / max(eq, 1) > 0.004 or first:
        emit("VALUE", f"account ${eq:,.0f}, cash ${cash:,.0f}"
             + ("" if first else f" (was ${st.get('equity', 0):,.0f})"))
        st["equity"] = eq
    if (cash >= 0) != (st.get("cash", cash) >= 0):
        emit("VALUE", "margin balance cleared — no longer borrowing" if cash >= 0
             else "account is now borrowing on margin")
    st["cash"] = cash


def tail_logs(st):
    offs = st.setdefault("offs", {})
    for tag, f in LOGS.items():
        if not f.exists():
            continue
        size = f.stat().st_size
        if size < offs.get(tag, 0):
            offs[tag] = 0
        if size <= offs.get(tag, size == 0 and -1 or 0):
            offs.setdefault(tag, size)
        if size > offs.get(tag, 0):
            with f.open() as fh:
                fh.seek(offs[tag])
                chunk = fh.read()
            offs[tag] = size
            for raw in chunk.splitlines():
                if tag == "RESEARCH":
                    try:
                        j = json.loads(raw)
                        if j.get("outcome") in ("PASS", "FAIL"):
                            emit("RESEARCH", f"experiment {j['name']}: "
                                 f"{j['outcome']}"
                                 + (f" — {j.get('failure_reason','')[:90]}"
                                    if j.get("outcome") == "FAIL" else " ✔"))
                    except Exception:
                        pass
                    continue
                for ptag, rx, tmpl in PATTERNS:
                    if ptag != tag:
                        continue
                    m = re.search(rx, raw)
                    if m:
                        emit(tag, tmpl.format(*m.groups()))
                        break


def main():
    st = load_state()
    first = not st
    for tag, f in LOGS.items():                 # start log-tails at end of file
        if f.exists():
            st.setdefault("offs", {}).setdefault(tag, f.stat().st_size)
    emit("START", "actions log running — trades, holdings and material changes only")
    snapshot(st, first=True)
    STATE.write_text(json.dumps(st))
    while True:
        time.sleep(45)
        try:
            tail_logs(st)
            snapshot(st)
            STATE.write_text(json.dumps(st))
        except Exception as e:
            emit("ERROR", str(e)[:100])


if __name__ == "__main__":
    main()
