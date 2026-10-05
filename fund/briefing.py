#!/usr/bin/env python3
"""Build the desk briefing: one markdown pack of ground truth for all seats.

Deterministic code gathers data; agents only judge it. Agents never get
tools or credentials — a seat cannot invent a number that isn't in here,
which is what makes eval-scoring for hallucination possible at all.
"""
import json, subprocess, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qr.execution.live_paper import (_db, _headers, PAPER_BASE, DATA_BASE,
                                     positions, latest_targets)  # noqa: E402

LIVE = ROOT / "qr" / "reports" / "live"


def _tail(path, n=12):
    try:
        return "\n".join(Path(path).read_text().splitlines()[-n:])
    except Exception as e:
        return f"(unreadable: {e})"


def build() -> str:
    h = _headers()
    now = datetime.now().astimezone()
    clock = requests.get(f"{PAPER_BASE}/clock", headers=h, timeout=20).json()
    acct = requests.get(f"{PAPER_BASE}/account", headers=h, timeout=20).json()

    pos = positions()
    sleeve = pos[~pos.symbol.isin({"IWM"})] if len(pos) else pos
    crypto = positions(asset_class="crypto")
    opt = positions(asset_class="us_option")

    tgt = latest_targets()
    want = {r.symbol: r.target_shares for r in tgt.itertuples()
            if r.target_shares > 0} if len(tgt) else {}
    held = set(sleeve.symbol) if len(sleeve) else set()
    missing = sorted(s for s in want if s not in held)
    # MISSING history: requested by seat execution_verifier (journal
    # 2026-08-26) so a stalled drift-repair is distinguishable from a
    # shrinking gap. Append-once-per-build, keep the last 10 in the pack.
    hist_f = Path(__file__).resolve().parent / "reports" / "missing_history.jsonl"
    hist_f.parent.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with hist_f.open("a") as fh:
        fh.write(json.dumps({"ts": stamp, "missing": len(missing),
                             "held": len(held), "targets": len(want)}) + "\n")
    hist = [json.loads(l) for l in hist_f.read_text().splitlines()[-10:]]
    hist_line = " -> ".join(f"{r['missing']}@{r['ts'][5:]}" for r in hist)

    since = (datetime.now(timezone.utc) - timedelta(hours=30)
             ).strftime("%Y-%m-%dT%H:%M:%SZ")
    orders = requests.get(f"{PAPER_BASE}/orders", headers=h,
                          params={"status": "all", "limit": 500,
                                  "after": since}, timeout=30).json()
    ostat = {}
    for o in orders:
        ostat[o["status"]] = ostat.get(o["status"], 0) + 1
    partial = sum(1 for o in orders
                  if 0 < float(o["filled_qty"] or 0) < float(o["qty"] or 1))

    con = _db()
    try:
        d2 = pd.read_sql("SELECT session_date,n_positions,sleeve_mv,"
                         "book_day_pct,spy_day_pct,excess_vs_spy_pct,"
                         "coverage_pct FROM daily_v2 ORDER BY session_date "
                         "DESC LIMIT 6", con).to_string(index=False)
    except Exception as e:
        d2 = f"(daily_v2 unavailable: {e})"

    heads = []
    try:
        syms = list(sleeve.nlargest(10, "market_value").symbol) if len(sleeve) else []
        r = requests.get("https://data.alpaca.markets/v1beta1/news",
                         headers=h, params={"symbols": ",".join(syms + ["SPY", "BTCUSD"]),
                                            "limit": 12}, timeout=20)
        if r.ok:
            heads = [f"- [{a.get('symbols')}] {a.get('headline','')[:120]}"
                     for a in r.json().get("news", [])]
    except Exception:
        pass

    big = sleeve.nlargest(8, "market_value") if len(sleeve) else sleeve
    eq = float(acct.get("equity", 0)); gross = float(acct.get("long_market_value", 0) or 0)
    # --- sprint section: latest experiment artifacts + proposals ---
    sprint = ROOT / "qr" / "sprint"
    sp_lines = []
    try:
        arts = sorted((sprint / "artifacts").glob("*.json"),
                      key=lambda f: f.stat().st_mtime)[-3:]
        for f in arts:
            a = json.loads(f.read_text())
            v = a.get("results", {}).get("validation", {})
            sp_lines.append(
                f"- {a['name']}: verdict={a.get('verdict')} "
                f"n={v.get('n')} mean_net={v.get('mean_net_bps')}bps "
                f"t={v.get('t_stat')} powered={a.get('powered')} "
                f"red_team={a.get('red_team')}")
        bl = sprint / "BACKLOG.md"
        if bl.exists():
            tops = [l.split("—")[0].replace("PRIORITY", "").strip(" 0123456789-")
                    for l in bl.read_text().splitlines()
                    if l.startswith("PRIORITY")][:3]
            sp_lines.append("- BACKLOG (top 3): " + " | ".join(tops))
        pend = sorted((sprint / "proposals").glob("*.json"))
        sp_lines.append(f"- proposals pending validation: {[f.stem for f in pend]}")
    except Exception as e:
        sp_lines.append(f"(sprint state unreadable: {e})")

    parts = [
        f"# Desk briefing — {now:%Y-%m-%d %H:%M %Z}",
        f"market_open={clock.get('is_open')}  next_open={clock.get('next_open')}  next_close={clock.get('next_close')}",
        "",
        "## Account (shared by all three books)",
        f"equity ${eq:,.0f} | cash ${float(acct.get('cash',0)):,.0f} | gross ${gross:,.0f} ({gross/max(eq,1):.2f}x)",
        "",
        "## Books",
        f"- LOW-VOL EQUITY: {len(sleeve)} names, ${float(sleeve.market_value.sum()) if len(sleeve) else 0:,.0f}."
        f"  Target book: {len(want)} names; MISSING (held=0 vs target): {len(missing)}"
        + (f" e.g. {missing[:8]}" if missing else "")
        + f"\n  MISSING trend (oldest->newest): {hist_line}",
        f"- CRYPTO TSMOM: {len(crypto)} positions, ${float(crypto.market_value.sum()) if len(crypto) else 0:,.0f}"
        + (": " + "; ".join(f"{r.symbol} mv=${r.market_value:,.0f}"
                            for r in crypto.itertuples()) if len(crypto) else "")
        + "  (qty display removed 08-26 after the int-truncation defect was FIXED; market value is authoritative)",
        f"- COVERED CALL: IWM ${float(pos[pos.symbol=='IWM'].market_value.sum()) if len(pos) else 0:,.0f}"
        f" + options ${float(opt.market_value.sum()) if len(opt) else 0:,.0f}",
        "",
        "## Largest equity positions (spec: ~$352/name)",
        big[["symbol", "qty", "market_value"]].to_string(index=False) if len(big) else "(none)",
        "",
        f"## Orders last 30h (n={len(orders)})",
        f"status counts: {ostat} | partial fills: {partial}",
        "NOTE: tif=cls (market-on-close) orders correctly sit at status "
        "'new' ALL DAY and fill in the 13:00 PT closing auction. A large "
        "'new' count before the close is expected mechanics, not a failure. "
        "Judge fills only AFTER the close.",
        "",
        "## Measurement record (daily_v2, latest first)",
        d2,
        "",
        "## Scheduler log tails",
        "### equity (cron.log)", _tail(LIVE / "cron.log"),
        "### crypto", _tail(LIVE / "crypto_cron.log", 8),
        "### options", _tail(LIVE / "options_cron.log", 8),
        "",
        "## Sprint (30-day intraday program; charter qr/sprint/CHARTER.md)",
        "\n".join(sp_lines),
        "",
        "## Headlines (Alpaca news; may be empty)",
        "\n".join(heads) if heads else "(none fetched)",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    print(build())
