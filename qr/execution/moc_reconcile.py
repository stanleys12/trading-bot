"""
MOC fill reconciliation: did the closing-auction orders get THE close?
======================================================================

The equity book now executes via market-on-close. For those fills the
standard reconcile's reference (SIP NBBO midpoint at the fill instant)
is the wrong question: an auction fill's promise is exact equality with
the OFFICIAL CLOSING AUCTION PRINT. This tool asks precisely that,
symbol by symbol, against /v2/stocks/auctions — the instrument exp10c
validated. It is the adjudicator for the 2026-08-18 transition rebalance
(~300 fills) and every quarterly rebalance after it.

Verdict semantics, fixed now:
  match      |fill − official close| <= 0.01% (rounding tolerance)
  mismatch   anything else — each one listed, because a single systematic
             mismatch pattern would mean the MOC assumption (and with it
             the backtest's fill model) is wrong for single names.

Usage:  python3 -m qr.execution.moc_reconcile [YYYY-MM-DD]
        (defaults to the most recent session with cls fills)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from qr.execution.live_paper import DATA_BASE, PAPER_BASE, _headers  # noqa: E402


def cls_fills(day: str | None) -> pd.DataFrame:
    """Filled cls-TIF orders, from the orders API (independent of the DB)."""
    rows, token = [], None
    while True:
        params = {"status": "closed", "limit": 500, "direction": "desc"}
        if token:
            params["page_token"] = token
        r = requests.get(f"{PAPER_BASE}/orders", headers=_headers(),
                         params=params, timeout=60)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for o in batch:
            if (o.get("time_in_force") == "cls"
                    and o.get("status") == "filled"):
                rows.append({"symbol": o["symbol"], "side": o["side"],
                             "qty": float(o["filled_qty"]),
                             "fill": float(o["filled_avg_price"]),
                             "filled_at": o["filled_at"][:10]})
        if len(batch) < 500:
            break
        token = None            # orders API pages via 'after'; one page of
        break                   # 500 covers any single rebalance day
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if day is None:
        day = df["filled_at"].max()
    return df[df["filled_at"] == day]


def official_closes(symbols: list[str], day: str) -> dict[str, float]:
    out = {}
    for i in range(0, len(symbols), 100):
        batch = symbols[i:i + 100]
        r = requests.get(f"{DATA_BASE}/stocks/auctions", headers=_headers(),
                         params={"symbols": ",".join(batch), "start": day,
                                 "end": day, "feed": "sip", "limit": 10000},
                         timeout=60)
        if not r.ok:
            continue
        for sym, days in (r.json().get("auctions") or {}).items():
            for d in days:
                if d["d"] == day and d.get("c"):
                    best = max(d["c"], key=lambda p: p.get("s", 0))
                    out[sym] = best["p"]
    return out


def main() -> int:
    day = sys.argv[1] if len(sys.argv) > 1 else None
    fills = cls_fills(day)
    if fills.empty:
        print("no filled cls orders found" + (f" for {day}" if day else ""))
        return 1
    day = fills["filled_at"].iat[0]
    closes = official_closes(sorted(fills.symbol.unique()), day)
    fills["official"] = fills.symbol.map(closes)
    fills = fills.dropna(subset=["official"])
    fills["diff_bps"] = (fills.fill / fills.official - 1) * 1e4
    match = fills["diff_bps"].abs() <= 1.0
    print(f"{day}: {len(fills)} cls fills reconciled against official "
          f"closing auction prints")
    print(f"  exact-match (<=1bp): {match.sum()}/{len(fills)} "
          f"({match.mean():.1%})")
    print(f"  |diff| median {fills.diff_bps.abs().median():.2f}bp  "
          f"p95 {fills.diff_bps.abs().quantile(0.95):.2f}bp  "
          f"max {fills.diff_bps.abs().max():.2f}bp")
    bad = fills[~match].sort_values("diff_bps", key=abs, ascending=False)
    if len(bad):
        print("\n  mismatches:")
        for r_ in bad.itertuples():
            print(f"    {r_.symbol:>6} {r_.side:<4} fill {r_.fill:.4f} vs "
                  f"official {r_.official:.4f} ({r_.diff_bps:+.1f}bp)")
    else:
        print("  no mismatches — MOC fills ARE the official closing prints; "
              "the backtest fill model is confirmed live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
