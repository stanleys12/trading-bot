"""
exp10b: Are the backtest's daily opens/closes the official auction prints?
==========================================================================

WHY THIS MATTERS
    exp10 found the overnight decomposition is real (gross) on SPY/QQQ/IWM in
    both research and validation, but FAILED promotion at the pre-registered
    1bp/side auction slippage. At 0bp, IWM passes every criterion. So the
    entire verdict hangs on one empirical question:

        does a retail-size MOO/MOC order fill at the price the backtest used?

    Mechanically it should: opening/closing auctions are single-price crosses,
    MOC/MOO orders are guaranteed the official auction print, and retail size
    cannot move a multi-billion-dollar auction. The remaining gap is DATA: is
    Yahoo's daily open/close (what the backtest fills at) actually the
    official auction print (what the order would receive)?

MEASUREMENT
    Compare the frozen ETF panel's open/close against Alpaca SIP daily bars
    (official consolidated prints) for SPY/QQQ/IWM, 2016-2022 (test window
    untouched). Report the distribution of |difference| in bps.

VERDICT RULE (fixed before running)
    median |diff| <= 0.5bp and p95 <= 2bp on both open and close
        -> the 0bp cost scenario is the physically correct one; promote the
           question to a LIVE paper measurement of Alpaca cls/opg handling.
    otherwise
        -> the 1bp charge stands and exp10's FAIL is final.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.execution.live_paper import DATA_BASE, _headers  # noqa: E402

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
SYMBOLS = ["SPY", "QQQ", "IWM"]
START, END = "2016-01-01", "2022-12-31"


def alpaca_daily(symbols: list[str]) -> pd.DataFrame:
    rows, token = [], None
    while True:
        params = {"symbols": ",".join(symbols), "timeframe": "1Day",
                  "start": START, "end": END, "adjustment": "raw",
                  "feed": "sip", "limit": 10000}
        if token:
            params["page_token"] = token
        r = requests.get(f"{DATA_BASE}/stocks/bars", params=params,
                         headers=_headers(), timeout=60)
        r.raise_for_status()
        js = r.json()
        for sym, bars in (js.get("bars") or {}).items():
            for b in bars:
                rows.append({"symbol": sym, "date": b["t"][:10],
                             "ap_open": b["o"], "ap_close": b["c"]})
        token = js.get("next_page_token")
        if not token:
            break
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> None:
    ap = alpaca_daily(SYMBOLS)
    etf = pd.read_parquet(PROCESSED / "etf_panel.parquet")
    etf["date"] = pd.to_datetime(etf["date"])
    m = etf.merge(ap, on=["symbol", "date"], how="inner")
    m = m[(m["date"] >= START) & (m["date"] <= END)]

    print(f"matched days: {len(m)} across {m.symbol.nunique()} symbols\n")
    ok = True
    for sym in SYMBOLS:
        s = m[m.symbol == sym]
        for leg, a, b in [("open", "open", "ap_open"), ("close", "close", "ap_close")]:
            diff_bps = ((s[a] - s[b]).abs() / s[b] * 1e4)
            med, p95, mx = diff_bps.median(), diff_bps.quantile(0.95), diff_bps.max()
            verdict = "OK " if (med <= 0.5 and p95 <= 2.0) else "BAD"
            ok &= (med <= 0.5 and p95 <= 2.0)
            print(f"{sym} {leg:>5}: n={len(s):5d} median={med:5.2f}bp "
                  f"p95={p95:5.2f}bp max={mx:7.2f}bp  {verdict}")
    print(f"\nverdict: {'0bp auction scenario SUPPORTED by print data; '
          'next adjudicator is the live cls/opg paper probe'
          if ok else '1bp charge stands; exp10 FAIL is final'}")


if __name__ == "__main__":
    main()
