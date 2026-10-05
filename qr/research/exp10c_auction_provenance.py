"""
exp10c: Auction-print provenance — the instrument exp10b said to use.
=====================================================================

THE QUESTION
    exp10 (overnight decomposition) died on one contingency: whether the
    daily open/close prices the backtest fills at are the official auction
    prints a retail MOO/MOC order is guaranteed to receive. exp10b compared
    Yahoo vs Alpaca DAILY BARS and found 0.5-5bp disagreement — but daily
    bars cannot say which source carries the auction print. The
    /v2/stocks/auctions endpoint returns the auction crosses themselves
    (e.g. IWM closing print, condition M, 1.1M shares on the primary
    exchange). That is the definitive reference.

MEASUREMENT
    For SPY/QQQ/IWM, 2016-2022 (equity TEST window untouched): per day take
    the largest-size print in the opening and closing auction arrays (the
    cross dwarfs everything else), and compare against the frozen ETF
    panel's open/close — the exact prices exp10's backtest filled at.

VERDICT RULE (fixed before running)
    A leg (open or close) is VERIFIED for a symbol if
        median |panel - auction| <= 0.25 bp  and  p95 <= 1.0 bp.
    The overnight trade buys the closing auction and sells the opening
    auction, so a symbol needs BOTH legs verified.

    If IWM passes both legs: register exp10c with outcome HOLD —
    the cost contingency resolves to ~0bp + SEC fee, but the statistical
    evidence (overnight mean t=2.5, vol 0.164 vs 0.264 in validation) is
    the SAME data exp10 already scored, so it earns no fresh statistical
    credit. Per the power rule adopted 2026-08-13: the powered claim here
    is the VARIANCE reduction (F-test over 1,259 days, decisive); the mean
    claim is suggestive only. Adjudication path: live cls/opg paper probe
    measuring realised auction fills, then the protocol's own gates.

    If no symbol passes both legs: the family stays dead, and this
    measurement is the record of why.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.execution.live_paper import _headers  # noqa: E402
from qr.research.protocol import Experiment, register  # noqa: E402

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
SYMBOLS = ["SPY", "QQQ", "IWM"]
START, END = "2016-01-01", "2022-12-31"


def fetch_auctions(symbol: str) -> pd.DataFrame:
    rows, token = [], None
    while True:
        params = {"symbols": symbol, "start": START, "end": END,
                  "feed": "sip", "limit": 10000}
        if token:
            params["page_token"] = token
        r = requests.get("https://data.alpaca.markets/v2/stocks/auctions",
                         params=params, headers=_headers(), timeout=60)
        r.raise_for_status()
        js = r.json()
        for day in (js.get("auctions") or {}).get(symbol, []):
            rec = {"date": day["d"]}
            for leg in ["o", "c"]:
                prints = day.get(leg) or []
                if prints:
                    best = max(prints, key=lambda p: p.get("s", 0))
                    rec[f"auction_{leg}"] = best["p"]
                    rec[f"auction_{leg}_size"] = best["s"]
            rows.append(rec)
        token = js.get("next_page_token")
        if not token:
            break
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> None:
    etf = pd.read_parquet(PROCESSED / "etf_panel.parquet")
    etf["date"] = pd.to_datetime(etf["date"])

    verified = {}
    metrics = {}
    for sym in SYMBOLS:
        a = fetch_auctions(sym)
        m = etf[etf.symbol == sym].merge(a, on="date", how="inner")
        m = m[(m.date >= START) & (m.date <= END)]
        legs_ok = {}
        for leg, panel_col, auc_col in [("open", "open", "auction_o"),
                                        ("close", "close", "auction_c")]:
            s = m.dropna(subset=[auc_col])
            diff = (s[panel_col] - s[auc_col]).abs() / s[auc_col] * 1e4
            med, p95 = diff.median(), diff.quantile(0.95)
            ok = bool(med <= 0.25 and p95 <= 1.0)
            legs_ok[leg] = ok
            metrics[f"{sym}_{leg}_median_bp"] = round(float(med), 3)
            metrics[f"{sym}_{leg}_p95_bp"] = round(float(p95), 3)
            cov = len(s) / len(m) if len(m) else 0
            print(f"{sym} {leg:>5}: n={len(s):5d} (cov {cov:4.0%}) "
                  f"median={med:6.3f}bp p95={p95:6.3f}bp "
                  f"median cross ${s[auc_col.replace('auction','auction') + '_size'].median()*s[auc_col].median()/1e6:,.0f}M"
                  f"  {'VERIFIED' if ok else 'not verified'}")
        verified[sym] = legs_ok["open"] and legs_ok["close"]

    any_ok = [s for s, v in verified.items() if v]
    print(f"\nsymbols with BOTH legs verified: {any_ok or 'none'}")

    register(Experiment(
        name="exp10c_auction_provenance",
        family="intraday_timing",
        hypothesis="the backtest's daily open/close ARE the official auction "
                   "crosses, so retail MOO/MOC implements exp10's overnight "
                   "decomposition at ~0bp + SEC fee",
        features=["official_auction_prints"],
        split="2016-2022 (measurement; equity TEST window untouched)",
        universe=",".join(SYMBOLS),
        rebalance_days=0, holding_days=0,
        n_observations=int(len(etf[etf.symbol.isin(SYMBOLS)])),
        metrics=metrics | {f"{s}_both_legs": bool(v) for s, v in verified.items()},
        outcome="HOLD" if any_ok else "FAIL",
        failure_reason="" if any_ok else
        "no symbol's panel prices match official auction crosses within "
        "0.25bp median / 1bp p95 on both legs; the overnight family's cost "
        "contingency stays unresolved and the family stays dead",
    ))
    if any_ok:
        print("\nHOLD registered: cost contingency resolved for", any_ok,
              "\nNo fresh statistical credit (same data as exp10). Next "
              "adjudicator: live cls/opg paper probe, then protocol gates.")


if __name__ == "__main__":
    main()
