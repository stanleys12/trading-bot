#!/usr/bin/env python3
"""Prediction-market spread recorder (Kalshi + Polymarket), phase 1: MEASURE.

Per the sprint charter's discipline: before any paper trade, record several
days of real two-venue quotes and answer, with data: how often do executable
cross-venue spreads exist net of fees, how big, and how long do they last?
Focus: crypto price markets (documented hottest cross-venue divergence) —
they also have unambiguous, comparable resolution (a number vs a strike).

Every 60s: snapshot all open Kalshi BTC/ETH markets (yes_bid/yes_ask) and
Polymarket crypto markets (bestBid/bestAsk) -> fund/reports/pm_snapshots/.
Matching + spread analysis run OFFLINE on the recorded data — the recorder
just captures faithfully. Zero model tokens; pure free-API polling.
"""
import json, re, time
from datetime import datetime, timezone
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "reports" / "pm_snapshots"
OUT.mkdir(parents=True, exist_ok=True)
KEY = re.compile(r"\b(btc|bitcoin|eth|ethereum)\b", re.I)


KALSHI_SERIES = ["KXBTC", "KXBTCD", "KXETH", "KXETHD"]

def kalshi():
    rows = []
    for st in KALSHI_SERIES:
        cursor = ""
        for _ in range(5):
            p = {"series_ticker": st, "status": "open", "limit": 200}
            if cursor:
                p["cursor"] = cursor
            r = requests.get("https://api.elections.kalshi.com/trade-api/v2/markets",
                             params=p, timeout=20).json()
            for m in r.get("markets", []):
                rows.append({"venue": "kalshi", "id": m["ticker"], "series": st,
                             "q": m.get("title"),
                             "strike": m.get("floor_strike") or m.get("cap_strike"),
                             "close": m.get("close_time"),
                             "yes_bid": m.get("yes_bid_dollars"),
                             "yes_ask": m.get("yes_ask_dollars"),
                             "bid_size": m.get("yes_bid_size_fp"),
                             "ask_size": m.get("yes_ask_size_fp"),
                             "last": m.get("last_price_dollars"),
                             "vol24": m.get("volume_24h_fp")})
            cursor = r.get("cursor")
            if not cursor:
                break
    return rows


STRICT = re.compile(r"above \$[\d,]+ on \w+ \d+", re.I)

def clob_book(token_id):
    try:
        b = requests.get("https://clob.polymarket.com/book",
                         params={"token_id": token_id}, timeout=15).json()
        bb = max(b.get("bids", []), key=lambda x: float(x["price"]), default=None)
        ba = min(b.get("asks", []), key=lambda x: float(x["price"]), default=None)
        return {"clob_bid": bb and bb["price"], "clob_bid_sz": bb and bb["size"],
                "clob_ask": ba and ba["price"], "clob_ask_sz": ba and ba["size"]}
    except Exception:
        return {}

def polymarket():
    rows = []
    for off in range(0, 1000, 500):
        r = requests.get("https://gamma-api.polymarket.com/markets",
                         params={"limit": 500, "offset": off, "closed": "false",
                                 "order": "volume24hr", "ascending": "false"},
                         timeout=20).json()
        if not r:
            break
        for m in r:
            if KEY.search(m.get("question", "")):
                extra = {}
                # strict-semantics markets get the REAL executable book:
                # gamma display quotes proved stale in the 09-03 ETH-2500 audit
                if STRICT.search(m.get("question", "")) and m.get("clobTokenIds"):
                    try:
                        extra = clob_book(json.loads(m["clobTokenIds"])[0])
                    except Exception:
                        pass
                rows.append({"venue": "poly", "id": m.get("slug"), **extra,
                             "q": m.get("question"), "close": m.get("endDate"),
                             "best_bid": m.get("bestBid"), "best_ask": m.get("bestAsk"),
                             "prices": m.get("outcomePrices"),
                             "vol": m.get("volumeNum") or m.get("volume"),
                             "liq": m.get("liquidityNum")})
    return rows


def main():
    while True:
        try:
            ts = datetime.now(timezone.utc)
            snap = {"ts": ts.isoformat(timespec="seconds"),
                    "kalshi": kalshi(), "poly": polymarket()}
            f = OUT / f"{ts:%Y-%m-%d}.jsonl"
            with f.open("a") as fh:
                fh.write(json.dumps(snap) + "\n")
        except Exception as e:
            with (OUT / "recorder.err").open("a") as fh:
                fh.write(f"{datetime.now()} {str(e)[:200]}\n")
        time.sleep(60)


if __name__ == "__main__":
    main()
