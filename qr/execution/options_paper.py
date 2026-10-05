"""
Covered-call paper book: the volatility risk premium, forward-tested.
=====================================================================

WHY THIS BOOK EXISTS
    The vol risk premium — buyers systematically overpaying for options —
    is the one heavily documented market premium this project never
    tested: free historical option chains do not exist, so it cannot be
    backtested honestly. It CAN be forward-tested for free. This runner
    sells one covered call on 100 IWM and measures, in the project's
    tradition, the numbers a backtest would have had to assume:
      1. option fill quality vs the quoted mid at retail size
      2. realised premium capture vs foregone upside, benchmarked against
         just holding IWM
      3. operational reality: assignment, expiry mechanics, thin quotes

SPEC (fixed 2026-08-17, before the first trade)
    Underlying   100 shares IWM (bought at market if not held)
    Call         nearest expiry inside 21-35 DTE (closest to 28); strike =
                 lowest strike >= spot * 1.025
    Entry order  SELL 1 contract, LIMIT at quoted mid (rounded to tick),
                 day TIF; if unfilled by the next daily run, reprice at
                 the BID (taker). Both attempts journalled.
    Management   none. Hold to expiry. Expires OTM -> sell the next one.
                 Assigned -> rebuy 100 IWM next run, sell the next one.
                 No rolling, no adjustments -- this measures the naive
                 harvest, not option-trader skill.
    Benchmark    buy-and-hold IWM, marked daily in the journal.

EXPECTATION, STATED HONESTLY
    Covered calls are long equity with capped upside. In a melt-up this
    book SHOULD trail its benchmark; the premium shows up in flat and
    choppy tape. Months of data before the comparison means anything;
    the fill-quality and premium-capture measurements accrue immediately.

SAFETY
    Paper only via _assert_paper. The account runs ~1.4x gross across
    three books; acceptable for measurement, journalled per book.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from qr.execution.live_paper import (  # noqa: E402
    PAPER_BASE, _assert_paper, _db, _headers, market_open)

logger = logging.getLogger("options")

OPT_DATA = "https://data.alpaca.markets/v1beta1/options"
UNDERLYING = "IWM"
SHARES = 100
OTM_PCT = 1.025
DTE_LO, DTE_HI, DTE_TARGET = 21, 35, 28


# ------------------------------------------------------------------- market
def stock_quote(symbol: str) -> dict:
    r = requests.get("https://data.alpaca.markets/v2/stocks/quotes/latest",
                     params={"symbols": symbol, "feed": "iex"},
                     headers=_headers(), timeout=30)
    q = ((r.json().get("quotes") or {}).get(symbol, {})) if r.ok else {}
    bid, ask = q.get("bp", 0), q.get("ap", 0)
    if bid and ask and ask > bid:
        return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2}
    return {}


def option_quote(occ: str) -> dict:
    r = requests.get(f"{OPT_DATA}/quotes/latest",
                     params={"symbols": occ, "feed": "indicative"},
                     headers=_headers(), timeout=30)
    q = ((r.json().get("quotes") or {}).get(occ, {})) if r.ok else {}
    bid, ask = q.get("bp", 0), q.get("ap", 0)
    if bid and ask and ask >= bid:
        return {"bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 2)}
    return {}


def pick_contract(spot: float) -> dict | None:
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    lo = (today + pd.Timedelta(days=DTE_LO)).strftime("%Y-%m-%d")
    hi = (today + pd.Timedelta(days=DTE_HI)).strftime("%Y-%m-%d")
    r = requests.get(f"{PAPER_BASE}/options/contracts", headers=_headers(),
                     params={"underlying_symbols": UNDERLYING, "type": "call",
                             "status": "active", "expiration_date_gte": lo,
                             "expiration_date_lte": hi, "limit": 1000},
                     timeout=45)
    if not r.ok:
        return None
    cs = r.json().get("option_contracts") or []
    min_strike = spot * OTM_PCT
    best = None
    for c in cs:
        strike = float(c["strike_price"])
        if strike < min_strike:
            continue
        dte = (pd.Timestamp(c["expiration_date"]) - today).days
        key = (abs(dte - DTE_TARGET), strike)
        if best is None or key < best[0]:
            best = (key, c)
    return best[1] if best else None


# ---------------------------------------------------------------- positions
def book_positions() -> dict:
    r = requests.get(f"{PAPER_BASE}/positions", headers=_headers(), timeout=45)
    out = {"shares": 0, "short_call": None, "call_qty": 0}
    if r.ok:
        for p in r.json():
            if p["symbol"] == UNDERLYING and p.get("asset_class") == "us_equity":
                out["shares"] = int(float(p["qty"]))
            elif (p.get("asset_class") == "us_option"
                  and p["symbol"].startswith(UNDERLYING)
                  and float(p["qty"]) < 0):
                out["short_call"] = p["symbol"]
                out["call_qty"] = int(float(p["qty"]))
                out["call_mv"] = float(p.get("market_value", 0))
    return out


def submit_order(body: dict) -> dict:
    url = f"{PAPER_BASE}/orders"
    _assert_paper(url)
    r = requests.post(url, headers=_headers(), json=body, timeout=45)
    if not r.ok:
        return {"status": "rejected", "error": r.text[:200]}
    return r.json()


def open_orders() -> list[dict]:
    r = requests.get(f"{PAPER_BASE}/orders", headers=_headers(),
                     params={"status": "open"}, timeout=30)
    return [o for o in (r.json() if r.ok else [])
            if o.get("symbol", "").startswith(UNDERLYING)]


# ---------------------------------------------------------------------- run
def _tables(con):
    con.execute("""CREATE TABLE IF NOT EXISTS opt_intents(
        run_id TEXT, ts TEXT, symbol TEXT, side TEXT, qty INTEGER,
        order_type TEXT, limit_price REAL, ref_bid REAL, ref_ask REAL,
        ref_mid REAL, order_id TEXT, status TEXT, note TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS opt_daily(
        ts TEXT, date TEXT PRIMARY KEY, iwm_shares INTEGER, iwm_mid REAL,
        short_call TEXT, call_mv REAL, book_value REAL)""")


def _journal_intent(con, run_id, symbol, side, qty, order_type, limit_price,
                    q, res, note=""):
    con.execute("INSERT INTO opt_intents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 symbol, side, qty, order_type, limit_price,
                 q.get("bid"), q.get("ask"), q.get("mid"),
                 str(res.get("id", "")), res.get("status", "?"),
                 note or res.get("error", "")))


def run_once(dry_run: bool = False) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    con = _db()
    _tables(con)
    actions = []

    if not market_open() and not dry_run:
        logger.info("market closed -- no action")
        return {"run_id": run_id, "actions": ["closed"]}

    pos = book_positions()
    uq = stock_quote(UNDERLYING)
    logger.info("book: %d IWM, short call %s | IWM mid %.2f",
                pos["shares"], pos["short_call"], uq.get("mid", 0))

    # 1) unfilled limit sell from a prior run -> reprice at the bid
    for o in open_orders():
        if o["symbol"] != UNDERLYING and o["side"] == "sell":
            oq = option_quote(o["symbol"])
            if not dry_run:
                requests.delete(f"{PAPER_BASE}/orders/{o['id']}",
                                headers=_headers(), timeout=30)
                if oq:
                    res = submit_order({"symbol": o["symbol"], "qty": "1",
                                        "side": "sell", "type": "limit",
                                        "limit_price": f"{oq['bid']:.2f}",
                                        "time_in_force": "day"})
                    _journal_intent(con, run_id, o["symbol"], "sell", 1,
                                    "limit_at_bid", oq["bid"], oq, res,
                                    "reprice: mid was not filled in a day")
                    actions.append("repriced_at_bid")

    # 2) ensure the 100-share cover
    if pos["shares"] < SHARES and uq:
        need = SHARES - pos["shares"]
        res = ({"status": "dry_run", "id": ""} if dry_run else
               submit_order({"symbol": UNDERLYING, "qty": str(need),
                             "side": "buy", "type": "market",
                             "time_in_force": "day"}))
        _journal_intent(con, run_id, UNDERLYING, "buy", need, "market",
                        None, uq, res, "establish/restore covered shares")
        actions.append(f"buy_{need}_shares")

    # 3) sell a call if none is short and none is resting
    resting_sells = [o for o in open_orders()
                     if o["symbol"] != UNDERLYING and o["side"] == "sell"]
    if (pos["short_call"] is None and not resting_sells and uq
            and pos["shares"] + (SHARES if f"buy_{SHARES}_shares"
                                 in " ".join(actions) else 0) >= SHARES):
        c = pick_contract(uq["mid"])
        if c:
            occ = c["symbol"]
            oq = option_quote(occ)
            if oq and oq["mid"] > 0:
                res = ({"status": "dry_run", "id": ""} if dry_run else
                       submit_order({"symbol": occ, "qty": "1",
                                     "side": "sell", "type": "limit",
                                     "limit_price": f"{oq['mid']:.2f}",
                                     "time_in_force": "day"}))
                _journal_intent(con, run_id, occ, "sell", 1, "limit_at_mid",
                                oq["mid"], oq, res,
                                f"strike {c['strike_price']} exp "
                                f"{c['expiration_date']}")
                actions.append(f"sell_call_{occ}")

    # 4) daily mark
    pos2 = pos if dry_run else book_positions()
    call_mv = pos2.get("call_mv", 0.0) or 0.0
    book_val = pos2["shares"] * uq.get("mid", 0) + call_mv
    con.execute("INSERT OR REPLACE INTO opt_daily VALUES(?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d"),
                 pos2["shares"], uq.get("mid"), pos2["short_call"],
                 call_mv, book_val))
    con.commit()
    logger.info("actions: %s | book value $%.0f", actions or ["hold"], book_val)
    return {"run_id": run_id, "actions": actions or ["hold"]}


def status() -> dict:
    con = _db()
    d = pd.read_sql("SELECT * FROM opt_daily ORDER BY date DESC LIMIT 7", con)
    i = pd.read_sql("SELECT ts, symbol, side, order_type, limit_price, status "
                    "FROM opt_intents ORDER BY ts DESC LIMIT 10", con)
    return {"recent_marks": d.to_dict("records"),
            "recent_intents": i.to_dict("records")}


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dryrun"
    try:
        if cmd == "dryrun":
            print(json.dumps(run_once(dry_run=True), indent=1))
        elif cmd == "run":
            print(json.dumps(run_once(dry_run=False), indent=1))
        elif cmd == "status":
            print(json.dumps(status(), indent=1, default=str))
        else:
            print("usage: options_paper.py [dryrun|run|status]")
            return 2
        return 0
    except Exception:                                      # noqa: BLE001
        import traceback
        logger.error("OPTIONS RUN FAILED\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
