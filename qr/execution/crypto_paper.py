"""
Crypto TSMOM paper forward test — frozen spec exp14 (hash f3a5afb0).
====================================================================

WHAT THIS TRADES
    The one candidate that passed pre-registered criteria in 220 trials:
    30-day time-series momentum on BTC/ETH, long-or-flat, evaluated every
    7 calendar days, equal weight. Validation 2021-23: net Sharpe 1.13 vs
    buy-hold 0.75, MaxDD -50% vs -76%. It is crypto beta with a mechanical
    tail-cutter, NOT provable alpha — the forward test exists to measure
    (1) live tracking vs the backtest rule, (2) realised fill quality vs
    the modelled 11bps/side, (3) operational survival. Same philosophy as
    the low-vol equity runner: measurement, not profit.

SPEC FIDELITY
    - Signal source is Yahoo daily closes (same as the frozen backtest),
      cached in qr/data/interim/crypto_majors_daily.parquet. Alpaca is the
      fill venue only. Divergence between the two is journalled, not hidden.
    - The 7-day evaluation grid is anchored at the FIRST ROW of the cached
      panel (backtest used iloc[::7]); extending the cache preserves phase.
    - Backtest applies the position from close D+1 after an evaluation at
      close D. This runner executes daily at 06:15 PT and reconciles the
      held position to the latest grid target — inside the (D, D+1] window
      the backtest allows, never ahead of the signal.
    - Positions are banded: rebalance only when held notional deviates from
      target by >5% of the per-coin sleeve, so price drift does not generate
      daily dust trades the backtest never modelled.

CAPITAL
    $20,000 notional sleeve ($10,000 per coin when long) inside the same
    paper account as the $100,000 low-vol equity book. Combined gross can
    reach 1.2x account equity on margin buying power; acceptable for a
    measurement account, journalled per-strategy so attribution stays clean.

SAFETY
    Orders go through `_assert_paper` — the live endpoint raises. Crypto
    orders use time_in_force=gtc (crypto has no session close).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from qr.execution.live_paper import (  # noqa: E402
    PAPER_BASE, _assert_paper, _db, _headers)

logger = logging.getLogger("crypto")

CRYPTO_DATA = "https://data.alpaca.markets/v1beta3/crypto/us"
CACHE = HERE.parent / "data" / "interim" / "crypto_majors_daily.parquet"

COINS = {"BTC-USD": "BTC/USD", "ETH-USD": "ETH/USD"}   # yahoo -> alpaca
LOOKBACK = 30
GRID_DAYS = 7
SLEEVE = 20_000.0
PER_COIN = SLEEVE / len(COINS)
BAND = 0.05                      # rebalance when off-target by >5% of sleeve
MIN_ORDER_USD = 10.0


# ------------------------------------------------------------------- signal
def load_prices() -> pd.DataFrame:
    """Extend the frozen backtest's price cache; drop today's partial bar."""
    import yfinance as yf
    cached = pd.read_parquet(CACHE)
    cached.index = pd.to_datetime(cached.index)
    start = cached.index.max() - pd.Timedelta(days=5)
    fresh = {}
    for y in COINS:
        h = yf.Ticker(y).history(start=start.strftime("%Y-%m-%d"),
                                 auto_adjust=False)
        h.index = pd.to_datetime(h.index).tz_localize(None)
        fresh[y] = h["Close"]
    fdf = pd.DataFrame(fresh)
    out = pd.concat([cached[~cached.index.isin(fdf.index)], fdf]).sort_index()
    today_utc = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    out = out[out.index < today_utc]          # completed days only
    out.to_parquet(CACHE)
    return out


def grid_target(px: pd.DataFrame) -> dict:
    """Latest 7-day-grid evaluation and its long/flat target per coin."""
    n = len(px)
    last_grid_i = ((n - 1) // GRID_DAYS) * GRID_DAYS      # iloc anchor at row 0
    grid_date = px.index[last_grid_i]
    mom = px.iloc[last_grid_i] / px.iloc[last_grid_i - LOOKBACK] - 1.0
    return {y: {"grid_date": str(grid_date.date()),
                "mom_30d": float(mom[y]),
                "target_usd": PER_COIN if mom[y] > 0 else 0.0}
            for y in COINS}


# ----------------------------------------------------------------- exchange
def crypto_quote(alp_sym: str) -> dict:
    r = requests.get(f"{CRYPTO_DATA}/latest/quotes",
                     params={"symbols": alp_sym}, headers=_headers(), timeout=30)
    if not r.ok:
        return {}
    q = (r.json().get("quotes") or {}).get(alp_sym, {})
    bid, ask = q.get("bp", 0), q.get("ap", 0)
    if not bid or not ask or ask <= bid:
        return {}
    return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2}


def crypto_positions() -> dict:
    r = requests.get(f"{PAPER_BASE}/positions", headers=_headers(), timeout=45)
    out = {}
    if r.ok:
        for p in r.json():
            if p.get("asset_class") == "crypto":
                out[p["symbol"].replace("USD", "/USD")] = {
                    "qty": float(p["qty"]),
                    "market_value": float(p["market_value"])}
    return out


def submit_crypto(alp_sym: str, side: str, notional: float | None = None,
                  qty: float | None = None) -> dict:
    url = f"{PAPER_BASE}/orders"
    _assert_paper(url)
    body = {"symbol": alp_sym, "side": side, "type": "market",
            "time_in_force": "gtc"}
    if notional is not None:
        body["notional"] = f"{notional:.2f}"
    else:
        body["qty"] = f"{qty:.9f}".rstrip("0").rstrip(".")
    r = requests.post(url, headers=_headers(), json=body, timeout=45)
    if not r.ok:
        return {"status": "rejected", "error": r.text[:200]}
    return r.json()


def await_fill(order_id: str, tries: int = 10) -> dict:
    for _ in range(tries):
        r = requests.get(f"{PAPER_BASE}/orders/{order_id}",
                         headers=_headers(), timeout=30)
        if r.ok:
            o = r.json()
            if o.get("status") == "filled":
                return o
        time.sleep(2)
    return o if r.ok else {}


# ---------------------------------------------------------------------- run
def _tables(con):
    con.execute("""CREATE TABLE IF NOT EXISTS crypto_intents(
        run_id TEXT, ts TEXT, symbol TEXT, side TEXT, notional REAL,
        grid_date TEXT, mom_30d REAL, target_usd REAL, held_usd REAL,
        ref_bid REAL, ref_ask REAL, ref_mid REAL,
        order_id TEXT, status TEXT, note TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS crypto_fills(
        run_id TEXT, ts TEXT, symbol TEXT, side TEXT, qty REAL,
        fill_price REAL, ref_mid REAL, realised_slip_bps REAL, notional REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS crypto_daily(
        ts TEXT, date TEXT PRIMARY KEY, grid_date TEXT,
        btc_mom REAL, eth_mom REAL, btc_target REAL, eth_target REAL,
        btc_held REAL, eth_held REAL, sleeve_value REAL)""")


def run_once(dry_run: bool = False) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    con = _db()
    _tables(con)

    px = load_prices()
    tgt = grid_target(px)
    held = {} if dry_run else crypto_positions()
    logger.info("grid %s | %s", next(iter(tgt.values()))["grid_date"],
                {y: f"mom={t['mom_30d']:+.3f} target=${t['target_usd']:.0f}"
                 for y, t in tgt.items()})

    orders = []
    for y, alp in COINS.items():
        t = tgt[y]
        held_usd = held.get(alp, {}).get("market_value", 0.0)
        delta = t["target_usd"] - held_usd
        quote = crypto_quote(alp)
        act = abs(delta) > max(BAND * PER_COIN, MIN_ORDER_USD)
        note = ""
        if act:
            side = "buy" if delta > 0 else "sell"
            if dry_run:
                res = {"status": "dry_run", "id": ""}
            elif side == "buy":
                res = submit_crypto(alp, "buy", notional=abs(delta))
            elif t["target_usd"] == 0 and held.get(alp):
                res = submit_crypto(alp, "sell", qty=held[alp]["qty"])
            else:
                qty = abs(delta) / quote["mid"] if quote else None
                res = (submit_crypto(alp, "sell", qty=qty) if qty else
                       {"status": "skipped", "error": "no quote for partial sell"})
            status, oid = res.get("status", "?"), str(res.get("id", ""))
            note = res.get("error", "")
        else:
            side, status, oid = "hold", "in_band", ""
        orders.append({"run_id": run_id,
                       "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "symbol": alp, "side": side, "notional": abs(delta),
                       "grid_date": t["grid_date"], "mom_30d": t["mom_30d"],
                       "target_usd": t["target_usd"], "held_usd": held_usd,
                       "ref_bid": quote.get("bid"), "ref_ask": quote.get("ask"),
                       "ref_mid": quote.get("mid"),
                       "order_id": oid, "status": status, "note": note})
        # fill journal with slippage vs decision-time mid
        if not dry_run and oid and status not in ("rejected", "skipped"):
            o = await_fill(oid)
            if o.get("status") == "filled" and quote:
                fp = float(o["filled_avg_price"])
                sgn = 1 if o["side"] == "buy" else -1
                slip = sgn * (fp / quote["mid"] - 1) * 1e4
                con.execute("INSERT INTO crypto_fills VALUES(?,?,?,?,?,?,?,?,?)",
                            (run_id, o.get("filled_at", ""), alp, o["side"],
                             float(o["filled_qty"]), fp, quote["mid"], slip,
                             float(o["filled_qty"]) * fp))
                logger.info("%s %s filled %.6f @ %.2f (slip %+.1fbps vs mid)",
                            alp, o["side"], float(o["filled_qty"]), fp, slip)

    od = pd.DataFrame(orders)
    od.to_sql("crypto_intents", con, if_exists="append", index=False)

    held2 = {} if dry_run else crypto_positions()
    b, e = tgt["BTC-USD"], tgt["ETH-USD"]
    con.execute("INSERT OR REPLACE INTO crypto_daily VALUES(?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                 b["grid_date"], b["mom_30d"], e["mom_30d"],
                 b["target_usd"], e["target_usd"],
                 held2.get("BTC/USD", {}).get("market_value", 0.0),
                 held2.get("ETH/USD", {}).get("market_value", 0.0),
                 sum(p["market_value"] for p in held2.values())))
    con.commit()
    acted = [o for o in orders if o["status"] not in ("in_band",)]
    logger.info("done: %d order(s), %d in-band hold(s)",
                len(acted), len(orders) - len(acted))
    return {"run_id": run_id, "orders": len(acted), "dry_run": dry_run}


def status() -> dict:
    con = _db()
    d = pd.read_sql("SELECT * FROM crypto_daily ORDER BY date DESC LIMIT 7", con)
    f = pd.read_sql("SELECT COUNT(*) n, AVG(realised_slip_bps) slip "
                    "FROM crypto_fills", con)
    return {"recent": d.to_dict("records"),
            "fills": int(f.n[0]), "mean_slip_bps": (None if f.n[0] == 0
                                                    else round(float(f.slip[0]), 2))}


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
            print("usage: crypto_paper.py [dryrun|run|status]")
            return 2
        return 0
    except Exception:                                      # noqa: BLE001
        import traceback
        logger.error("CRYPTO RUN FAILED\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
