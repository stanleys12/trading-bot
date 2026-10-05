"""
Daily forward-test runner. Scheduled, unattended, paper money only.
===================================================================

WHAT IT DOES EACH TRADING DAY
  1. Mark the portfolio to market and record equity.
  2. Record the benchmark (SPY) the same day, so every number is relative.
  3. Rebalance ONLY on scheduled days -- the frozen spec is monthly. Marking
     daily and trading monthly is the point: turnover is what killed most of
     this project's candidates, and a runner that rebalanced daily would be
     testing a different strategy than the one that was researched.
  4. Reconcile any fills that are now old enough to check against SIP.

WHAT ONE DAY OF P&L MEANS
Nothing. A 25-name low-volatility book runs ~11% annualised volatility, so a
typical day is +/-0.7% of noise. To distinguish a 4%/yr alpha from zero at
this volatility takes decades -- that arithmetic is the central finding of the
whole project and it does not change because we are now live.

What accrues quickly is the thing this is actually for: realised slippage
versus modelled cost, and whether the pipeline survives real trading days.
Those converge in weeks, not decades.

SAFETY
Paper endpoint only, enforced in live_paper._assert_paper. This file cannot
place a real order. It also never raises out of the scheduled path -- a cron
job that dies silently is worse than one that logs a failure and continues.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE.parent / "reports" / "live"
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE.parents[1]))
from qr.execution.live_paper import (  # noqa: E402
    DATA_BASE, PAPER_BASE, RunConfig, _db, _headers, market_open, positions,
    reconcile, repair_drift, run_once, status)

logger = logging.getLogger("daily")

REBALANCE_DAY = 1          # (unused) legacy monthly constant; spec is quarterly

# Symbols in this account that belong to ANOTHER book and must never be
# counted as, or traded by, the low-vol equity strategy. IWM is the
# covered-call book's collateral.
RESERVED = {"IWM"}


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_DIR / "daily.log"),
                  logging.StreamHandler()])


def daily_closes(symbols: list[str], days: int = 12) -> dict[str, list]:
    """
    {symbol: [(date, close), ...]} oldest-first, from DAILY bars.

    2026-08-26: this replaces a call to /stocks/bars/latest, which returns
    the latest ONE-MINUTE bar, not the day's bar. The benchmark was therefore
    SPY's move over a single minute -- typically +/-0.01% -- against which
    the portfolio's multi-hour return looked like enormous alpha. Every
    excess-return number recorded before this date is meaningless for that
    reason; see daily_v2 below.
    """
    h = _headers()
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    out: dict[str, list] = {}
    for i in range(0, len(symbols), 100):
        batch = symbols[i:i + 100]
        r = requests.get(f"{DATA_BASE}/stocks/bars", headers=h,
                         params={"symbols": ",".join(batch), "timeframe": "1Day",
                                 "start": start, "feed": "iex", "limit": 10000},
                         timeout=45)
        if not r.ok:
            logger.warning("daily bars: %s", r.text[:120])
            continue
        for sym, bars in (r.json().get("bars") or {}).items():
            out[sym] = [(b["t"][:10], float(b["c"])) for b in bars
                        if b.get("c")]
    return out


def session_returns(held: pd.DataFrame) -> dict:
    """
    The equity sleeve's return over the most recent COMPLETED session, and
    SPY's return over that identical session.

    Two deliberate choices:

    * CLOSE-TO-CLOSE over a finished session, not "prior close to right now".
      The old code chained prior-close->09:40 snapshots, which silently
      dropped the 09:40->close part of every day, so the cumulative series
      measured overnight drift rather than the strategy.

    * The sleeve is priced FROM ITS OWN HOLDINGS, never from account equity.
      Three books share this Alpaca account; account equity also moves with
      the crypto TSMOM sleeve, the covered-call book and margin. On
      2026-08-26 that conflation was reporting +3.4% "strategy" return when
      the equity book itself was -0.8% and every dollar of the gain was
      crypto's.

    Weights are current market values. Between quarterly rebalances the book
    does not trade, so these are the session's starting weights.
    """
    syms = [s for s in held.symbol.tolist() if s not in RESERVED]
    closes = daily_closes(sorted(set(syms + ["SPY"])))
    spy = closes.get("SPY", [])
    if len(spy) < 2:
        return {}
    session_date, spy_c = spy[-1]
    spy_prev = spy[-2][1]
    spy_pct = (spy_c / spy_prev - 1) * 100

    mv = dict(zip(held.symbol, held.market_value.abs()))
    num = den = 0.0
    priced = 0
    for s in syms:
        c = closes.get(s, [])
        if len(c) < 2 or c[-1][0] != session_date:
            continue                      # halted, delisted, or no bar today
        r = c[-1][1] / c[-2][1] - 1
        w = mv.get(s, 0.0)
        if w <= 0:
            continue
        num += w * r
        den += w
        priced += 1
    if den <= 0:
        return {"session_date": session_date, "spy_day_pct": spy_pct,
                "spy_close": spy_c, "n_priced": 0, "coverage_pct": 0.0}
    book_pct = (num / den) * 100
    return {"session_date": session_date, "spy_close": spy_c,
            "spy_day_pct": spy_pct, "book_day_pct": book_pct,
            "excess_vs_spy_pct": book_pct - spy_pct,
            "n_priced": priced,
            "coverage_pct": 100.0 * priced / max(len(syms), 1)}


FORCE_FLAG = LOG_DIR / "FORCE_REBALANCE"


def is_rebalance_day() -> bool:
    """
    First trading day of the QUARTER, per Alpaca's calendar.

    2026-08-17: was monthly, which was part of the spec drift (the frozen
    validated spec is quarterly). A one-time transition to the corrected
    spec is triggered by the FORCE_REBALANCE flag file, which is consumed
    on use so it cannot fire twice.
    """
    if FORCE_FLAG.exists():
        FORCE_FLAG.unlink()
        logger.info("FORCE_REBALANCE flag consumed -- one-time rebalance")
        return True
    h = _headers()
    today = pd.Timestamp.now(tz="America/New_York").normalize()
    if today.month not in (1, 4, 7, 10):
        return False
    first = today.replace(day=1)
    r = requests.get(f"{PAPER_BASE}/calendar", headers=h,
                     params={"start": first.strftime("%Y-%m-%d"),
                             "end": today.strftime("%Y-%m-%d")}, timeout=30)
    if not r.ok:
        return False
    days = [d["date"] for d in r.json()]
    return bool(days) and days[0] == today.strftime("%Y-%m-%d")


def mark_to_market() -> dict:
    """
    Record each book SEPARATELY, keyed by the completed session it measures.

    Written to `daily_v2`. The original `daily` table is left untouched as a
    record of what was reported, but it must not be chained: its `day_pct`
    was whole-account (three books plus margin) and its `spy_day_pct` was a
    one-minute bar. Both defects are described in daily_closes() and
    session_returns(). Starting a clean table is deliberate -- silently
    rewriting the old rows would erase the evidence of the error.
    """
    h = _headers()
    a = requests.get(f"{PAPER_BASE}/account", headers=h, timeout=30).json()
    pos = positions()                          # us_equity, includes IWM
    sleeve = pos[~pos.symbol.isin(RESERVED)] if len(pos) else pos
    crypto = positions(asset_class="crypto")
    opt = positions(asset_class="us_option")

    ret = session_returns(sleeve) if len(sleeve) else {}

    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_date": ret.get("session_date"),
        "run_date": pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d"),
        # --- the strategy under test -------------------------------------
        "sleeve_mv": float(sleeve.market_value.abs().sum()) if len(sleeve) else 0.0,
        "n_positions": len(sleeve),
        "book_day_pct": ret.get("book_day_pct"),
        "spy_day_pct": ret.get("spy_day_pct"),
        "spy_close": ret.get("spy_close"),
        "excess_vs_spy_pct": ret.get("excess_vs_spy_pct"),
        "coverage_pct": ret.get("coverage_pct"),
        # --- other books in the same account, for reference only ---------
        "crypto_mv": float(crypto.market_value.abs().sum()) if len(crypto) else 0.0,
        "option_mv": float(opt.market_value.sum()) if len(opt) else 0.0,
        "iwm_mv": float(pos[pos.symbol.isin(RESERVED)].market_value.sum())
                  if len(pos) else 0.0,
        "account_equity": float(a.get("equity", 0)),
        "cash": float(a.get("cash", 0)),
        "gross_exposure": float(a.get("long_market_value", 0) or 0),
    }

    con = _db()
    con.execute("""CREATE TABLE IF NOT EXISTS daily_v2(
        ts TEXT, session_date TEXT PRIMARY KEY, run_date TEXT,
        sleeve_mv REAL, n_positions INTEGER, book_day_pct REAL,
        spy_day_pct REAL, spy_close REAL, excess_vs_spy_pct REAL,
        coverage_pct REAL, crypto_mv REAL, option_mv REAL, iwm_mv REAL,
        account_equity REAL, cash REAL, gross_exposure REAL)""")
    if row["session_date"]:
        cols = ["ts", "session_date", "run_date", "sleeve_mv", "n_positions",
                "book_day_pct", "spy_day_pct", "spy_close",
                "excess_vs_spy_pct", "coverage_pct", "crypto_mv", "option_mv",
                "iwm_mv", "account_equity", "cash", "gross_exposure"]
        con.execute(f"INSERT OR REPLACE INTO daily_v2({','.join(cols)}) "
                    f"VALUES({','.join('?' * len(cols))})",
                    [row.get(c) for c in cols])
        con.commit()
    return row


def performance() -> pd.DataFrame:
    """
    Cumulative record, one row per COMPLETED session, correctly chained.

    Still means little over days -- a ~270-name low-vol book runs roughly 11%
    annualised vol, so daily noise dwarfs any plausible alpha. That arithmetic
    is the project's central finding and did not change because it went live.
    """
    con = _db()
    try:
        d = pd.read_sql("SELECT * FROM daily_v2 ORDER BY session_date", con)
    except Exception:                                      # noqa: BLE001
        return pd.DataFrame()
    if d.empty:
        return d
    d["cum_pct"] = ((1 + d.book_day_pct.fillna(0) / 100).cumprod() - 1) * 100
    d["cum_spy_pct"] = ((1 + d.spy_day_pct.fillna(0) / 100).cumprod() - 1) * 100
    d["cum_excess_pct"] = d.cum_pct - d.cum_spy_pct
    return d


def main() -> int:
    _setup_logging()
    logger.info("=" * 64)
    try:
        if not market_open():
            logger.info("market closed -- no action")
            return 0

        pos = positions()
        rebal = is_rebalance_day() or len(pos) == 0
        logger.info("positions=%d  rebalance_day=%s", len(pos), rebal)

        if rebal:
            res = run_once(RunConfig(), dry_run=False)
            logger.info("rebalanced: %s", json.dumps(res))
        else:
            # NOT a no-op. The spec trades quarterly, but the book must
            # actually REACH its target book: on 2026-08-19, 224 of 281 MOC
            # orders expired unfilled and nothing retried them, leaving a
            # half-built portfolio that would have been measured as the
            # strategy until October. repair_drift converges to the stored
            # target set without recomputing the signal, so the traded
            # cross-section is still the one the rebalance chose.
            logger.info("holding -- quarterly schedule, no new signal today")
            try:
                rep = repair_drift(dry_run=False)
                if rep.get("n_orders"):
                    logger.info("drift repair: %s", json.dumps(rep))
                elif rep.get("status") == "no_targets":
                    logger.warning("drift repair: no stored target book yet "
                                   "(seeded at the next rebalance)")
                else:
                    logger.info("drift repair: book on target "
                                "(%d skipped as sub-$25, %d exhausted)",
                                rep.get("skipped_small", 0),
                                rep.get("exhausted", 0))
            except Exception as exc:                       # noqa: BLE001
                logger.warning("drift repair skipped: %s", str(exc)[:160])

        row = mark_to_market()
        if row.get("book_day_pct") is None:
            logger.warning("no completed-session return computed "
                           "(session_date=%s, positions=%d)",
                           row.get("session_date"), row["n_positions"])
        else:
            logger.info(
                "session %s | LOW-VOL BOOK %+.2f%%  SPY %+.2f%%  excess %+.2f%%  "
                "(%d names, $%s, %.0f%% priced)",
                row["session_date"], row["book_day_pct"], row["spy_day_pct"],
                row["excess_vs_spy_pct"], row["n_positions"],
                f"{row['sleeve_mv']:,.0f}", row.get("coverage_pct") or 0)
        logger.info("other books (not the strategy): crypto $%s  IWM $%s  "
                    "short call $%s | account equity $%s  cash $%s  gross %.2fx",
                    f"{row['crypto_mv']:,.0f}", f"{row['iwm_mv']:,.0f}",
                    f"{row['option_mv']:,.0f}", f"{row['account_equity']:,.0f}",
                    f"{row['cash']:,.0f}",
                    row["gross_exposure"] / max(row["account_equity"], 1))

        try:
            m = reconcile()
            if len(m):
                g = m[m.n_quotes > 0]
                if len(g):
                    logger.info(
                        "reconciled %d fills | realised cost %.1fbps median "
                        "vs modelled %.1fbps (NBBO half-spread %.1fbps)",
                        len(g), g.slip_vs_nbbo_bps.median(),
                        g.modelled_cost_bps.median(),
                        (g.nbbo_spread_bps / 2).median())
                else:
                    logger.warning("reconcile: no NBBO recovered for %d fills",
                                   len(m))
        except Exception as exc:                           # noqa: BLE001
            logger.warning("reconcile skipped: %s", str(exc)[:120])

        perf = performance()
        if len(perf) > 1:
            last = perf.iloc[-1]
            logger.info("cumulative LOW-VOL BOOK %+.2f%%  SPY %+.2f%%  "
                        "excess %+.2f%%  over %d completed sessions",
                        last.cum_pct, last.cum_spy_pct,
                        last.cum_excess_pct, len(perf))
        return 0

    except Exception:                                      # noqa: BLE001
        # never die silently in a scheduled context
        logger.error("RUN FAILED\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
