"""
Paper-trading runner: the only remaining source of new information.
===================================================================

WHAT THIS IS FOR, PRECISELY

Not to make money. The strategy it trades (low-volatility, liquid universe)
has ~+4% alpha that cannot be validated in any obtainable sample, and is
purchasable as USMV for 15bps. It is here because it is the least-bad thing
211 trials produced and because trading SOMETHING is required to learn
anything about execution.

The spread audit already measured quoted spreads from 2M real quotes. What it
could NOT tell us is whether we actually GET that spread. Knowing the NBBO is
7.5bps does not tell you your fill lands at the midpoint, at the touch, or
through it. Alpaca's paper engine fills against live quotes, so the difference
between our modelled cost and the realised fill is a real measurement.

Two questions, then:

  1. EXECUTION.  realised slippage vs the quoted midpoint at decision time,
     per name, per dollar size. This is the number every backtest in this
     project assumed and none of them observed.

  2. OPERATIONS.  does the pipeline survive halts, missing quotes, stale
     bars, fractional-share rounding, rejected orders, and a symbol that
     stops existing? Backtests never encounter any of these. Live systems
     encounter all of them.

SAFETY
LIVE trading is structurally impossible here. The base URL is asserted to be
the paper endpoint on every call, and there is no code path that constructs a
live client. Flipping a flag is not sufficient to trade real money with this
file, which is deliberate.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DB = HERE.parent / "data" / "paper_trading.db"

PAPER_BASE = "https://paper-api.alpaca.markets/v2"
DATA_BASE = "https://data.alpaca.markets/v2"


def _headers() -> dict:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")
    if not k or not s:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET missing from .env")
    return {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}


def _assert_paper(url: str) -> None:
    """Structural guarantee. Every request that could place an order goes
    through here; a live endpoint raises rather than trading."""
    if not url.startswith(PAPER_BASE):
        raise RuntimeError(
            f"refusing to send an order to {url!r}. This module trades PAPER "
            "only, and that is not a configuration option.")


@dataclass
class RunConfig:
    """
    2026-08-17: corrected to the FROZEN VALIDATED SPEC. The runner had been
    trading 25 names rebalanced monthly -- 11x the concentration and 3x the
    turnover of what was actually validated (top 20% lowest-vol quantile,
    ~280 names, quarterly). That drift was flagged in the project report and
    is fixed now, four days into the forward test, while the cost of
    switching is one rebalance rather than months of data on the wrong spec.
    """
    capital: float = 100_000.0
    top_quantile: float = 0.20      # lowest-vol 20% of the eligible universe
    min_adv: float = 5e7            # $50M/day -- the liquid band we validated
    vol_lookback: int = 63
    max_position_pct: float = 0.05
    universe_size: int = 300


# --------------------------------------------------------------- market data
BAR_CACHE = HERE.parent / "data" / "interim" / "live_bars.parquet"


def get_bars(symbols: list[str], days: int = 200,
             min_coverage: float = 0.60) -> pd.DataFrame:
    """
    Daily bars, cached locally and refreshed incrementally.

    The first scheduled run of this function timed out and lost the whole
    trading day. It was re-fetching 200 days of history for ~1,800 symbols on
    every single run -- dozens of sequential paginated requests, any one of
    which could stall. A backtest never encounters that failure mode, which is
    precisely why it was not designed for.

    Three changes:
      - persist bars to disk and fetch only the missing tail
      - retry each batch with backoff instead of aborting the run
      - proceed on PARTIAL data if coverage clears `min_coverage`, because a
        signal computed on 1,400 of 1,800 names is still the signal, whereas
        trading nothing is a decision we did not intend to make.

    It refuses below the coverage floor rather than silently trading a
    truncated universe -- the same failure the download pipeline had earlier.
    """
    cached = pd.DataFrame()
    if BAR_CACHE.exists():
        try:
            cached = pd.read_parquet(BAR_CACHE)
            cached["date"] = pd.to_datetime(cached["date"])
        except Exception:                                  # noqa: BLE001
            cached = pd.DataFrame()

    # tz-naive throughout: parsed bar dates are naive, and mixing the two
    # raises on every comparison.
    end = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=2)
    need_from = end - pd.Timedelta(days=days * 2)
    if len(cached):
        have_to = cached["date"].max()
        if have_to >= end:
            logger.info("bar cache current through %s -- no fetch needed",
                        have_to.date())
            return cached[cached["date"] >= need_from]
        # only fetch the gap, plus a small overlap for safety
        need_from = max(need_from, have_to - pd.Timedelta(days=5))
        logger.info("bar cache through %s; fetching gap from %s",
                    have_to.date(), need_from.date())

    fresh = _fetch_bars_range(symbols, need_from, end)
    out = (pd.concat([cached, fresh], ignore_index=True)
           .drop_duplicates(subset=["symbol", "date"], keep="last")
           if len(cached) else fresh)
    if len(out):
        out = out.sort_values(["symbol", "date"])
        BAR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(BAR_CACHE)

    recent = out[out["date"] >= end - pd.Timedelta(days=10)] if len(out) else out
    cov = recent.symbol.nunique() / max(len(symbols), 1) if len(recent) else 0.0
    logger.info("bars: %d rows, %d/%d symbols with recent data (%.0f%% coverage)",
                len(out), recent.symbol.nunique() if len(recent) else 0,
                len(symbols), cov * 100)
    if cov < min_coverage:
        raise RuntimeError(
            f"only {cov:.0%} of the universe has recent bars (floor "
            f"{min_coverage:.0%}). Refusing to trade a truncated universe -- "
            "the signal would be computed on a different cross-section than "
            "the one that was researched.")
    return out[out["date"] >= end - pd.Timedelta(days=days * 2)]


def _fetch_bars_range(symbols: list[str], start, end,
                      days: int = 200) -> pd.DataFrame:
    """Daily bars from Alpaca -- same venue as the fills, so no source mismatch."""
    """Paginated fetch with per-batch retry. Never aborts the whole run."""
    h = _headers()
    out, failed = [], []
    for i in range(0, len(symbols), 100):
        batch = symbols[i:i + 100]
        page, ok = None, False
        for attempt in range(3):
            try:
                while True:
                    p = {"symbols": ",".join(batch), "timeframe": "1Day",
                         "limit": 10000, "adjustment": "all", "feed": "sip",
                         "start": start.strftime("%Y-%m-%d"),
                         "end": end.strftime("%Y-%m-%d")}
                    if page:
                        p["page_token"] = page
                    r = requests.get(f"{DATA_BASE}/stocks/bars", headers=h,
                                     params=p, timeout=30)
                    r.raise_for_status()
                    j = r.json()
                    for sym, rows in (j.get("bars") or {}).items():
                        for b in rows:
                            out.append({"symbol": sym, "date": b["t"][:10],
                                        "close": b["c"], "volume": b["v"],
                                        "vwap": b.get("vw")})
                    page = j.get("next_page_token")
                    if not page:
                        break
                ok = True
                break
            except Exception as exc:                       # noqa: BLE001
                logger.warning("bars batch %d attempt %d: %s",
                               i // 100, attempt + 1, str(exc)[:80])
                page = None
                time.sleep(4 * (attempt + 1))
        if not ok:
            failed.append(i // 100)
        time.sleep(0.2)
    if failed:
        logger.warning("%d batch(es) failed after retries: %s", len(failed), failed)
    df = pd.DataFrame(out)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
    return df


def get_quotes(symbols: list[str]) -> pd.DataFrame:
    """
    Quote visible AT DECISION TIME. Necessarily IEX: `quotes/latest` counts as
    recent SIP data, which the free tier blocks, so the consolidated NBBO is
    not observable in real time.

    That is a real operational constraint, not a workaround -- a live system on
    this tier genuinely cannot see the NBBO when it decides. We record what we
    could actually see, and reconcile later against historical SIP quotes at
    the fill timestamp (see `true_nbbo_at`), which becomes available at T+2.
    The gap between the two is itself a measurement worth having.
    """
    h = _headers()
    rows = []
    for i in range(0, len(symbols), 100):
        batch = symbols[i:i + 100]
        r = requests.get(f"{DATA_BASE}/stocks/quotes/latest", headers=h,
                         params={"symbols": ",".join(batch), "feed": "iex"}, timeout=45)
        if not r.ok:
            logger.warning("quotes: %s", r.text[:120])
            continue
        for sym, q in (r.json().get("quotes") or {}).items():
            bid, ask = q.get("bp", 0), q.get("ap", 0)
            if not bid or not ask or ask <= bid:
                bid = ask = 0          # one-sided IEX quote: unusable reference
            rows.append({"symbol": sym, "bid": bid, "ask": ask,
                         "mid": (bid + ask) / 2 if bid and ask else np.nan,
                         "spread_bps": ((ask - bid) / ((ask + bid) / 2) * 1e4)
                                        if bid and ask and ask > bid else np.nan,
                         "quote_time": q.get("t")})
        time.sleep(0.25)
    cols = ["symbol", "bid", "ask", "mid", "spread_bps", "quote_time"]
    return pd.DataFrame(rows, columns=cols) if not rows else pd.DataFrame(rows)


def true_nbbo_at(symbol: str, ts_iso: str, window_s: int = 60) -> dict:
    """
    Consolidated NBBO around a fill timestamp, from historical SIP.

    Only queryable once the data is >~1 day old, which is why reconciliation
    runs at T+2 rather than immediately. This is the honest reference price:
    what the whole market showed at the moment we traded, not what IEX alone
    happened to be quoting.
    """
    t = pd.Timestamp(ts_iso)
    p = {"symbols": symbol, "feed": "sip", "limit": 200,
         "start": (t - pd.Timedelta(seconds=window_s)).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "end": (t + pd.Timedelta(seconds=window_s)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    r = requests.get(f"{DATA_BASE}/stocks/quotes", headers=_headers(),
                     params=p, timeout=45)
    if not r.ok:
        return {}
    qs = [q for q in (r.json().get("quotes") or {}).get(symbol, [])
          if q.get("bp", 0) > 0 and q.get("ap", 0) > q.get("bp", 0)]
    if not qs:
        return {}
    mids = [(q["ap"] + q["bp"]) / 2 for q in qs]
    sprd = [(q["ap"] - q["bp"]) / m * 1e4 for q, m in zip(qs, mids)]
    return {"nbbo_mid": float(np.median(mids)),
            "nbbo_spread_bps": float(np.median(sprd)), "n_quotes": len(qs)}


# ------------------------------------------------------------------ universe
def tradable_universe(cfg: RunConfig) -> list[str]:
    """
    COMMON STOCKS ONLY, intersected with the researched panel.

    The first dry run of this function returned a universe including
    ARCA-listed ETFs, and "lowest 63-day volatility" then selected entirely
    bond funds -- BIL, AGG, BND, BLV. Not wrong as an allocation, but it is
    not the strategy that was validated, which ran on common stocks. Trading
    it would have been a silent specification mismatch: the live system doing
    something the research never tested.

    So the live universe is now pinned to the same instrument type as the
    research: symbols present in the point-in-time common-stock panel, which
    was built from Nasdaq symbol directories with ETFs and funds excluded.
    """
    h = _headers()
    r = requests.get(f"{PAPER_BASE}/assets", headers=h,
                     params={"asset_class": "us_equity", "status": "active"},
                     timeout=90)
    r.raise_for_status()
    alpaca = {a["symbol"] for a in r.json()
              if a.get("tradable") and a.get("exchange") in ("NYSE", "NASDAQ")
              and a["symbol"].isalpha() and len(a["symbol"]) <= 4}

    memb = pd.read_parquet(HERE.parent / "data" / "interim" /
                           "universe_membership.parquet")
    stocks = set(memb.loc[memb.is_active, "symbol"])
    syms = sorted(alpaca & stocks)
    logger.info("universe: %d alpaca-tradable ∩ %d researched common stocks = %d",
                len(alpaca), len(stocks), len(syms))
    if len(syms) < 200:
        raise RuntimeError(
            f"only {len(syms)} symbols after intersection -- refusing to trade "
            "a universe that does not resemble the researched one")
    return syms


# -------------------------------------------------------------------- signal
def low_vol_targets(bars: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """
    The frozen specification: lowest trailing 63-day volatility among names
    above the ADV floor, equal weighted. Nothing here is fitted at runtime.
    """
    b = bars.sort_values(["symbol", "date"])
    rows = []
    for sym, g in b.groupby("symbol", sort=False):
        if len(g) < cfg.vol_lookback + 5:
            continue
        r = g["close"].pct_change(fill_method=None)
        adv = (g["close"] * g["volume"]).tail(21).mean()
        if not np.isfinite(adv) or adv < cfg.min_adv:
            continue
        vol = r.tail(cfg.vol_lookback).std() * np.sqrt(252)
        if not np.isfinite(vol) or vol <= 0:
            continue
        rows.append({"symbol": sym, "vol": vol, "adv": adv,
                     "price": g["close"].iloc[-1]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    picks = df[df["vol"] <= df["vol"].quantile(cfg.top_quantile)].copy()
    per = min(cfg.capital / len(picks), cfg.capital * cfg.max_position_pct)
    picks["target_dollars"] = per
    # MOC orders (see submit) do not accept fractional shares, so targets are
    # whole shares. A name whose price exceeds 1.5x the per-name allocation
    # would round to a >50% overweight or to zero; those are excluded and
    # counted -- a small, documented deviation from the ~equal-weight spec.
    picks["target_shares"] = (per / picks["price"]).round().astype(int)
    n_pricey = int((picks["price"] > 1.5 * per).sum())
    if n_pricey:
        logger.info("excluding %d names priced above 1.5x the $%.0f "
                    "per-name allocation", n_pricey, per)
    picks = picks[(picks.target_shares > 0) & (picks["price"] <= 1.5 * per)]
    return picks.reset_index(drop=True)


# ------------------------------------------------------------------ ordering
def submit(symbol: str, qty: int, side: str,
           tif: str = "cls") -> dict:
    """
    Rebalance orders are MARKET-ON-CLOSE by default (tif='cls').

    exp10c measured the closing-auction print against the panel's close:
    median difference 0.000bp on every symbol tested. The closing auction
    is the one execution venue where the live fill and the backtest fill
    are the same number, and the backtest's convention has always been
    close-price fills. The 09:40 market orders this runner used before
    paid a measured ~6.8bps to the touch; MOC replaces that with the
    auction cross. If the venue rejects cls (submission window closes
    ~15:50 ET), fall back to a day market order rather than not trading.
    """
    url = f"{PAPER_BASE}/orders"
    _assert_paper(url)
    body = {"symbol": symbol, "qty": str(qty), "side": side,
            "type": "market", "time_in_force": tif}
    r = requests.post(url, headers=_headers(), json=body, timeout=45)
    if not r.ok and tif == "cls":
        logger.warning("%s: cls rejected (%s) -- falling back to day order",
                       symbol, r.text[:80])
        body["time_in_force"] = "day"
        r = requests.post(url, headers=_headers(), json=body, timeout=45)
    if not r.ok:
        return {"symbol": symbol, "status": "rejected", "error": r.text[:200]}
    return r.json()


def positions(asset_class: str = "us_equity") -> pd.DataFrame:
    """
    Open positions, EQUITY ONLY by default. The account also carries the
    crypto TSMOM sleeve (crypto_paper.py); without this filter its fractional
    positions truncated to qty=0 here, and -- worse -- their mere presence
    made daily_run's `len(pos) == 0` initial-entry gate read as "already
    invested", which would have silently kept the equity book in cash until
    the next first-of-month. One strategy's positions must never gate
    another's.
    """
    r = requests.get(f"{PAPER_BASE}/positions", headers=_headers(), timeout=45)
    if not r.ok or not r.json():
        return pd.DataFrame(columns=["symbol", "qty", "avg_entry_price"])
    rows = [{"symbol": p["symbol"], "qty": int(float(p["qty"])),
             "avg_entry_price": float(p["avg_entry_price"]),
             "market_value": float(p["market_value"])}
            for p in r.json()
            if p.get("asset_class", "us_equity") == asset_class]
    return pd.DataFrame(rows, columns=["symbol", "qty", "avg_entry_price",
                                       "market_value"])


def market_open() -> bool:
    r = requests.get(f"{PAPER_BASE}/clock", headers=_headers(), timeout=30)
    return bool(r.ok and r.json().get("is_open"))


# ---------------------------------------------------------------- journaling
def _db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS intents(
        run_id TEXT, ts TEXT, symbol TEXT, side TEXT, qty INTEGER,
        ref_bid REAL, ref_ask REAL, ref_mid REAL, quoted_spread_bps REAL,
        modelled_cost_bps REAL, order_id TEXT, status TEXT, note TEXT)""")
    # nbbo_* columns are the consolidated-tape reference; ref_mid/*_iex_* are
    # the decision-time IEX view, kept only to measure how wrong it was.
    c.execute("""CREATE TABLE IF NOT EXISTS fills(
        run_id TEXT, order_id TEXT, ts TEXT, filled_at TEXT, symbol TEXT,
        side TEXT, qty INTEGER, fill_price REAL,
        nbbo_mid REAL, nbbo_spread_bps REAL, n_quotes INTEGER,
        ref_mid REAL, quoted_spread_bps REAL, modelled_cost_bps REAL,
        slip_vs_nbbo_bps REAL, slip_vs_iex_bps REAL, iex_ref_error_bps REAL,
        notional REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS equity(
        run_id TEXT, ts TEXT, equity REAL, cash REAL, positions INTEGER)""")
    return c


def modelled_cost_bps(adv: float) -> float:
    """The fitted measured-spread model: half-spread as the cost estimate."""
    return 0.5 * 10 ** (4.288 - 0.416 * np.log10(max(adv, 1e5)))


def run_once(cfg: RunConfig = RunConfig(), dry_run: bool = False) -> dict:
    """One rebalance. Records intent, submits, then reconciles fills."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    con = _db()

    if not market_open() and not dry_run:
        logger.warning("market closed -- orders would queue to the next open")

    # NOT truncated. An earlier version sliced the first N of a SORTED list,
    # which silently restricted the live universe to companies beginning with
    # A-H and made the traded cross-section different from the researched one.
    # Selection must come from the ADV filter, never from the alphabet.
    syms = tradable_universe(cfg)
    logger.info("fetching bars for %d symbols (full universe, no truncation)",
                len(syms))
    bars = get_bars(syms)
    if bars.empty:
        raise RuntimeError("no bars returned")
    logger.info("bars: %d rows, %d symbols", len(bars), bars.symbol.nunique())

    tgt = low_vol_targets(bars, cfg)
    if tgt.empty:
        raise RuntimeError("signal produced no targets")
    logger.info("targets: %d names, median vol %.1f%%, median ADV $%.0fM",
                len(tgt), tgt.vol.median() * 100, tgt.adv.median() / 1e6)

    is_open = market_open()
    q = get_quotes(tgt.symbol.tolist())
    if not is_open and len(q):
        # After hours IEX quotes are stale and frequently one-sided: the first
        # dry run saw a median quoted spread of 616bps on names whose true
        # spread is ~1bp. Recording them as a cost reference would be worse
        # than recording nothing.
        bad = q["spread_bps"].gt(200).sum()
        logger.warning("market CLOSED: %d/%d quotes show >200bps (stale/one-sided). "
                       "Reference prices from this run are not usable for "
                       "slippage measurement.", int(bad), len(q))
    tgt = tgt.merge(q, on="symbol", how="left")
    tgt["modelled_cost_bps"] = tgt["adv"].apply(modelled_cost_bps)

    # Persist the target book. THIS IS WHAT MAKES THE BOOK REPAIRABLE.
    # 2026-08-26: the 08-19 transition submitted 277 MOC orders and 224 of
    # them EXPIRED unfilled in the paper closing auction. Nothing noticed,
    # because the runner only reconciles on a rebalance day and the frozen
    # spec is quarterly -- so a half-built book would have been measured as
    # if it were the strategy until October. The target set is now stored so
    # repair_drift() can converge to it on later days WITHOUT recomputing the
    # signal (recomputing daily would silently turn this into a daily-
    # rebalanced strategy, which is a different strategy than was validated).
    con.execute("""CREATE TABLE IF NOT EXISTS targets(
        run_id TEXT, ts TEXT, symbol TEXT, target_shares INTEGER,
        PRIMARY KEY (run_id, symbol))""")
    if not dry_run:
        con.executemany(
            "INSERT OR REPLACE INTO targets(run_id, ts, symbol, target_shares) "
            "VALUES(?,?,?,?)",
            [(run_id, datetime.now(timezone.utc).isoformat(timespec="seconds"),
              r.symbol, int(r.target_shares)) for r in tgt.itertuples()])
        # a new target book starts with a clean escalation ledger, so a name
        # that was unfillable last quarter still gets its MOC attempts now
        con.execute("CREATE TABLE IF NOT EXISTS repair_attempts("
                    "symbol TEXT PRIMARY KEY, attempts INTEGER, last_ts TEXT)")
        con.execute("DELETE FROM repair_attempts")
        con.commit()

    cur = positions()
    held = dict(zip(cur.symbol, cur.qty)) if len(cur) else {}

    orders = []
    for _, row in tgt.iterrows():
        delta = int(row.target_shares) - held.get(row.symbol, 0)
        if delta == 0:
            continue
        side = "buy" if delta > 0 else "sell"
        note = ""
        if not np.isfinite(row.get("mid", np.nan)):
            note = "no two-sided quote at decision time"
        res = {"status": "dry_run", "id": ""} if dry_run else submit(row.symbol, abs(delta), side)
        orders.append({"run_id": run_id, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "symbol": row.symbol, "side": side, "qty": abs(delta),
                       "ref_bid": row.get("bid"), "ref_ask": row.get("ask"),
                       "ref_mid": row.get("mid"), "quoted_spread_bps": row.get("spread_bps"),
                       "modelled_cost_bps": row.modelled_cost_bps,
                       "order_id": str(res.get("id", "")), "status": res.get("status", "?"),
                       "note": note or res.get("error", "")})
    # exit anything no longer in the target set -- EXCEPT symbols owned by
    # another book in this shared account. The options runner holds 100 IWM
    # as covered-call collateral; without this reservation the low-vol exit
    # loop would liquidate another strategy's position (the same
    # cross-book interaction that nearly suppressed this book's initial
    # entry via the crypto positions on 2026-08-13).
    reserved = {"IWM"}
    for sym, qty in held.items():
        if sym not in set(tgt.symbol) and sym not in reserved and qty > 0:
            res = {"status": "dry_run", "id": ""} if dry_run else submit(sym, qty, "sell")
            orders.append({"run_id": run_id, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "symbol": sym, "side": "sell", "qty": qty,
                           "ref_bid": None, "ref_ask": None, "ref_mid": None,
                           "quoted_spread_bps": None, "modelled_cost_bps": None,
                           "order_id": str(res.get("id", "")), "status": res.get("status", "?"),
                           "note": "exit -- dropped from target set"})

    od = pd.DataFrame(orders)
    if len(od):
        od.to_sql("intents", con, if_exists="append", index=False)
    logger.info("submitted %d orders (%d rejected)", len(od),
                int((od.status == "rejected").sum()) if len(od) else 0)
    con.commit()
    return {"run_id": run_id, "n_targets": len(tgt), "n_orders": len(od),
            "dry_run": dry_run}


def latest_targets() -> pd.DataFrame:
    """The most recently stored target book (empty frame if none)."""
    con = _db()
    try:
        rid = con.execute("SELECT run_id FROM targets ORDER BY ts DESC "
                          "LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        return pd.DataFrame(columns=["symbol", "target_shares"])
    if not rid:
        return pd.DataFrame(columns=["symbol", "target_shares"])
    return pd.read_sql(
        "SELECT symbol, target_shares FROM targets WHERE run_id = ?",
        con, params=(rid[0],))


MAX_REPAIR_ATTEMPTS = 8        # stop pestering a name that will never fill
# Auction fills are preferred (exp10c: the closing cross is the one venue
# where the live fill equals the backtest's close-price convention), but the
# 2026-08-19 evidence is that this paper venue fills MOC poorly: of 281
# orders, 51 filled, 81 partially filled and 224 expired. Two MOC attempts
# harvest whatever auction liquidity exists, then day orders finish the book.
# An unbuilt book measures nothing at all, which is the worse error.
CLS_ATTEMPTS_BEFORE_DAY = 2


def repair_drift(dry_run: bool = False, min_notional: float = 25.0) -> dict:
    """
    Converge the live book to the LAST STORED TARGET SET.

    This is not a rebalance: it never recomputes the signal, so the traded
    cross-section stays the one chosen at the last scheduled rebalance. It
    only finishes the job that rebalance started.

    Why it exists: MOC orders in the paper venue fill partially or not at
    all (2026-08-19: 224 of 281 expired). A book that is 50% built measures
    neither the strategy nor its costs. Orders start as MOC to preserve
    exp10c's auction-provenance measurement, then escalate to day orders
    after CLS_ATTEMPTS_BEFORE_DAY sessions -- a fill with known worse
    provenance is more informative than a position that never exists.
    """
    con = _db()
    con.execute("""CREATE TABLE IF NOT EXISTS repair_attempts(
        symbol TEXT PRIMARY KEY, attempts INTEGER, last_ts TEXT)""")
    con.commit()

    tgt = latest_targets()
    if tgt.empty:
        return {"status": "no_targets", "n_orders": 0}

    cur = positions()
    held = dict(zip(cur.symbol, cur.qty)) if len(cur) else {}
    prior = {r[0]: r[1] for r in
             con.execute("SELECT symbol, attempts FROM repair_attempts").fetchall()}

    # a price for each name we might touch, to skip trivial dollar deltas
    syms = sorted(set(tgt.symbol) | set(held))
    px = {}
    if syms:
        try:
            q = get_quotes(syms)
            px = {r.symbol: r.mid for r in q.itertuples()
                  if np.isfinite(getattr(r, "mid", np.nan))}
        except Exception as exc:                           # noqa: BLE001
            logger.warning("repair: quote fetch failed (%s); "
                           "using position marks", str(exc)[:80])
    if len(cur):
        for r in cur.itertuples():
            px.setdefault(r.symbol, abs(r.market_value) / max(abs(r.qty), 1))

    reserved = {"IWM"}          # covered-call collateral, another book's asset
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "R"
    want = dict(zip(tgt.symbol, tgt.target_shares))

    deltas = []
    for sym, qty in want.items():
        d = int(qty) - held.get(sym, 0)
        if d != 0:
            deltas.append((sym, d))
    for sym, qty in held.items():
        if sym not in want and sym not in reserved and qty > 0:
            deltas.append((sym, -qty))

    orders, skipped, exhausted = [], 0, 0
    for sym, d in deltas:
        price = px.get(sym)
        if price and abs(d) * price < min_notional:
            skipped += 1
            continue
        n = prior.get(sym, 0)
        if n >= MAX_REPAIR_ATTEMPTS:
            exhausted += 1
            continue
        tif = "day" if n >= CLS_ATTEMPTS_BEFORE_DAY else "cls"
        side = "buy" if d > 0 else "sell"
        res = ({"status": "dry_run", "id": ""} if dry_run
               else submit(sym, abs(d), side, tif=tif))
        orders.append({
            "run_id": run_id,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "symbol": sym, "side": side, "qty": abs(d),
            "ref_bid": None, "ref_ask": None, "ref_mid": px.get(sym),
            "quoted_spread_bps": None, "modelled_cost_bps": None,
            "order_id": str(res.get("id", "")),
            "status": res.get("status", "?"),
            "note": f"repair attempt {n + 1} tif={tif}"})
        if not dry_run:
            con.execute(
                "INSERT INTO repair_attempts(symbol, attempts, last_ts) "
                "VALUES(?,?,?) ON CONFLICT(symbol) DO UPDATE SET "
                "attempts = attempts + 1, last_ts = excluded.last_ts",
                (sym, 1, datetime.now(timezone.utc).isoformat(timespec="seconds")))

    if orders and not dry_run:
        pd.DataFrame(orders).to_sql("intents", con, if_exists="append",
                                    index=False)
    con.commit()
    return {"run_id": run_id, "n_targets": len(want), "n_held": len(held),
            "n_orders": len(orders), "skipped_small": skipped,
            "exhausted": exhausted, "dry_run": dry_run}


def reconcile(run_id: str | None = None, min_age_hours: float = 1.0,
              use_sip: bool = True) -> pd.DataFrame:
    """
    THE MEASUREMENT: what did trading actually cost?

    Measuring that needs a reference price -- what the stock was "really" worth
    at the instant we traded. The gap between our fill and that reference is
    the cost.

    The first version used the wrong reference. It compared fills to the quote
    we saw at decision time, which comes from IEX: a single venue carrying
    roughly 2% of US volume. IEX's best bid and offer is frequently stale or a
    single odd lot, so it is not a fair picture of the market. On the first
    real trading day it reported a 3,079 bps spread for CNI and 1,377 bps for
    BMO -- both megacaps whose true spreads are 1-3 bps -- and priced BMO at
    $197.29 when it filled at $184.97. Every cost number that came out of it
    was noise.

    The fix is to use the consolidated tape (SIP), which aggregates all
    venues. Real-time SIP is a paid product, but HISTORICAL SIP is free, and
    reconciliation is inherently backward-looking -- so we simply wait a day
    and then ask what the whole market showed at the fill timestamp.
    `true_nbbo_at()` already did this and was never wired in.

    Two reference prices are now recorded per fill:
      slip_vs_nbbo_bps   fill vs the consolidated midpoint -- THE cost number
      slip_vs_iex_bps    fill vs what we could see when deciding -- kept
                         because the gap between the two measures how blind
                         the free data tier makes us at decision time

    `min_age_hours` holds back fills too recent for SIP history to cover. The
    free tier withholds only the last 15 minutes, verified against real fills
    at 7 hours old, so an hour is ample margin. Anything deferred is picked up
    by a later run rather than silently measured against nothing.
    """
    con = _db()
    intents = pd.read_sql("SELECT * FROM intents", con)
    if run_id:
        intents = intents[intents.run_id == run_id]

    # never re-measure a fill that is already recorded
    try:
        done = set(pd.read_sql("SELECT DISTINCT order_id FROM fills", con)
                   .order_id.dropna().tolist())
    except Exception:                                      # noqa: BLE001
        done = set()
    if done:
        intents = intents[~intents.order_id.isin(done)]

    ids = [i for i in intents.order_id.tolist() if i]
    if not ids:
        return pd.DataFrame()

    h = _headers()
    rows = []
    for oid in ids:
        r = requests.get(f"{PAPER_BASE}/orders/{oid}", headers=h, timeout=30)
        if not r.ok:
            continue
        o = r.json()
        if o.get("status") != "filled" or not o.get("filled_avg_price"):
            continue
        rows.append({"order_id": oid, "fill_price": float(o["filled_avg_price"]),
                     "filled_qty": int(float(o.get("filled_qty", 0))),
                     "filled_at": o.get("filled_at")})
        time.sleep(0.12)

    f = pd.DataFrame(rows)
    if f.empty:
        return f
    m = intents.merge(f, on="order_id", how="inner")

    # hold back fills that historical SIP does not cover yet
    now = pd.Timestamp.now(tz="UTC")
    age_h = (now - pd.to_datetime(m.filled_at, utc=True, format="mixed")
             ).dt.total_seconds() / 3600
    too_new = age_h < min_age_hours
    if too_new.any():
        logger.info("%d fill(s) younger than %.0fh -- deferred to a later run",
                    int(too_new.sum()), min_age_hours)
    m = m[~too_new]
    if m.empty:
        return m

    # --- the honest reference: consolidated NBBO at the fill timestamp ---
    nbbo = []
    for _, r in m.iterrows():
        q = true_nbbo_at(r.symbol, r.filled_at) if use_sip else {}
        nbbo.append({"nbbo_mid": q.get("nbbo_mid", np.nan),
                     "nbbo_spread_bps": q.get("nbbo_spread_bps", np.nan),
                     "n_quotes": q.get("n_quotes", 0)})
        time.sleep(0.12)
    m = pd.concat([m.reset_index(drop=True), pd.DataFrame(nbbo)], axis=1)

    sign = np.where(m.side == "buy", 1.0, -1.0)
    m["slip_vs_nbbo_bps"] = np.where(
        m.nbbo_mid > 0,
        sign * (m.fill_price - m.nbbo_mid) / m.nbbo_mid * 1e4, np.nan)
    m["slip_vs_iex_bps"] = np.where(
        m.ref_mid > 0,
        sign * (m.fill_price - m.ref_mid) / m.ref_mid * 1e4, np.nan)
    # how misleading the decision-time view was, per fill
    m["iex_ref_error_bps"] = np.where(
        (m.nbbo_mid > 0) & (m.ref_mid > 0),
        (m.ref_mid - m.nbbo_mid) / m.nbbo_mid * 1e4, np.nan)
    m["notional"] = m.fill_price * m.filled_qty

    got = int((m.n_quotes > 0).sum())
    logger.info("reconciled %d fill(s); NBBO recovered for %d (%.0f%%)",
                len(m), got, got / max(len(m), 1) * 100)

    out = m[["run_id", "order_id", "ts", "filled_at", "symbol", "side",
             "filled_qty", "fill_price", "nbbo_mid", "nbbo_spread_bps",
             "n_quotes", "ref_mid", "quoted_spread_bps", "modelled_cost_bps",
             "slip_vs_nbbo_bps", "slip_vs_iex_bps", "iex_ref_error_bps",
             "notional"]].rename(columns={"filled_qty": "qty"})
    out.to_sql("fills", con, if_exists="append", index=False)
    con.commit()
    return m


def report() -> pd.DataFrame:
    """
    What trading actually cost, versus what the backtest assumed.

    This is the whole point of paper trading. If real costs land near the
    model, backtested numbers mean something. If they land far above it,
    every candidate this project ever scored was scored too generously.

    Medians, not means: a single bad print would otherwise dominate.
    """
    con = _db()
    f = pd.read_sql("SELECT * FROM fills", con)
    if f.empty or "slip_vs_nbbo_bps" not in f.columns:
        return pd.DataFrame()
    g = f[f.n_quotes > 0]
    if g.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "n_fills": [len(g)],
        "realised_cost_bps_median": [g.slip_vs_nbbo_bps.median()],
        "realised_cost_bps_mean": [g.slip_vs_nbbo_bps.mean()],
        "modelled_cost_bps_median": [g.modelled_cost_bps.median()],
        "nbbo_half_spread_bps_median": [(g.nbbo_spread_bps / 2).median()],
        "iex_ref_error_bps_median": [g.iex_ref_error_bps.abs().median()],
        "notional_traded": [g.notional.sum()],
    })


# ---------------------------------------------------------------------- CLI
def status() -> dict:
    """Account + open positions + what has been measured so far."""
    h = _headers()
    a = requests.get(f"{PAPER_BASE}/account", headers=h, timeout=30).json()
    pos = positions()
    con = _db()
    fills = pd.read_sql("SELECT * FROM fills", con)
    return {
        "equity": float(a.get("equity", 0)),
        "cash": float(a.get("cash", 0)),
        "n_positions": len(pos),
        "market_open": market_open(),
        "fills_measured": len(fills),
        "mean_realised_slip_bps": float(fills.realised_slip_bps.mean())
                                   if len(fills) else None,
        "mean_modelled_cost_bps": float(fills.modelled_cost_bps.mean())
                                   if len(fills) else None,
    }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description="paper-trading runner (PAPER ONLY)")
    ap.add_argument("cmd", choices=["rebalance", "dryrun", "reconcile",
                                    "status", "report"])
    ap.add_argument("--force-closed", action="store_true",
                    help="rebalance even though the market is shut (orders queue)")
    a = ap.parse_args()

    if a.cmd == "status":
        print(json.dumps(status(), indent=2))
    elif a.cmd == "dryrun":
        print(json.dumps(run_once(dry_run=True), indent=2))
    elif a.cmd == "rebalance":
        if not market_open() and not a.force_closed:
            raise SystemExit(
                "market is closed. Quotes are stale and one-sided outside "
                "hours, so the reference prices this run records would be "
                "unusable for slippage measurement -- which is the entire "
                "point. Re-run during 09:30-16:00 ET, or pass --force-closed "
                "if you only want the orders queued.")
        print(json.dumps(run_once(dry_run=False), indent=2))
    elif a.cmd == "reconcile":
        m = reconcile()
        print(f"reconciled {len(m)} fills")
        if len(m):
            print(m[["symbol", "side", "fill_price", "ref_mid",
                     "realised_slip_bps", "modelled_cost_bps"]].to_string(index=False))
    elif a.cmd == "report":
        r = report()
        print(r if len(r) else "no fills measured yet")
