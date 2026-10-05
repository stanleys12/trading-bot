"""
exp17: 5-day reversal harvested as a LIQUIDITY PROVIDER — pre-registration.
===========================================================================

STATUS: DESIGN FROZEN BEFORE ANY TAPE WAS FETCHED. The scoring code will be
added beneath this docstring; the rules below may not change after the first
quote is downloaded. Changing them means exp17b and another unit of K.

WHY THIS EXISTS
    The strongest confirmed effect in 224 trials: 5-day reversal in the
    $1-10M ADV band, IC t = 6.7, gross ~25%/yr — killed by ~60%/yr of
    TAKER cost at 85x turnover. The premium is compensation for providing
    liquidity to distressed sellers; every prior test paid the toll it was
    supposed to collect. This experiment flips the side: rest limit orders,
    collect the spread, and let the fill simulation charge the adverse
    selection that makers actually suffer.

THE TWO TRAPS, HANDLED IN DESIGN

  1. FILL REALISM. Passive fills are adversely selected: you are filled
     precisely when price moves through you. Rules (all conservative):
       - entry: limit buy resting at the PRIOR CLOSE minus half the
         prior-day quoted spread, GTC for day D+1 only
       - a fill requires a SIP print STRICTLY BELOW the limit (trade
         through the level, queue-last assumption); touches do not fill
       - unfilled entries lapse — no chasing, no taker fallback on entry
       - exit: after 5 sessions, limit sell at prevailing ask at 10:00;
         if unfilled by close of session 7, exit MARKET (pay full spread
         + impact) — unfilled passive exits are the adverse names, and
         they pay taker by construction
  2. SURVIVORSHIP (E2: buying losers is the most inflated trade on this
     panel). Any position whose symbol departs the point-in-time universe
     during the holding window with no exit price is charged **-55%**
     (Shumway-Warther performance-delisting estimate), pre-registered.

SAMPLE AND POWER (fixed before data)
    Validation window 2018-2022, weekly rebalance, bottom decile of 5-day
    return within the $1-10M ADV band, max 40 names resting per week.
    Tape cost forces sampling: **100 of the ~260 weeks, chosen by
    numpy seed 17**, fetched and scored. If the gross effect carries at
    its measured size (~0.5%/wk at ~2%/wk portfolio sd), t over 100 weeks
    ~ 2.5; the null passes t >= 2 ~2.5% of the time. The criterion can
    fail and can pass: powered. Research-window (2010-2017) tape does not
    exist on Alpaca (SIP from 2016), so this experiment scores VALIDATION
    ONLY and therefore demands the higher bar below.

PRE-REGISTERED CRITERIA — all required
    C1  net weekly mean > 0 with t >= 2.5 (clustered by week; single
        window compensates the missing research split)
    C2  fill rate >= 25% of resting orders (below that, the strategy
        cannot deploy capital and the result is a curiosity)
    C3  net result survives doubling the adverse-selection charge
        (fills re-priced as if executed at the day's low following the
        trade-through) with mean still > 0
    C4  delisting-haircut scenario (-55% on unpriced departures) already
        applied in C1 — reported separately so its weight is visible
    Any survival earns HOLD -> live paper limit orders, reconciled against
    SIP at T+2 (infrastructure already exists). Never PASS from simulation.

CAPACITY, STATED HONESTLY
    ~40 names x ~$2.5k resting = ~$100k deployed, in names doing $1-10M/day.
    This is a micro-capacity strategy by construction; that is the point.
    It does not scale, and nothing about this test claims it does.
"""
# ---------------------------------------------------------------------------
# IMPLEMENTATION NOTES — added 2026-08-14, BEFORE any scoring ran. Each is a
# resolution of an ambiguity or defect in the frozen contract, and each is
# resolved AGAINST the strategy or neutrally, never in its favour:
#
# 1. C3 DRAFTING DEFECT. The frozen parenthetical says fills are "re-priced
#    as if executed at the day's low" — but re-pricing a BUY at the low is
#    FAVOURABLE, which contradicts C3's stated purpose (a harder bar). The
#    conservative operationalization used: stressed return = base return
#    minus (limit - day_low)/limit, i.e. the full continuation below our
#    fill is charged AGAIN as additional adverse selection. Strictly harder
#    than any favourable reading.
# 2. EXIT SNAPSHOT TIME. The quote fetch took the exit-day ask at 15:00Z,
#    which is 10:00 ET in winter but 11:00 ET in summer (DST drafting slip).
#    No lookahead either way: the fill window opens at 15:02Z, strictly
#    after the snapshot. The label "10:00" is approximate; the mechanics
#    are clean.
# 3. SPLIT GUARD. Candidates whose adj_factor changes >10% between signal
#    and final date are excluded (raw-price fill simulation is meaningless
#    across a split). Count reported.
# 4. DIVIDENDS IGNORED. Ex-dates during the hold drag raw returns down and
#    the long side never receives the credit here — a small charge against
#    the strategy.
# 5. Weeks with zero fills are excluded from C1's weekly series (deployment
#    is C2's job); the count is reported.
# ---------------------------------------------------------------------------
from __future__ import annotations  # noqa: E402

import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register  # noqa: E402

logger = logging.getLogger("exp17")

INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

T_REQUIRED = 2.5
MIN_FILL_RATE = 0.25
DELIST_HAIRCUT = -0.55
MARKET_EXIT_SPREAD_MULT = 1.5     # full spread + half-spread impact


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cand = pd.read_parquet(INTERIM / "exp17_candidates.parquet")
    quotes = pd.read_parquet(INTERIM / "exp17_quotes.parquet")
    # the resumed fetch re-downloaded already-cached quotes (its done-set
    # compared two different datetime string formats), appending duplicates;
    # values are identical historical data, so keep-last is safe
    quotes = quotes.drop_duplicates(subset=["symbol", "signal_date"], keep="last")
    cand = cand.merge(quotes, on=["symbol", "signal_date"], how="left")

    # DEFECT FOUND 2026-08-14, first scoring run: panel `close`/`low` are
    # Yahoo split-ADJUSTED prices (this project's §6 defect, third
    # occurrence) while exit fills use TRUE SIP quotes. Names with later
    # reverse splits — concentrated in exactly this crashed-microcap
    # cohort — carried entry prices inflated by the back-propagated split
    # factor against true-priced exits: manufactured −90% losses. The
    # first run's mean of −29.4%/position with mean << median was the
    # incoherence that exposed it. Corrected: every price is converted to
    # the TRUE traded basis via true_close/close, and candidates'
    # prior_close is rebuilt from true_close at the signal date.
    syms = cand.symbol.unique().tolist()
    panel = pd.read_parquet(
        PROCESSED / "panel_v5_full.parquet",
        columns=["symbol", "date", "low", "close", "true_close", "adj_factor"])
    panel = panel[panel.symbol.isin(syms)]
    panel["date"] = pd.to_datetime(panel["date"])
    ratio = panel["true_close"] / panel["close"]
    panel["low"] = panel["low"] * ratio
    panel["close"] = panel["true_close"]
    true_at = panel.set_index(["symbol", "date"])["close"]
    key = pd.MultiIndex.from_frame(cand[["symbol", "signal_date"]])
    cand["prior_close"] = true_at.reindex(key).values

    minute = pd.read_parquet(INTERIM / "exp17_minute.parquet")
    minute["ts"] = pd.to_datetime(minute["t"], utc=True)
    by_sym = {s: g.sort_values("ts") for s, g in minute.groupby("symbol")}
    return cand, panel, by_sym


def score_positions(cand: pd.DataFrame, panel: pd.DataFrame,
                    minute: dict) -> pd.DataFrame:
    pl = {(s, d): (lo, cl, af) for s, d, lo, cl, af in
          panel[["symbol", "date", "low", "close", "adj_factor"]].itertuples(index=False)}
    last_date = panel.groupby("symbol")["date"].max().to_dict()

    rows = []
    n_no_spread = n_split = 0
    for r in cand.itertuples(index=False):
        if not np.isfinite(r.spread_d):
            n_no_spread += 1
            continue
        af_sig = pl.get((r.symbol, r.signal_date), (np.nan,) * 3)[2]
        af_fin = pl.get((r.symbol, r.final_date), (np.nan,) * 3)[2]
        if (np.isfinite(af_sig) and np.isfinite(af_fin)
                and abs(af_fin / af_sig - 1) > 0.10):
            n_split += 1
            continue

        if not np.isfinite(r.prior_close) or r.prior_close < 1.0:
            n_no_spread += 1          # true price unknown or sub-$1: skip
            continue
        limit = r.prior_close * (1 - r.spread_d / 2)
        entry = pl.get((r.symbol, r.entry_date))
        low_e = entry[0] if entry else np.nan
        filled = bool(np.isfinite(low_e) and low_e < limit)
        if not filled:
            rows.append({"week": r.week, "filled": False, "ret": np.nan,
                         "ret_stress": np.nan, "exit_kind": "none"})
            continue

        # --- exit ---
        delisted = (last_date.get(r.symbol, pd.Timestamp.min) < r.exit_date)
        if delisted:
            ret = ret_stress = DELIST_HAIRCUT
            kind = "delisted"
        else:
            exit_px, kind = np.nan, "market"
            m = minute.get(r.symbol)
            ask = r.ask_1000_exit
            # sanity guard: an exit quote outside [0.2x, 5x] of the true
            # prior close is a bad print or basis mismatch -> market exit
            if np.isfinite(ask) and not (0.2 * r.prior_close < ask
                                         < 5.0 * r.prior_close):
                ask = np.nan
            if m is not None and np.isfinite(ask):
                w_lo = pd.Timestamp(r.exit_date, tz="UTC") + pd.Timedelta(hours=15, minutes=2)
                w_hi = pd.Timestamp(r.final_date, tz="UTC") + pd.Timedelta(hours=21)
                win = m[(m.ts >= w_lo) & (m.ts <= w_hi)]
                if len(win) and win["high"].max() > ask:
                    exit_px, kind = ask, "passive"
            if not np.isfinite(exit_px):
                fin = pl.get((r.symbol, r.final_date))
                if fin is None or not np.isfinite(fin[1]):
                    ret = ret_stress = DELIST_HAIRCUT
                    rows.append({"week": r.week, "filled": True, "ret": ret,
                                 "ret_stress": ret_stress, "exit_kind": "delisted"})
                    continue
                exit_px = fin[1] * (1 - MARKET_EXIT_SPREAD_MULT * r.spread_d)
                kind = "market"
            ret = exit_px / limit - 1
            ret_stress = ret - (limit - low_e) / limit
        rows.append({"week": r.week, "filled": True, "ret": ret,
                     "ret_stress": ret_stress, "exit_kind": kind})
    logger.info("skipped: %d no-spread, %d split-guard", n_no_spread, n_split)
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cand, panel, minute = load()
    pos = score_positions(cand, panel, minute)

    resting = len(pos)
    fills = pos[pos.filled]
    fill_rate = len(fills) / max(resting, 1)
    wk = fills.groupby("week").agg(ret=("ret", "mean"),
                                   ret_stress=("ret_stress", "mean"),
                                   n=("ret", "size"))
    n_weeks = len(wk)
    t = wk.ret.mean() / (wk.ret.std() / np.sqrt(n_weeks)) if n_weeks > 2 else np.nan
    t_stress = (wk.ret_stress.mean() / (wk.ret_stress.std() / np.sqrt(n_weeks))
                if n_weeks > 2 else np.nan)

    print(f"\nresting orders scored: {resting} | fills: {len(fills)} "
          f"({fill_rate:.1%}) | weeks with fills: {n_weeks}")
    print(f"exit kinds: {fills.exit_kind.value_counts().to_dict()}")
    print(f"mean per-position net return: {fills.ret.mean()*100:+.2f}% "
          f"(median {fills.ret.median()*100:+.2f}%)")
    print(f"weekly mean net: {wk.ret.mean()*100:+.3f}%  t = {t:+.2f}")
    print(f"weekly mean stressed (C3): {wk.ret_stress.mean()*100:+.3f}%  t = {t_stress:+.2f}")

    c1 = bool(n_weeks > 2 and wk.ret.mean() > 0 and t >= T_REQUIRED)
    c2 = bool(fill_rate >= MIN_FILL_RATE)
    c3 = bool(n_weeks > 2 and wk.ret_stress.mean() > 0)
    ok = c1 and c2 and c3
    print(f"\nC1 weekly net mean>0, t>={T_REQUIRED}: {c1} (t={t:+.2f})")
    print(f"C2 fill rate >= {MIN_FILL_RATE:.0%}: {c2} ({fill_rate:.1%})")
    print(f"C3 stressed mean > 0: {c3} ({wk.ret_stress.mean()*100:+.3f}%/wk)")
    print(f"\n{'HOLD -- live paper limit-order probe next; never PASS from simulation' if ok else 'REJECTED'}")

    register(Experiment(
        name="exp17_reversal_maker",
        family="microstructure",
        hypothesis="the 5-day reversal premium in $1-10M ADV names is "
                   "harvestable as a LIQUIDITY PROVIDER: rest limit buys, "
                   "collect the spread the taker tests all paid",
        features=["r5_reversal", "trade_through_fills"],
        split="validation 2018-2022 only (SIP tape begins 2016); 100 weeks, seed 17",
        universe="$1-10M ADV, price>=$1, bottom decile 5d return",
        rebalance_days=5, holding_days=5,
        n_observations=int(len(fills)),
        metrics={"resting": resting, "fill_rate": round(fill_rate, 3),
                 "weeks": n_weeks,
                 "weekly_mean_pct": round(float(wk.ret.mean() * 100), 3),
                 "t": round(float(t), 2),
                 "weekly_stress_mean_pct": round(float(wk.ret_stress.mean() * 100), 3),
                 "t_stress": round(float(t_stress), 2),
                 "C1": c1, "C2": c2, "C3": c3},
        outcome="HOLD" if ok else "FAIL",
        failure_reason="" if ok else
        f"failed pre-registered criterion(s): C1={c1} C2={c2} C3={c3}",
    ))


if __name__ == "__main__":
    main()
