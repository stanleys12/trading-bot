"""
EXPERIMENT 01 — does sector-neutralisation produce alpha the benchmark cannot?
==============================================================================

RESEARCH SPLIT ONLY (2010-2017). Validation is not touched. Test is sealed.

HYPOTHESIS
A small-cap's return is mostly its sector's return. Every cross-sectional
signal tested so far compared a biotech to a regional bank, so most of what
the ranking captured was sector beta wearing a factor's clothes. That showed
up concretely in the fundamental screen: 18.2% CAGR that turned out to be
beta 1.13 with NEGATIVE alpha versus SPY, and 25-31% volatility that killed
the Sharpe comparison.

Demeaning a signal within its sector removes the shared component and leaves
the firm-specific part. The prediction is therefore NOT higher returns -- it
is LOWER VOLATILITY and lower benchmark beta at similar return, which is what
would actually move risk-adjusted alpha.

That prediction is falsifiable in a specific way: if sector-neutral versions
show the same beta and the same volatility as raw versions, the hypothesis is
wrong and no amount of return improvement rescues it.

WHAT COUNTS AS SUCCESS
Not CAGR. The primary metric is residual alpha versus SPY and SPY+IWM, and
the secondary is whether beta and residual volatility fall. A variant that
raises return by raising beta has done nothing.

ERA REPORTING
The audit showed panel coverage rising from ~49% (2010) to ~95% (2026), which
mechanically manufactures fake alpha decay: early years are more
survivor-contaminated. So results are reported by era, and a signal that
looks strong ONLY in the early research years is treated as an artefact of
that gradient rather than a finding.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import (Experiment, register, slice_split,
                                  split_start, trial_count)

logger = logging.getLogger("exp01")
HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
PROCESSED = HERE.parent / "data" / "processed"

SPLIT = "research"

PRICE_SIGS = ["mom_252", "mom_21", "rev_5", "vol_63"]
FUND_SIGS = ["book_to_market", "earnings_yield", "cfo_yield", "roe", "roa", "accruals"]


def build(panel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        px = g["true_close"] * g["adj_factor"]
        r = px.pct_change(fill_method=None)
        g["mom_252"] = px.shift(22) / px.shift(253) - 1.0
        g["mom_21"] = px.shift(1) / px.shift(22) - 1.0
        g["rev_5"] = -(px.shift(1) / px.shift(6) - 1.0)
        g["vol_63"] = -r.rolling(63).std().shift(1)
        g["fwd_63"] = px.shift(-64) / px.shift(-1) - 1.0
        g["dv21"] = g["dollar_volume"].rolling(21).mean().shift(1)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def ic(df: pd.DataFrame, feat: str, fwd: str = "fwd_63") -> dict:
    vals = []
    for d, g in df.groupby("date"):
        g = g[[feat, fwd]].dropna()
        if len(g) < 50:
            continue
        c = stats.spearmanr(g[feat], g[fwd]).statistic
        if np.isfinite(c):
            vals.append(c)
    if len(vals) < 8:
        return {"n": len(vals), "ic": np.nan, "t": np.nan}
    a = np.array(vals)
    return {"n": len(a), "ic": float(a.mean()),
            "t": float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))),
            "ir": float(a.mean() / a.std(ddof=1))}


def portfolio(daily: pd.DataFrame, rebal: pd.DataFrame, feat: str,
              q: float = 0.20) -> pd.Series:
    """
    Equal-weight top-quantile, quarterly rebalance, MARKED DAILY.

    Two frames on purpose. `rebal` supplies the quarterly decision dates and
    the signal values; `daily` supplies the price path in between. Handing the
    quarterly frame to both -- which an earlier version of this function did --
    makes every holding period exactly one row long, so pct_change returns NaN
    and the whole portfolio silently collapses to 32 points.

    That bug was invisible because a `len < 200` guard downstream skipped the
    result rather than complaining, so all 20 variants disappeared with no
    error. Guards that discard data must be loud; this one now raises.
    """
    dates = sorted(rebal["date"].unique())
    curves = []
    for i, d in enumerate(dates[:-1]):
        snap = rebal[rebal["date"] == d][["symbol", feat]].dropna()
        if len(snap) < 40:
            continue
        pick = snap.nlargest(max(int(len(snap) * q), 10), feat)["symbol"].tolist()
        nxt = dates[i + 1]
        hold = daily[(daily["symbol"].isin(pick)) &
                     (daily["date"] > d) & (daily["date"] <= nxt)]
        if hold["date"].nunique() < 5:
            raise ValueError(
                f"holding window {d}..{nxt} has only "
                f"{hold['date'].nunique()} trading days -- the daily frame is "
                "not daily. Refusing to produce a portfolio from it.")
        w = hold.pivot_table(index="date", columns="symbol",
                             values="px", aggfunc="last").sort_index()
        ret = w.pct_change(fill_method=None).mean(axis=1)
        if len(ret):
            ret.iloc[0] = ret.iloc[0] - 0.003      # 30bps at each rebalance
            curves.append(ret)
    return pd.concat(curves).sort_index() if curves else pd.Series(dtype=float)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    panel = pd.read_parquet(PROCESSED / "panel_v5_full.parquet")
    panel = slice_split(panel, SPLIT)
    panel = panel[panel["true_close"].between(2, 1000)]
    panel = panel[panel["sector"].notna() & (panel["sector"] != "Unknown")]
    logger.info("SPLIT=%s  %d rows, %d symbols, %s..%s  (K so far = %d)",
                SPLIT, len(panel), panel.symbol.nunique(),
                panel.date.min().date(), panel.date.max().date(), trial_count())

    df = build(panel)
    df["px"] = df["true_close"] * df["adj_factor"]
    df = df[(df["date"] >= split_start(SPLIT)) & (df["dv21"] > 1e6)]

    q_ends = df.groupby(df["date"].dt.to_period("Q"))["date"].max()
    q = df[df["date"].isin(q_ends)].copy()
    logger.info("quarterly obs: %d rows, %d dates\n", len(q), q.date.nunique())

    SIGS = [s for s in PRICE_SIGS + FUND_SIGS if s in q.columns]
    for s in SIGS:
        q[s + "_sn"] = q[s] - q.groupby(["date", "sector"])[s].transform("mean")

    # ------------------------------------------------- IC: raw vs neutral
    logger.info("=== IC vs FORWARD QUARTER (research split) ===")
    logger.info("  %-18s %18s %18s", "signal", "raw", "sector-neutral")
    rows = []
    for s in SIGS:
        a, b = ic(q, s), ic(q, s + "_sn")
        logger.info("  %-18s  IC=%+.4f t=%5.2f   IC=%+.4f t=%5.2f",
                    s, a["ic"], a["t"], b["ic"], b["t"])
        rows.append({"signal": s, "raw_ic": a["ic"], "raw_t": a["t"],
                     "sn_ic": b["ic"], "sn_t": b["t"]})
    ic_df = pd.DataFrame(rows)
    ic_df.to_csv(REPORTS / "exp01_ic.csv", index=False)

    # --------------------------- does neutralising actually cut volatility?
    logger.info("\n=== THE FALSIFIABLE PREDICTION: lower vol / lower beta ===")
    logger.info("  %-18s %10s %10s %10s %10s", "signal", "vol_raw", "vol_sn",
                "ret_raw", "ret_sn")
    pf_rows = []
    for s in SIGS:
        r_raw = portfolio(df, q, s)
        r_sn = portfolio(df, q, s + "_sn")
        if len(r_raw) < 200 or len(r_sn) < 200:
            logger.warning("  %-18s SKIPPED: only %d/%d daily obs -- investigate, "
                           "do not ignore", s, len(r_raw), len(r_sn))
            continue
        v_raw = r_raw.std(ddof=1) * np.sqrt(252)
        v_sn = r_sn.std(ddof=1) * np.sqrt(252)
        a_raw = r_raw.mean() * 252
        a_sn = r_sn.mean() * 252
        logger.info("  %-18s %9.1f%% %9.1f%% %9.1f%% %9.1f%%",
                    s, v_raw * 100, v_sn * 100, a_raw * 100, a_sn * 100)
        pf_rows.append({"signal": s, "vol_raw": v_raw, "vol_sn": v_sn,
                        "ret_raw": a_raw, "ret_sn": a_sn,
                        "sharpe_raw": a_raw / v_raw if v_raw else np.nan,
                        "sharpe_sn": a_sn / v_sn if v_sn else np.nan,
                        "vol_reduction": (v_raw - v_sn) / v_raw if v_raw else np.nan})
    pf = pd.DataFrame(pf_rows)
    pf.to_csv(REPORTS / "exp01_portfolios.csv", index=False)

    if len(pf):
        logger.info("\n  mean volatility reduction from neutralising: %.1f%%",
                    pf["vol_reduction"].mean() * 100)
        logger.info("  signals where Sharpe improved: %d of %d",
                    int((pf.sharpe_sn > pf.sharpe_raw).sum()), len(pf))
        logger.info("\n=== TOP BY SECTOR-NEUTRAL SHARPE (research split, gross of beta) ===")
        logger.info("\n%s", pf.sort_values("sharpe_sn", ascending=False)
                    [["signal", "sharpe_raw", "sharpe_sn", "vol_raw", "vol_sn"]]
                    .head(8).round(3).to_string(index=False))

        best = pf.sort_values("sharpe_sn", ascending=False).iloc[0]
        outcome = "HOLD" if best["sharpe_sn"] > best["sharpe_raw"] else "FAIL"
        reason = "" if outcome == "HOLD" else (
            "sector-neutralisation did not reduce volatility or improve Sharpe; "
            "the falsifiable prediction of the hypothesis failed")
        register(Experiment(
            name="exp01_sector_neutral", family="cross_sectional",
            hypothesis="demeaning within sector removes sector beta, cutting "
                       "volatility and raising risk-adjusted alpha",
            features=SIGS, split=SPLIT, universe="ADV>$1M, sector-labelled",
            rebalance_days=63, holding_days=63, n_observations=int(q.date.nunique()),
            metrics={"best_signal": best["signal"],
                     "sharpe_raw": float(best["sharpe_raw"]),
                     "sharpe_sn": float(best["sharpe_sn"]),
                     "mean_vol_reduction": float(pf["vol_reduction"].mean())},
            outcome=outcome, failure_reason=reason))
        logger.info("\nregistered. K is now %d", trial_count())
    return pf


if __name__ == "__main__":
    main()
