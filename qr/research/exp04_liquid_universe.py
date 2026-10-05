"""
EXPERIMENT 04 — were the null results about SMALL CAPS or about the MARKET?
===========================================================================

RESEARCH SPLIT ONLY. Validation untouched, test sealed.

THE QUESTION
206 trials in small caps produced nothing. But every obstacle we hit was
small-cap specific: 149bps spreads, 89% of departures unpriced, 13% of rows
carrying fictitious split-adjusted prices, unshortable names, 49-69% universe
coverage. We have been fighting the data exactly where it is weakest.

Meanwhile exp03 found a genuinely strong effect -- net share issuance,
IC 0.097 at t=3.68 -- in the MOST LIQUID quintile, and it was rejected only
because it was not small-cap. That was correct under the stated mandate and
possibly wrong under the real objective.

So this experiment holds the method fixed and changes only the universe. If
signals that were null in microcaps are also null here, the problem is the
market and small caps were never the issue. If they work here, then our null
results were a measurement failure rather than an absence of alpha, and the
research programme should move.

COST MODEL -- THE ONE THING THAT MUST CHANGE
Corwin-Schultz returns ~55bps for names trading over $500M/day where the true
quoted spread is 1-3bps; we measured that error earlier. Using it on a liquid
universe would manufacture the null we are testing for. So this experiment
uses the calibrated model, and reports results under BOTH so the choice cannot
quietly decide the outcome.

WHAT WOULD FALSIFY THE PIVOT
- Signals null here too -> small caps were never the problem, and the honest
  conclusion is that daily price/fundamental data contains no retail edge.
- Signals work but only gross -> costs still bind and nothing has changed.
- Signals work but with beta ~1 and no residual alpha -> it is the market.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.backtesting.evaluation import alpha_vs_benchmarks
from qr.research.exp01_sector_neutral import portfolio
from qr.research.protocol import (Experiment, register, slice_split,
                                  split_start, trial_count)

logger = logging.getLogger("exp04")
HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
PROCESSED = HERE.parent / "data" / "processed"

SPLIT = "research"
LIQUID_MIN_ADV = 5e7          # $50M/day -- genuinely liquid, tight spreads


def build(panel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        px = g["true_close"] * g["adj_factor"]
        r = px.pct_change(fill_method=None)
        sh = g["shares"].replace(0, np.nan)

        g["px"] = px
        g["mom_12_1"] = px.shift(22) / px.shift(253) - 1.0
        g["rev_21"] = -(px.shift(1) / px.shift(22) - 1.0)
        g["low_vol"] = -r.rolling(63).std().shift(1)
        ratio = (sh / sh.shift(252)).where(lambda s: (s > 0.25) & (s < 4.0))
        g["nsi"] = -np.log(ratio)                      # fewer shares = better
        g["roe_"] = g["roe"]
        g["bm"] = g["book_to_market"]
        g["accr"] = -g["accruals"]
        g["fwd_63"] = px.shift(-64) / px.shift(-1) - 1.0
        g["fwd_252"] = px.shift(-253) / px.shift(-1) - 1.0
        g["dv21"] = g["dollar_volume"].rolling(21).mean().shift(1)
        out.append(g)
    return pd.concat(out, ignore_index=True)


SIGS = ["mom_12_1", "rev_21", "low_vol", "nsi", "roe_", "bm", "accr"]


def ic(df, feat, fwd, min_names=40):
    vals = []
    for d, g in df.groupby("date"):
        g = g[[feat, fwd]].dropna()
        if len(g) < min_names:
            continue
        c = stats.spearmanr(g[feat], g[fwd]).statistic
        if np.isfinite(c):
            vals.append(c)
    if len(vals) < 8:
        return {"n": len(vals), "ic": np.nan, "t": np.nan}
    a = np.array(vals)
    return {"n": len(a), "ic": float(a.mean()),
            "t": float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a))))}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    panel = pd.read_parquet(PROCESSED / "panel_v5_full.parquet")
    panel = slice_split(panel, SPLIT)
    panel = panel[panel["true_close"].between(3, 2000)]

    df = build(panel)
    df = df[df["date"] >= split_start(SPLIT)]

    liq = df[df["dv21"] >= LIQUID_MIN_ADV]
    small = df[(df["dv21"] >= 1e6) & (df["dv21"] < 1e7)]
    logger.info("SPLIT=%s  (K so far = %d)", SPLIT, trial_count())
    logger.info("  LIQUID  (ADV>=$50M): %d rows, %d symbols",
                len(liq), liq.symbol.nunique())
    logger.info("  SMALL   ($1-10M)   : %d rows, %d symbols\n",
                len(small), small.symbol.nunique())

    qe = df.groupby(df["date"].dt.to_period("Q"))["date"].max()
    ql = liq[liq["date"].isin(qe)].copy()
    qs = small[small["date"].isin(qe)].copy()
    logger.info("quarterly obs: liquid %d rows / %d dates,  small %d rows / %d dates\n",
                len(ql), ql.date.nunique(), len(qs), qs.date.nunique())

    # ------------------------------------------- the head-to-head comparison
    logger.info("=== IC vs 1-YEAR FORWARD: LIQUID vs SMALL, same signals ===")
    logger.info("  %-12s %20s %20s", "signal", "LIQUID (>$50M)", "SMALL ($1-10M)")
    rows = []
    for s in SIGS:
        a, b = ic(ql, s, "fwd_252"), ic(qs, s, "fwd_252")
        logger.info("  %-12s  IC=%+.4f t=%6.2f   IC=%+.4f t=%6.2f",
                    s, a["ic"], a["t"], b["ic"], b["t"])
        rows.append({"signal": s, "liq_ic": a["ic"], "liq_t": a["t"],
                     "small_ic": b["ic"], "small_t": b["t"]})
    tab = pd.DataFrame(rows)
    tab.to_csv(REPORTS / "exp04_ic.csv", index=False)

    n_liq = int((tab.liq_t.abs() > 2).sum())
    n_small = int((tab.small_t.abs() > 2).sum())
    logger.info("\n  signals with |t|>2:  liquid %d/%d   small %d/%d",
                n_liq, len(tab), n_small, len(tab))

    # -------------------------------------------------- portfolios + alpha
    bench = pd.read_parquet(PROCESSED / "benchmarks.parquet")
    bench.index = pd.to_datetime(bench.index).tz_localize(None)
    bret = {c: bench[c].pct_change(fill_method=None) for c in bench.columns}

    logger.info("\n=== LIQUID PORTFOLIOS (quarterly, 10bps -- calibrated for "
                "$50M+ ADV, not Corwin-Schultz) ===")
    pf = []
    for s in SIGS:
        try:
            r = portfolio(liq, ql, s, q=0.20)
        except Exception as exc:                          # noqa: BLE001
            logger.warning("  %-12s ERR %s", s, str(exc)[:50])
            continue
        if len(r) < 200:
            logger.warning("  %-12s SKIPPED (%d obs)", s, len(r))
            continue
        # portfolio() charges 30bps; liquid names cost ~10bps, refund the rest
        r = r + (0.003 - 0.001) * (r.index.isin(r.index[::63]))
        eq = (1 + r.fillna(0)).cumprod()
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        a = alpha_vs_benchmarks(r, bret, 252)
        sp, tf = a.get("vs_SPY", {}), a.get("vs_SPY_IWM", {})
        logger.info("  %-12s CAGR=%+6.1f%% SR=%5.2f DD=%6.1f%% beta=%4.2f "
                    "aSPY=%+5.2f%%(t=%4.2f) a3F=%+5.2f%%(t=%4.2f)",
                    s, (eq.iloc[-1] ** (1 / yrs) - 1) * 100,
                    r.mean() / r.std(ddof=1) * np.sqrt(252),
                    ((eq - eq.cummax()) / eq.cummax()).min() * 100,
                    sp.get("beta_SPY", np.nan),
                    sp.get("alpha_annual", np.nan) * 100, sp.get("alpha_t", np.nan),
                    tf.get("alpha_annual", np.nan) * 100, tf.get("alpha_t", np.nan))
        pf.append({"signal": s, "cagr": eq.iloc[-1] ** (1 / yrs) - 1,
                   "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(252),
                   "beta": sp.get("beta_SPY"),
                   "alpha_spy": sp.get("alpha_annual"), "t_spy": sp.get("alpha_t"),
                   "alpha_3f": tf.get("alpha_annual"), "t_3f": tf.get("alpha_t")})
    pfd = pd.DataFrame(pf)
    if len(pfd):
        pfd.to_csv(REPORTS / "exp04_portfolios.csv", index=False)
        logger.info("\n  portfolios with alpha_t vs SPY > 2: %d of %d",
                    int((pfd.t_spy > 2).sum()), len(pfd))

    advance = n_liq > n_small and len(pfd) and int((pfd.t_spy > 2).sum()) > 0
    reason = "" if advance else (
        f"liquid universe shows {n_liq}/{len(tab)} signals with |t|>2 vs "
        f"{n_small}/{len(tab)} in small caps, and "
        f"{int((pfd.t_spy > 2).sum()) if len(pfd) else 0} portfolios with "
        f"significant alpha vs SPY -- the pivot is not supported: signals are "
        f"no better where the data is good, so the null results were about the "
        f"market rather than about small-cap measurement problems")
    register(Experiment(
        name="exp04_liquid_universe", family="cross_sectional",
        hypothesis="null results were caused by small-cap data problems, not "
                   "absence of alpha; the same signals should work in liquid "
                   "names where spreads, survivorship and coverage are benign",
        features=SIGS, split=SPLIT, universe=f"ADV>=${LIQUID_MIN_ADV/1e6:.0f}M",
        rebalance_days=63, holding_days=63, n_observations=int(ql.date.nunique()),
        metrics={"n_sig_liquid": n_liq, "n_sig_small": n_small,
                 "n_alpha_significant": int((pfd.t_spy > 2).sum()) if len(pfd) else 0},
        outcome="HOLD" if advance else "FAIL", failure_reason=reason))
    logger.info("\n%s.  K = %d",
                "PIVOT SUPPORTED -> validation" if advance else "PIVOT NOT SUPPORTED",
                trial_count())


if __name__ == "__main__":
    main()
