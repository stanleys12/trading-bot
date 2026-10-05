"""
Low-turnover fundamental screen.
================================

THE STRUCTURAL ARGUMENT

Everything in this project has died to one fraction:

        gross alpha
    ---------------------
     turnover x cost

We have attacked the numerator eleven different ways and failed. The 5-day
reversal effect was genuinely real -- t=6.7, survived a bid-ask-bounce test,
stronger in tradeable open-to-open returns -- and still lost 60%/yr, because
it turned the book over 85 times a year at ~120bps a trade.

Fundamental factors attack the DENOMINATOR instead. Book value and earnings
change quarterly, not daily. An annually rebalanced portfolio turns over about
once, so the same 120bps cost becomes a ~1.2%/yr drag rather than ~100%.
That is roughly a 70x improvement in the thing that has killed everything, and
it is the only structural fix available on free data.

For reference, the historical small-cap value premium is on the order of
4-6%/yr. Against a 1.2% drag that is a fight worth having. Against a 100%
drag nothing is.

THE RISK THAT MATTERS MOST HERE

Value strategies buy cheap stocks. Cheap stocks are distressed stocks.
Distressed stocks are the ones that delist -- and 37% of our point-in-time
universe has no recoverable price history, disproportionately the bankruptcies.

So of every strategy family we could test, VALUE IS THE MOST SURVIVORSHIP-
INFLATED. A book-to-market decile portfolio on this panel is measuring
"cheap companies that survived", which is a different and much better
business than "cheap companies".

We therefore report every value result twice: raw, and with the distress
filters we built earlier (reverse-split exclusion, price-cap, range guard).
The gap between them is a lower bound on the bias, not a correction for it.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.backtesting.engine import BacktestConfig, prepare_panel, run_backtest
from qr.backtesting.metrics import summarize

logger = logging.getLogger("fund")
HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
PROCESSED = HERE.parent / "data" / "processed"

VALUE = ["book_to_market", "earnings_yield", "sales_yield", "cfo_yield"]
QUALITY = ["roe", "roa", "gross_profitability"]
NEGATIVE = ["accruals", "leverage"]        # lower is better
ALL_F = VALUE + QUALITY + NEGATIVE


def zs(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / sd).clip(-3, 3)


def ic_table(df: pd.DataFrame, feats: list[str], fwd: str,
              overlap: float = 1.0) -> pd.DataFrame:
    rows = []
    for f in feats:
        vals = []
        for d, g in df.groupby("date"):
            g = g[[f, fwd]].dropna()
            if len(g) < 50:
                continue
            c = stats.spearmanr(g[f], g[fwd]).statistic
            if np.isfinite(c):
                vals.append(c)
        if len(vals) < 8:
            rows.append({"factor": f, "n": len(vals), "ic": np.nan, "t_adj": np.nan})
            continue
        a = np.array(vals)
        # OVERLAP CORRECTION. Sampling 1-year forward returns every quarter
        # means consecutive observations share 3 of their 4 quarters, so the
        # naive t-stat overstates significance by roughly sqrt(overlap).
        # 28 quarterly obs of an annual return is ~7 independent years.
        naive_t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))
        eff_n = max(len(a) / overlap, 2.0)
        rows.append({"factor": f, "n": len(a), "eff_n": round(eff_n, 1),
                     "ic": float(a.mean()), "t_naive": float(naive_t),
                     "t_adj": float(a.mean() / (a.std(ddof=1) / np.sqrt(eff_n))),
                     "ir": float(a.mean() / a.std(ddof=1))})
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    REPORTS.mkdir(exist_ok=True)

    p = pd.read_parquet(PROCESSED / "panel_v4_fundamentals.parquet")
    p = p[p["true_close"].between(2, 1000)]
    logger.info("panel: %d rows, %d symbols, %s..%s", len(p), p.symbol.nunique(),
                p.date.min().date(), p.date.max().date())

    # forward returns at LOW frequency -- this is the whole point
    out = []
    for sym, g in p.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        px = g["true_close"] * g["adj_factor"]
        g["fwd_63"] = px.shift(-64) / px.shift(-1) - 1.0     # quarter
        g["fwd_252"] = px.shift(-253) / px.shift(-1) - 1.0   # year
        g["dv21"] = g["dollar_volume"].rolling(21).mean().shift(1)
        out.append(g)
    df = pd.concat(out, ignore_index=True)

    d = np.sort(df["date"].unique())
    d_test = pd.Timestamp(d[int(len(d) * 0.8)])
    dv = df[(df["date"] < d_test) & (df["dv21"] > 1e6)].copy()

    # quarterly sampling: one observation per name per quarter
    q_ends = dv.groupby(dv["date"].dt.to_period("Q"))["date"].max()
    q = dv[dv["date"].isin(q_ends)].copy()
    logger.info("quarterly sample: %d rows, %d dates, %d symbols",
                len(q), q.date.nunique(), q.symbol.nunique())
    for f in ALL_F:
        logger.info("  coverage %-22s %.0f%%", f, q[f].notna().mean() * 100)

    # ------------------------------------------------ information coefficients
    logger.info("\n=== IC vs FORWARD 1-YEAR RETURN (quarterly obs) ===")
    t1 = ic_table(q, ALL_F, "fwd_252", overlap=4.0)
    logger.info("\n%s", t1.round(4).to_string(index=False))

    logger.info("\n=== IC vs FORWARD 1-QUARTER RETURN ===")
    t2 = ic_table(q, ALL_F, "fwd_63", overlap=1.0)
    logger.info("\n%s", t2.round(4).to_string(index=False))
    pd.concat([t1.assign(horizon="1y"), t2.assign(horizon="1q")]).to_csv(
        REPORTS / "fundamental_ic.csv", index=False)

    # ------------------------------------------------------------ backtests
    logger.info("\n=== LOW-TURNOVER BACKTESTS (long only, real costs) ===")
    base = BacktestConfig(
        holding_days=252, rebalance_days=252, lookback_days=300,
        long_only=True, long_quantile=0.20, min_price=2.0,
        min_adv_dollar=2e6, min_names_per_side=20, capital=100_000.0,
        price_cap=1000.0, max_price_range=50.0)

    # panel_v4 kept only the columns needed for the as-of join; the engine
    # needs full OHLC for the spread estimator. Re-attach it.
    ohlc = pd.read_parquet(PROCESSED / "panel_v2_actions.parquet")[
        ["symbol", "date", "open", "high", "low", "close", "volume",
         "rsplit_252d", "cum_rsplit_factor"]]
    src = p.merge(ohlc, on=["symbol", "date"], how="left", suffixes=("", "_o"))
    src["close"] = src["true_close"]
    src = src.dropna(subset=["high", "low", "open"])
    prep = prepare_panel(src, base)
    prep_dv = prep[prep["date"] < d_test]

    def scorer(cols, weights=None, exclude_distress=False):
        def f(vis):
            last = vis[vis["date"] == vis["date"].max()].set_index("symbol")
            parts = []
            for c in cols:
                if c not in last.columns:
                    continue
                s = last[c]
                if c in NEGATIVE:
                    s = -s
                parts.append(zs(s))
            if not parts:
                return pd.Series(dtype=float)
            score = pd.concat(parts, axis=1).mean(axis=1)
            if exclude_distress and "rsplit_252d" in last.columns:
                bad = set(last.index[last["rsplit_252d"].fillna(0) > 0])
                score = score.loc[[i for i in score.index if i not in bad]]
            return score.dropna()
        return f

    rows = []
    combos = [
        ("value (4 factors)", VALUE, False),
        ("value, distress-filtered", VALUE, True),
        ("quality (3 factors)", QUALITY, False),
        ("value + quality", VALUE + QUALITY, False),
        ("value + quality, filtered", VALUE + QUALITY, True),
        ("book_to_market alone", ["book_to_market"], False),
        ("gross_profitability alone", ["gross_profitability"], False),
        ("low accruals alone", ["accruals"], False),
    ]
    for hold, hlabel in ((252, "annual"), (63, "quarterly")):
        cfg = replace(base, holding_days=hold, rebalance_days=hold)
        pr = prepare_panel(src, cfg)
        pr = pr[pr["date"] < d_test]
        for label, cols, filt in combos:
            try:
                r = run_backtest(pr, scorer(cols, exclude_distress=filt), cfg)
                m = summarize(r.daily, cfg.capital)
            except Exception as exc:                      # noqa: BLE001
                logger.info("  %-30s %-10s ERR %s", label, hlabel, str(exc)[:40])
                continue
            if m.get("n_periods", 0) < 4:
                continue
            rows.append({"strategy": label, "rebalance": hlabel,
                         **{k: m.get(k) for k in
                            ("n_periods", "sharpe", "cagr", "max_drawdown",
                             "win_rate", "avg_cost_bps", "turnover_annual_x",
                             "cost_drag_annual", "avg_n_long")}})
            logger.info("  %-30s %-10s SR=%6.2f CAGR=%+7.1f%% DD=%6.1f%% "
                        "turn=%4.1fx cost_drag=%.1f%%",
                        label, hlabel, m["sharpe"], m["cagr"] * 100,
                        m["max_drawdown"] * 100, m["turnover_annual_x"],
                        m["cost_drag_annual"] * 100)

    res = pd.DataFrame(rows)
    if len(res):
        res.to_csv(REPORTS / "fundamental_backtests.csv", index=False)
        logger.info("\n=== BEATS SPY (SR 0.82, CAGR 14.8%%)? ===")
        beat = res[(res.sharpe > 0.82) & (res.cagr > 0.148)]
        logger.info("  %d of %d", len(beat), len(res))
        if len(beat):
            logger.info("\n%s", beat.round(3).to_string(index=False))
        logger.info("\n=== TOP 5 BY SHARPE ===")
        logger.info("\n%s", res.sort_values("sharpe", ascending=False)
                    .head(5).round(3).to_string(index=False))
    return res


if __name__ == "__main__":
    main()
