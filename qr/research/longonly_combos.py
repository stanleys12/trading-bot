"""
Long-only sweep and signal combinations.

TWO CHANGES FROM THE v1 SCREEN, both forced by what v1 found.

1. LONG ONLY. The rejected candidate got 77% of its profit from shorting
   microcaps at an assumed 5%/yr borrow. We have no borrow data and cannot get
   it free, so a long-short result is measuring an assumption. Long-only makes
   the borrow question disappear. It walks straight into the survivorship
   question instead -- 37% of the universe is missing and it is the losing
   37% -- so every number here is OPTIMISTICALLY BIASED by an unknown amount.
   Stated up front rather than buried.

2. COMBINATIONS. v1 tested 17 signals in isolation and never got to
   interactions, which the brief asked for. Cost is the binding constraint at
   ~120bps a trade, so the interesting combinations are the ones that trade
   LESS, not the ones that score higher: agreement filters that only act when
   several independent signals point the same way.

Benchmark to beat: SPY, CAGR +14.79%, Sharpe 0.82, MaxDD -33.7%.
A long-only equity book that cannot beat that is not worth running.
"""
from __future__ import annotations

import itertools
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.backtesting.engine import BacktestConfig, prepare_panel, run_backtest
from qr.backtesting.metrics import summarize
from qr.strategies.signals import REGISTRY

logger = logging.getLogger("lo")
HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
PROCESSED = HERE.parent / "data" / "processed"


def zscore(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / sd).clip(-3, 3)


def combo_sum(fns):
    """Average of z-scored components. Classic, and the baseline to beat."""
    def g(vis):
        parts = []
        for fn in fns:
            s = fn(vis)
            if s is not None and len(s):
                parts.append(zscore(s))
        if not parts:
            return pd.Series(dtype=float)
        return pd.concat(parts, axis=1).mean(axis=1)
    return g


def combo_agreement(fns, thresh=0.6):
    """
    Only score a name when the components AGREE.

    Hypothesis: with a 120bps cost floor, the win is not a better ranking, it
    is trading less often and only on high-conviction names. Names where the
    components disagree get zeroed out, so the book concentrates.
    """
    def g(vis):
        parts = []
        for fn in fns:
            s = fn(vis)
            if s is not None and len(s):
                parts.append(zscore(s))
        if len(parts) < 2:
            return pd.Series(dtype=float)
        m = pd.concat(parts, axis=1)
        mean = m.mean(axis=1)
        # fraction of components sharing the sign of the mean
        agree = (np.sign(m).eq(np.sign(mean), axis=0)).mean(axis=1)
        return mean.where(agree >= thresh, 0.0)
    return g


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    REPORTS.mkdir(exist_ok=True)
    panel = pd.read_parquet(PROCESSED / "panel_v1_frozen.parquet")

    base = BacktestConfig(
        holding_days=21, rebalance_days=21, lookback_days=280,
        long_only=True, long_quantile=0.03,
        min_price=2.0, min_adv_dollar=2e6, min_names_per_side=10,
        capital=100_000.0, price_cap=1000.0, max_price_range=50.0)
    prep = prepare_panel(panel, base)
    d = np.sort(prep["date"].unique())
    d_test = pd.Timestamp(d[int(len(d) * 0.8)])
    devval = prep[prep["date"] < d_test]        # TEST stays sealed

    rows = []

    # ---------------------------------------------- singles, long only
    logger.info("=== LONG-ONLY SINGLES ===")
    for name, (cat, fn) in REGISTRY.items():
        for q in (0.03, 0.10):
            cfg = replace(base, name=f"{name}|q{q}", long_quantile=q)
            try:
                m = summarize(run_backtest(devval, fn, cfg).daily, cfg.capital)
            except Exception as exc:                      # noqa: BLE001
                continue
            if m.get("n_periods", 0) < 8:
                continue
            rows.append({"kind": "single", "signal": name, "category": cat,
                         "quantile": q, **{k: m.get(k) for k in
                         ("n_periods", "sharpe", "sharpe_gross", "cagr",
                          "max_drawdown", "mean_period_ret", "win_rate",
                          "avg_cost_bps", "turnover_annual_x", "avg_n_long")}})
            logger.info("  %-20s q=%.2f  SR=%6.2f  CAGR=%+7.1f%%  DD=%6.1f%%",
                        name, q, m["sharpe"], m["cagr"] * 100,
                        m["max_drawdown"] * 100)

    # ------------------------------------------------- combinations
    # Pick the components by CATEGORY diversity, not by v1 rank, to avoid
    # compounding the selection already spent on that screen.
    pool = {
        "mom": REGISTRY["mom_12_1"][1],
        "lowvol": REGISTRY["low_vol"][1],
        "rs": REGISTRY["rs_vs_market"][1],
        "vol_shock": REGISTRY["volume_shock"][1],
        "revert": REGISTRY["reversal_1m"][1],
        "illiq": REGISTRY["illiquidity_amihud"][1],
    }
    logger.info("\n=== COMBINATIONS (long only) ===")
    for r in (2, 3):
        for keys in itertools.combinations(sorted(pool), r):
            fns = [pool[k] for k in keys]
            for style, maker in (("sum", combo_sum),
                                 ("agree", combo_agreement)):
                fn = maker(fns)
                cfg = replace(base, name=f"{'+'.join(keys)}|{style}")
                try:
                    m = summarize(run_backtest(devval, fn, cfg).daily, cfg.capital)
                except Exception:                          # noqa: BLE001
                    continue
                if m.get("n_periods", 0) < 8:
                    continue
                rows.append({"kind": f"combo_{style}", "signal": "+".join(keys),
                             "category": "combo", "quantile": base.long_quantile,
                             **{k: m.get(k) for k in
                             ("n_periods", "sharpe", "sharpe_gross", "cagr",
                              "max_drawdown", "mean_period_ret", "win_rate",
                              "avg_cost_bps", "turnover_annual_x", "avg_n_long")}})
                logger.info("  %-28s %-6s SR=%6.2f CAGR=%+7.1f%% DD=%6.1f%% turn=%.1fx",
                            "+".join(keys), style, m["sharpe"], m["cagr"] * 100,
                            m["max_drawdown"] * 100, m["turnover_annual_x"])

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("nothing scored")
        return
    df.to_csv(REPORTS / "longonly_combos.csv", index=False)

    SPY_SR, SPY_CAGR = 0.82, 0.1479
    logger.info("\n=== VS SPY (Sharpe 0.82, CAGR +14.8%%, DD -33.7%%) ===")
    beat = df[(df.sharpe > SPY_SR) & (df.cagr > SPY_CAGR)]
    logger.info("  beat SPY on BOTH Sharpe and CAGR: %d of %d", len(beat), len(df))
    if len(beat):
        logger.info("\n%s", beat.sort_values("sharpe", ascending=False)
                    [["kind", "signal", "quantile", "sharpe", "cagr",
                      "max_drawdown", "turnover_annual_x", "avg_n_long"]]
                    .head(15).round(3).to_string(index=False))
    logger.info("\n=== TOP 10 BY SHARPE ===")
    logger.info("\n%s", df.sort_values("sharpe", ascending=False)
                [["kind", "signal", "quantile", "sharpe", "sharpe_gross", "cagr",
                  "max_drawdown", "avg_cost_bps"]].head(10).round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
