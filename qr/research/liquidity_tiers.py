"""
EXPERIMENT: does any liquidity tier make a cross-sectional signal tradeable?

The v1 screen found gross Sharpe of at best ~1.0 against an estimated one-way
cost of ~120bps. That kills everything. Two possible explanations:

  (a) the signals are weak everywhere, or
  (b) we are trading the wrong part of the universe -- the $2M-ADV floor
      admits names whose spreads are enormous, and a higher floor would cut
      cost faster than it cuts signal.

(b) is the standard resolution in the literature: anomaly returns are
concentrated in illiquid names, and so are the costs, and the net is often
zero by construction. But it has to be MEASURED, not assumed, because if the
cost curve falls faster than the alpha curve there is a workable middle.

This script sweeps the liquidity floor and the portfolio concentration and
reports GROSS alpha, cost, and NET side by side for each tier. It also fixes
the canary threshold: a strongly NEGATIVE honest signal is not leakage, so the
check must be one-sided.
"""
from __future__ import annotations

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

logger = logging.getLogger("tiers")
HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
PROCESSED = HERE.parent / "data" / "processed"

# Signals with the best GROSS Sharpe from the v1 screen. We are now asking a
# different question (does cost structure ever permit trading?), so reusing
# them is legitimate -- but it does mean the tier sweep is exploratory and the
# TEST window stays untouched.
CANDIDATES = ["mom_12_1", "overnight_intraday", "volume_shock",
              "dist_from_ma", "bollinger_z", "low_vol"]

TIERS = [
    ("microcap",   2e6,   None),
    ("smallcap",   20e6,  None),
    ("midcap",     100e6, None),
    ("liquid",     500e6, None),
]
CONCENTRATIONS = [(0.10, "decile"), (0.03, "top3pct")]


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    panel = pd.read_parquet(PROCESSED / "prices.parquet")

    base = BacktestConfig(holding_days=21, rebalance_days=21, lookback_days=280,
                          min_price=2.0, min_names_per_side=8, capital=100_000.0)
    prep = prepare_panel(panel, base)
    d = np.sort(prep["date"].unique())
    d_test = pd.Timestamp(d[int(len(d) * 0.80)])
    devval = prep[prep["date"] < d_test]        # TEST stays untouched

    rows = []
    for tier_name, adv_floor, _ in TIERS:
        for q, qlabel in CONCENTRATIONS:
            for sig in CANDIDATES:
                cat, fn = REGISTRY[sig]
                cfg = replace(base, name=f"{sig}|{tier_name}|{qlabel}",
                              min_adv_dollar=adv_floor,
                              long_quantile=q, short_quantile=q)
                try:
                    res = run_backtest(devval, fn, cfg)
                    m = summarize(res.daily, cfg.capital)
                except Exception as exc:                  # noqa: BLE001
                    logger.warning("  %s %s %s -> %s", sig, tier_name, qlabel,
                                   str(exc)[:60])
                    continue
                if m.get("n_periods", 0) < 8:
                    continue
                rows.append({
                    "signal": sig, "tier": tier_name, "adv_floor": adv_floor,
                    "concentration": qlabel,
                    "n_periods": m["n_periods"],
                    "n_long": round(m["avg_n_long"], 1),
                    "spread_bps": round(m["avg_spread_bps"], 1),
                    "cost_bps": round(m["avg_cost_bps"], 1),
                    "gross_sharpe": round(m["sharpe_gross"], 3),
                    "net_sharpe": round(m["sharpe"], 3),
                    "gross_ret_pp": round(
                        (m["mean_period_ret"] + m["cost_drag_annual"] / 12) * 100, 3),
                    "net_ret_pp": round(m["mean_period_ret"] * 100, 3),
                    "maxdd": round(m["max_drawdown"], 3),
                    "turnover_x": round(m["turnover_annual_x"], 1),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("no tier produced a scorable result")
        return
    df.to_csv(REPORTS / "liquidity_tiers.csv", index=False)

    logger.info("\n=== COST BY LIQUIDITY TIER (the key question) ===")
    bytier = df.groupby("tier").agg(
        spread_bps=("spread_bps", "mean"), cost_bps=("cost_bps", "mean"),
        gross_sharpe=("gross_sharpe", "mean"), net_sharpe=("net_sharpe", "mean"),
        n_long=("n_long", "mean")).reindex([t[0] for t in TIERS])
    logger.info("\n%s", bytier.round(2).to_string())

    logger.info("\n=== BEST NET RESULTS ACROSS ALL TIERS ===")
    logger.info("\n%s", df.sort_values("net_sharpe", ascending=False)
                .head(15).to_string(index=False))

    logger.info("\n=== ANY POSITIVE NET SHARPE? ===")
    pos = df[df.net_sharpe > 0]
    logger.info("  %d of %d configurations have net Sharpe > 0", len(pos), len(df))
    if len(pos):
        logger.info("\n%s", pos.sort_values("net_sharpe", ascending=False)
                    .to_string(index=False))

    (REPORTS / "liquidity_tiers_summary.json").write_text(json.dumps({
        "n_configs": int(len(df)),
        "n_positive_net": int(len(pos)),
        "by_tier": bytier.round(3).to_dict(),
        "best": df.sort_values("net_sharpe", ascending=False).head(5).to_dict("records"),
    }, indent=2, default=str))
    return df


if __name__ == "__main__":
    main()
