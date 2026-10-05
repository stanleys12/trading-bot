"""
Systematic pre-registered signal screen.
========================================

PROTOCOL — fixed before any result is looked at.

  Split          DEV   : first 60% of dates   (explore, look freely)
                 VAL   : next  20%            (choose ONE candidate)
                 TEST  : final 20%            (scored ONCE, never re-run)

  Costs          charged on every trade, from the Corwin-Schultz spread
                 estimate plus square-root impact plus commission. No flat
                 assumption anywhere.

  Universe       point-in-time, from archived Nasdaq symbol directories.
                 Delisted names included where price history is recoverable;
                 the unrecoverable fraction is reported, not hidden.

  Multiple       K signals x 2 configurations are tested. Significance is
  testing        judged by Bonferroni-adjusted p AND by the Deflated Sharpe
                 Ratio with n_trials = K_total. A raw p of 0.03 across 34
                 tests is not a discovery.

  Decision rule  A signal is promoted to "worth a deeper dive" only if it
                 (a) has consistent sign across DEV and VAL,
                 (b) survives costs in both,
                 (c) beats its own permutation null,
                 (d) is not driven by fewer than 5 names or one year,
                 (e) has a stated economic reason to exist.
                 Passing TEST is necessary but is scored last and once.

  Failure        Every signal that fails is recorded WITH ITS REASON. No
  logging        signal is modified and re-run to make it work.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.backtesting.engine import BacktestConfig, prepare_panel, run_backtest
from qr.backtesting.metrics import benchmark_series, summarize
from qr.backtesting.validation import (leakage_canary, make_folds,
                                       multiple_testing_report,
                                       permutation_test, walk_forward)
from qr.strategies.signals import REGISTRY

logger = logging.getLogger("screen")

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
PROCESSED = HERE.parent / "data" / "processed"

DEV_FRAC, VAL_FRAC = 0.60, 0.20


def split_dates(panel: pd.DataFrame) -> tuple:
    d = np.sort(panel["date"].unique())
    i1 = int(len(d) * DEV_FRAC)
    i2 = int(len(d) * (DEV_FRAC + VAL_FRAC))
    return pd.Timestamp(d[0]), pd.Timestamp(d[i1]), pd.Timestamp(d[i2]), pd.Timestamp(d[-1])


def slice_panel(panel: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp,
                warmup_days: int = 420) -> pd.DataFrame:
    """Include warm-up history so signals have a lookback, but the engine only
    trades from `lo` because of its own lookback_days setting."""
    return panel[(panel["date"] >= lo - pd.Timedelta(days=warmup_days)) &
                 (panel["date"] <= hi)]


def score(panel: pd.DataFrame, fn, cfg: BacktestConfig,
          start: pd.Timestamp) -> dict:
    res = run_backtest(panel, fn, cfg)
    inside = res.daily[res.daily["date"] >= start] if len(res.daily) else res.daily
    m = summarize(inside, cfg.capital)
    m["_trades"] = res.trades
    m["_daily"] = inside
    return m


def concentration(trades: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Is this a strategy or three lucky names / one lucky year?"""
    out = {}
    if trades is not None and len(trades):
        pnl = trades.assign(pnl=trades["dollars"] * trades["fwd_ret"])
        by_sym = pnl.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
        tot = by_sym.sum()
        out["top3_symbol_share"] = float(by_sym.head(3).sum() / tot) if tot else np.nan
        out["n_symbols_traded"] = int(by_sym.shape[0])
    if daily is not None and len(daily):
        yr = daily.set_index("date")["net_ret"].resample("YE").sum()
        tot = yr.sum()
        out["best_year_share"] = float(yr.max() / tot) if tot else np.nan
        out["n_positive_years"] = int((yr > 0).sum())
        out["n_years"] = int(len(yr))
    return out


KEEP = ("n_periods", "mean_period_ret", "sharpe", "sharpe_gross", "sortino",
        "cagr", "max_drawdown", "win_rate", "profit_factor", "t_stat",
        "p_value_one_sided", "worst_month", "worst_year", "turnover_annual_x",
        "cost_drag_annual", "avg_cost_bps", "avg_spread_bps", "psr",
        "avg_n_long", "avg_n_short", "vol_annual", "cvar_5pct")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    REPORTS.mkdir(exist_ok=True)

    panel = pd.read_parquet(PROCESSED / "prices.parquet")
    logger.info("panel: %d rows, %d symbols, %s..%s", len(panel),
                panel.symbol.nunique(), panel.date.min().date(), panel.date.max().date())

    base = BacktestConfig(
        holding_days=10, rebalance_days=10, lookback_days=280,
        long_quantile=0.10, short_quantile=0.10,
        min_price=2.0, min_adv_dollar=2e6, min_names_per_side=8,
        capital=100_000.0,
    )
    prep = prepare_panel(panel, base)

    d0, d_val, d_test, d_end = split_dates(prep)
    logger.info("DEV  %s .. %s", d0.date(), d_val.date())
    logger.info("VAL  %s .. %s", d_val.date(), d_test.date())
    logger.info("TEST %s .. %s  (scored once, at the end)", d_test.date(), d_end.date())

    dev = slice_panel(prep, d0, d_val)
    val = slice_panel(prep, d_val, d_test)
    test = slice_panel(prep, d_test, d_end)

    # ---------------------------------------------------------- canary first
    logger.info("\n=== LEAKAGE CANARY (must PASS before any result is believed) ===")
    canary = leakage_canary(dev, replace(base, name="canary"))
    for k, v in canary.items():
        logger.info("  %-22s %s", k, v)
    if canary["verdict"] != "PASS":
        logger.error("CANARY FAILED -- results below cannot be trusted.")

    # ------------------------------------------------------------ the screen
    configs = {
        "ls_10d": replace(base, name="ls_10d", long_only=False,
                          holding_days=10, rebalance_days=10),
        "ls_21d": replace(base, name="ls_21d", long_only=False,
                          holding_days=21, rebalance_days=21),
    }
    n_trials = len(REGISTRY) * len(configs)
    logger.info("\n=== SCREEN: %d signals x %d configs = %d trials ===",
                len(REGISTRY), len(configs), n_trials)

    rows, failures = [], []
    for sig_name, (category, fn) in REGISTRY.items():
        for cfg_name, cfg in configs.items():
            t0 = time.time()
            try:
                dm = score(dev, fn, replace(cfg, name=f"{sig_name}:{cfg_name}"), d0)
                vm = score(val, fn, replace(cfg, name=f"{sig_name}:{cfg_name}"), d_val)
            except Exception as exc:                      # noqa: BLE001
                failures.append({"signal": sig_name, "config": cfg_name,
                                 "reason": f"exception: {str(exc)[:120]}"})
                continue

            if dm.get("n_periods", 0) < 8 or vm.get("n_periods", 0) < 4:
                failures.append({"signal": sig_name, "config": cfg_name,
                                 "reason": f"too few rebalances "
                                           f"(dev={dm.get('n_periods')}, val={vm.get('n_periods')})"})
                continue

            conc = concentration(dm.pop("_trades"), dm.pop("_daily"))
            vm.pop("_trades", None); vm.pop("_daily", None)

            row = {"signal": sig_name, "category": category, "config": cfg_name,
                   "secs": round(time.time() - t0, 1)}
            row.update({f"dev_{k}": dm.get(k) for k in KEEP})
            row.update({f"val_{k}": vm.get(k) for k in KEEP})
            row.update(conc)
            row["sign_consistent"] = bool(
                np.sign(dm.get("mean_period_ret", 0)) ==
                np.sign(vm.get("mean_period_ret", 0)))
            rows.append(row)
            logger.info("  %-20s %-8s dev SR=%6.2f ret=%+7.3f%% | "
                        "val SR=%6.2f ret=%+7.3f%% | %s",
                        sig_name, cfg_name,
                        dm.get("sharpe", np.nan) or np.nan,
                        (dm.get("mean_period_ret") or 0) * 100,
                        vm.get("sharpe", np.nan) or np.nan,
                        (vm.get("mean_period_ret") or 0) * 100,
                        "consistent" if row["sign_consistent"] else "SIGN FLIP")

    results = pd.DataFrame(rows)
    results.to_csv(REPORTS / "screen_results.csv", index=False)
    pd.DataFrame(failures).to_csv(REPORTS / "screen_failures.csv", index=False)

    if results.empty:
        logger.error("no signal produced a scorable result")
        return

    # ------------------------------------------------- multiple testing
    pvals = {f"{r.signal}:{r.config}": r.val_p_value_one_sided
             for r in results.itertuples() if np.isfinite(r.val_p_value_one_sided or np.nan)}
    mt = multiple_testing_report(pvals)
    mt.to_csv(REPORTS / "multiple_testing.csv")
    logger.info("\n=== MULTIPLE TESTING (K=%d) ===", len(pvals))
    logger.info("\n%s", mt.head(10).to_string())

    # ------------------------------------------------- promotion decision
    cand = results[
        results.sign_consistent
        & (results.dev_mean_period_ret > 0)
        & (results.val_mean_period_ret > 0)
        & (results.val_n_periods >= 4)
    ].copy()
    cand["combined_sharpe"] = cand[["dev_sharpe", "val_sharpe"]].mean(axis=1)
    cand = cand.sort_values("combined_sharpe", ascending=False)

    logger.info("\n=== CANDIDATES SURVIVING DEV+VAL (before TEST) ===")
    if cand.empty:
        logger.info("  NONE. No signal was positive and sign-consistent across dev and val.")
    else:
        logger.info("\n%s", cand[["signal", "category", "config", "dev_sharpe",
                                  "val_sharpe", "dev_mean_period_ret",
                                  "val_mean_period_ret", "top3_symbol_share",
                                  "avg_cost_bps"]].head(10).to_string(index=False))

    summary = {
        "canary": {k: v for k, v in canary.items() if not isinstance(v, dict)},
        "n_trials": n_trials,
        "n_scored": int(len(results)),
        "n_failed": int(len(failures)),
        "n_candidates": int(len(cand)),
        "dev": [str(d0.date()), str(d_val.date())],
        "val": [str(d_val.date()), str(d_test.date())],
        "test": [str(d_test.date()), str(d_end.date())],
    }
    (REPORTS / "screen_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("\nwrote reports/screen_results.csv, screen_failures.csv, "
                "multiple_testing.csv, screen_summary.json")
    return results, cand, (dev, val, test), (d0, d_val, d_test, d_end), base


if __name__ == "__main__":
    main()
