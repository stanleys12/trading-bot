"""
Deep dive on whatever survived the screen.

Runs the four hurdles a candidate must clear before it is worth any further
effort. Each is designed to kill the candidate; passing means it resisted a
genuine attempt, not that it looked good once.

  H1  PERMUTATION       shuffle the signal within each date. If the shuffled
                        version scores as well, there is no signal.
  H2  WALK-FORWARD      purged, embargoed, expanding-window folds. A real
                        effect is positive in most folds, not one.
  H3  ROBUSTNESS        vary holding period, quantile width, cost assumption,
                        and universe liquidity floor. A real effect degrades
                        smoothly; an artefact vanishes off its exact settings.
  H4  BENCHMARK         compare to buy-and-hold SPY and IWM over the same
                        window. A long-short book that earns less than IWM
                        with more operational risk is not worth running.

Finally, the untouched TEST window is scored ONCE.
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

from qr.backtesting.costs import CostConfig
from qr.backtesting.engine import BacktestConfig, prepare_panel, run_backtest
from qr.backtesting.metrics import benchmark_series, deflated_sharpe, summarize
from qr.backtesting.validation import (make_folds, permutation_test,
                                       walk_forward)
from qr.strategies.signals import REGISTRY

logger = logging.getLogger("deep")
HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
PROCESSED = HERE.parent / "data" / "processed"

DEV_FRAC, VAL_FRAC = 0.60, 0.20

# Candidate configuration surfaced by research/liquidity_tiers.py:
#   12-1 momentum, microcap tier, concentrated (~25 names/side), 21-day hold.
# Concentration is the whole point -- at 83 names/side the $1 minimum
# commission alone is 17bps one way; at 25 names it is ~5bps.
import os
HOLD  = int(os.environ.get("QR_HOLD", "21"))
QUANT = float(os.environ.get("QR_QUANT", "0.03"))
ADV   = float(os.environ.get("QR_ADV", "2e6"))


def flip(fn):
    """Return the sign-inverted signal. Counted as its own trial."""
    def g(vis):
        s = fn(vis)
        return -s if s is not None and len(s) else s
    g.__name__ = f"neg_{fn.__name__}"
    return g


def main(signal_name: str, invert: bool = False, n_trials: int = 34):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    panel = pd.read_parquet(PROCESSED / "prices.parquet")

    base = BacktestConfig(
        holding_days=HOLD, rebalance_days=HOLD, lookback_days=280,
        long_quantile=QUANT, short_quantile=QUANT, min_price=2.0,
        min_adv_dollar=ADV, min_names_per_side=8, capital=100_000.0,
        name=f"{'neg_' if invert else ''}{signal_name}",
    )
    prep = prepare_panel(panel, base)
    d = np.sort(prep["date"].unique())
    d_val = pd.Timestamp(d[int(len(d) * DEV_FRAC)])
    d_test = pd.Timestamp(d[int(len(d) * (DEV_FRAC + VAL_FRAC))])

    category, fn = REGISTRY[signal_name]
    if invert:
        fn = flip(fn)

    devval = prep[prep["date"] < d_test]
    test = prep[prep["date"] >= d_test - pd.Timedelta(days=420)]

    out: dict = {"signal": signal_name, "inverted": invert,
                 "category": category, "n_trials_for_dsr": n_trials}

    # ------------------------------------------------------- H1 permutation
    logger.info("\n=== H1  PERMUTATION TEST (dev+val) ===")
    perm = permutation_test(devval, fn, replace(base, name="perm"), n_perm=25)
    out["permutation"] = perm
    for k, v in perm.items():
        logger.info("  %-26s %s", k, round(v, 5) if isinstance(v, float) else v)

    # ------------------------------------------------------ H2 walk-forward
    logger.info("\n=== H2  PURGED WALK-FORWARD ===")
    folds = make_folds(pd.Series(prep["date"].unique()), n_folds=5,
                       purge_days=20, embargo_days=10, min_train_days=500)
    wf = walk_forward(prep, fn, base, folds)
    out["walk_forward"] = wf.to_dict("records")
    logger.info("\n%s", wf.to_string(index=False))
    if "mean_period_ret" in wf.columns:
        pos = (wf["mean_period_ret"] > 0).sum()
        logger.info("  folds positive: %d/%d", pos, len(wf))
        out["folds_positive"] = int(pos)
        out["folds_total"] = int(len(wf))

    # -------------------------------------------------------- H3 robustness
    logger.info("\n=== H3  ROBUSTNESS TO SETTINGS ===")
    grid = []
    for hold in (5, 10, 21):
        for q in (0.05, 0.10, 0.20):
            for cost_mult, clabel in ((1.0, "modelled"), (2.0, "2x cost")):
                cfg = replace(base, holding_days=hold, rebalance_days=hold,
                              long_quantile=q, short_quantile=q,
                              costs=CostConfig(
                                  impact_eta=0.8 * cost_mult,
                                  spread_floor_bps=5.0 * cost_mult,
                                  spread_capture=1.0 * cost_mult))
                try:
                    p2 = prepare_panel(panel, cfg)
                    dv = p2[p2["date"] < d_test]
                    r = run_backtest(dv, fn, cfg)
                    m = summarize(r.daily, cfg.capital)
                    grid.append({"hold": hold, "quantile": q, "costs": clabel,
                                 "n": m.get("n_periods"),
                                 "mean_ret": m.get("mean_period_ret"),
                                 "sharpe": m.get("sharpe"),
                                 "maxdd": m.get("max_drawdown")})
                except Exception as exc:                  # noqa: BLE001
                    grid.append({"hold": hold, "quantile": q, "costs": clabel,
                                 "error": str(exc)[:60]})
    gdf = pd.DataFrame(grid)
    out["robustness"] = gdf.to_dict("records")
    logger.info("\n%s", gdf.to_string(index=False))
    if "mean_ret" in gdf.columns:
        frac = float((gdf["mean_ret"] > 0).mean())
        logger.info("  positive in %.0f%% of %d settings", frac * 100, len(gdf))
        out["robust_fraction_positive"] = frac

    # --------------------------------------------------------- H4 benchmark
    logger.info("\n=== H4  VS BUY-AND-HOLD BENCHMARKS ===")
    res_dv = run_backtest(devval, fn, base)
    m_dv = summarize(res_dv.daily, base.capital, n_trials=n_trials)
    bench = {}
    for sym in ("SPY", "IWM"):
        b = benchmark_series(panel, sym, res_dv.daily["date"])
        if len(b) > 3:
            bench[sym] = {
                "mean_period_ret": float(b.mean()),
                "sharpe": float(b.mean() / b.std(ddof=1) * np.sqrt(25.2))
                          if b.std(ddof=1) > 0 else np.nan,
                "total": float((1 + b).prod() - 1),
            }
    out["strategy_devval"] = {k: m_dv.get(k) for k in
                              ("mean_period_ret", "sharpe", "sortino", "cagr",
                               "max_drawdown", "win_rate", "profit_factor",
                               "turnover_annual_x", "cost_drag_annual",
                               "worst_month", "worst_year", "psr",
                               "deflated_sharpe", "avg_cost_bps", "vol_annual")}
    out["benchmarks"] = bench
    logger.info("  strategy : %s", {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in out["strategy_devval"].items()})
    for k, v in bench.items():
        logger.info("  %-8s : %s", k, {kk: round(vv, 4) for kk, vv in v.items()})

    # ---------------------------------------------------- FINAL: TEST once
    logger.info("\n=== FINAL  UNTOUCHED TEST WINDOW (scored once) ===")
    res_t = run_backtest(test, fn, base)
    inside = res_t.daily[res_t.daily["date"] >= d_test]
    m_t = summarize(inside, base.capital, n_trials=n_trials)
    out["TEST"] = {k: m_t.get(k) for k in
                   ("n_periods", "mean_period_ret", "sharpe", "sortino", "cagr",
                    "max_drawdown", "win_rate", "profit_factor", "t_stat",
                    "p_value_one_sided", "psr", "deflated_sharpe",
                    "worst_month", "avg_cost_bps", "turnover_annual_x")}
    for k, v in out["TEST"].items():
        logger.info("  %-22s %s", k, round(v, 4) if isinstance(v, float) else v)

    tag = f"{'neg_' if invert else ''}{signal_name}"
    (REPORTS / f"deep_dive_{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    logger.info("\nwrote reports/deep_dive_%s.json", tag)
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    inv = len(sys.argv) > 2 and sys.argv[2] == "invert"
    main(name, inv)
