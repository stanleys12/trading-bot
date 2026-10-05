"""
Candidate evaluation: residual alpha, survivorship bounding, scorecard.
=======================================================================

Three things this module exists to stop us doing.

1. REPORTING CAGR AS IF IT WERE SKILL.
   The fundamental screen produced 18-21% CAGR against SPY's 14.8% and it
   would have been easy to call that a win. It is not: those portfolios ran
   25-31% volatility against SPY's ~18%, so a levered SPY position dominates
   them. The question is never "did it make more money", it is "did it make
   money the benchmark could not have made for me more cheaply".
   `alpha_vs_benchmarks` answers that by regressing strategy returns on SPY
   and IWM and reporting the INTERCEPT.

2. TREATING SURVIVORSHIP AS A CAVEAT RATHER THAN A NUMBER.
   37% of the point-in-time universe has no recoverable prices, skewed toward
   bankruptcies. We cannot recover them -- that is a data-licensing wall, not
   an effort problem. But we CAN bound the damage: we know WHICH names
   vanished and WHEN, from the archived exchange directories. So we impute a
   terminal delisting return for every departed name at its last observed
   date and re-run. Following Shumway (1997), -30% is the conventional
   estimate for ordinary delistings and -55% for performance-related ones;
   we report a range from optimistic to punitive rather than a point estimate.
   That converts an unbounded worry into a measured interval.

3. LETTING A CANDIDATE PASS ON VIBES.
   `scorecard` applies fixed thresholds and returns PASS/FAIL per criterion.
   No candidate is promoted on a subset of green cells.

A note on the multiple-testing count. The honest denominator is not the number
of hypotheses in the current script, it is every configuration this project
has ever scored. `CUMULATIVE_TRIALS` tracks that, and the deflated Sharpe uses
it. It only goes up.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Every configuration scored anywhere in this project, cumulative.
# screen v1: 34 | liquidity tiers: 30 | deep dive robustness: 18
# long-only singles: 34 | exploration: 12 | price stress: 6 | reverse split: 6
# lead-lag/size: 11 | candidate+turnover: 15 | fundamentals IC+backtests: 26
CUMULATIVE_TRIALS = 192


@dataclass(frozen=True)
class Thresholds:
    min_sharpe: float = 0.80          # must at least match SPY over the period
    min_alpha_t: float = 2.0          # residual alpha must be significant
    min_alpha_annual: float = 0.02    # and economically meaningful
    max_drawdown: float = -0.40
    max_turnover: float = 12.0        # per year
    min_capacity_usd: float = 250_000
    min_periods: int = 24
    survives_cost_multiple: float = 2.0


def alpha_vs_benchmarks(strategy_ret: pd.Series,
                        benchmarks: dict[str, pd.Series],
                        periods_per_year: float) -> dict:
    """
    Regress strategy returns on benchmark returns; the intercept is alpha.

    Reports both single-factor (SPY) and multi-factor (SPY + IWM) versions.
    IWM matters here because a small-cap strategy loaded on small-cap beta is
    not producing alpha, it is producing IWM with extra steps.
    """
    out = {}
    idx = strategy_ret.dropna().index
    for name, b in benchmarks.items():
        idx = idx.intersection(b.dropna().index)
    if len(idx) < 20:
        return {"error": f"only {len(idx)} overlapping periods"}

    y = strategy_ret.loc[idx].values
    for label, cols in (("vs_SPY", ["SPY"]),
                        ("vs_SPY_IWM", ["SPY", "IWM"])):
        cols = [c for c in cols if c in benchmarks]
        if not cols:
            continue
        X = np.column_stack([benchmarks[c].loc[idx].values for c in cols])
        X = np.column_stack([np.ones(len(idx)), X])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            dof = len(y) - X.shape[1]
            se = np.sqrt(np.sum(resid ** 2) / dof *
                         np.diag(np.linalg.inv(X.T @ X)))
            out[label] = {
                "alpha_per_period": float(beta[0]),
                "alpha_annual": float(beta[0] * periods_per_year),
                "alpha_t": float(beta[0] / se[0]) if se[0] > 0 else np.nan,
                **{f"beta_{c}": float(b) for c, b in zip(cols, beta[1:])},
                "resid_vol_annual": float(resid.std(ddof=1) * np.sqrt(periods_per_year)),
                "r_squared": float(1 - resid.var() / y.var()) if y.var() else np.nan,
            }
        except np.linalg.LinAlgError:
            out[label] = {"error": "singular"}

    # The comparison that actually settles it: could levered benchmark do better?
    if "SPY" in benchmarks:
        spy = benchmarks["SPY"].loc[idx]
        s_vol = strategy_ret.loc[idx].std(ddof=1)
        b_vol = spy.std(ddof=1)
        if b_vol > 0:
            lev = s_vol / b_vol
            out["levered_benchmark"] = {
                "leverage_to_match_vol": float(lev),
                "levered_spy_return_annual": float(spy.mean() * lev * periods_per_year),
                "strategy_return_annual": float(strategy_ret.loc[idx].mean() * periods_per_year),
                "strategy_beats_levered_spy": bool(
                    strategy_ret.loc[idx].mean() > spy.mean() * lev),
            }
    return out


def impute_delisting_returns(panel: pd.DataFrame, membership: pd.DataFrame,
                             terminal_return: float = -0.30) -> pd.DataFrame:
    """
    Append a synthetic final observation for every departed name we can price.

    We know which symbols vanished and roughly when (archived exchange
    directories). For each, we add one extra row after its last observed date
    carrying `terminal_return`. Names that departed and were NEVER priced
    cannot be repaired this way -- their absence is the residual bias, and it
    is why the bound is one-sided.

    Shumway (1997) estimates roughly -30% for ordinary delistings and -55%
    where the delisting was performance-related. We take the level as a
    parameter and report a range rather than pretending to know it.
    """
    dead = set(membership[membership["delisted"]]["symbol"])
    have_dead = [s for s in panel["symbol"].unique() if s in dead]
    if not have_dead:
        return panel

    last = (panel[panel["symbol"].isin(have_dead)]
            .sort_values("date").groupby("symbol").tail(1).copy())
    # one synthetic bar, a week after the last real one
    last["date"] = last["date"] + pd.Timedelta(days=7)
    for c in ("close", "true_close", "open", "high", "low", "adj_close"):
        if c in last.columns:
            last[c] = last[c] * (1 + terminal_return)
    last["volume"] = 0.0
    last["dollar_volume"] = 0.0
    last["_imputed_delisting"] = True

    panel = panel.copy()
    panel["_imputed_delisting"] = False
    out = pd.concat([panel, last], ignore_index=True)
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def survivorship_bound(run_fn, panel: pd.DataFrame, membership: pd.DataFrame,
                       levels=(0.0, -0.30, -0.55, -1.00)) -> pd.DataFrame:
    """
    Re-run a strategy across delisting-return assumptions.

    `run_fn(panel) -> dict of metrics`. The spread across levels is the
    measured sensitivity to the one bias we cannot remove. A candidate whose
    result flips sign between 0% and -30% was never a candidate.
    """
    rows = []
    for lv in levels:
        p = panel if lv == 0.0 else impute_delisting_returns(panel, membership, lv)
        try:
            m = run_fn(p)
        except Exception as exc:                          # noqa: BLE001
            rows.append({"delisting_return": lv, "error": str(exc)[:60]})
            continue
        rows.append({"delisting_return": lv, **m})
    return pd.DataFrame(rows)


def scorecard(metrics: dict, alpha: dict, cost_sensitivity: dict | None = None,
              th: Thresholds = Thresholds()) -> pd.DataFrame:
    """Fixed criteria, PASS/FAIL. No partial credit."""
    a_spy = alpha.get("vs_SPY", {})
    a_both = alpha.get("vs_SPY_IWM", {})
    lev = alpha.get("levered_benchmark", {})

    checks = [
        ("out-of-sample Sharpe", metrics.get("sharpe"),
         f">= {th.min_sharpe}", (metrics.get("sharpe") or -9) >= th.min_sharpe),
        ("alpha vs SPY (annual)", a_spy.get("alpha_annual"),
         f">= {th.min_alpha_annual:.1%}",
         (a_spy.get("alpha_annual") or -9) >= th.min_alpha_annual),
        ("alpha t-stat vs SPY", a_spy.get("alpha_t"),
         f">= {th.min_alpha_t}", (a_spy.get("alpha_t") or -9) >= th.min_alpha_t),
        ("alpha vs SPY+IWM (annual)", a_both.get("alpha_annual"),
         f">= {th.min_alpha_annual:.1%}",
         (a_both.get("alpha_annual") or -9) >= th.min_alpha_annual),
        ("beats vol-matched levered SPY", lev.get("strategy_return_annual"),
         f"> {lev.get('levered_spy_return_annual', float('nan')):.3f}",
         bool(lev.get("strategy_beats_levered_spy", False))),
        ("max drawdown", metrics.get("max_drawdown"),
         f">= {th.max_drawdown}",
         (metrics.get("max_drawdown") or -9) >= th.max_drawdown),
        ("turnover per year", metrics.get("turnover_annual_x"),
         f"<= {th.max_turnover}",
         (metrics.get("turnover_annual_x") or 999) <= th.max_turnover),
        ("periods observed", metrics.get("n_periods"),
         f">= {th.min_periods}", (metrics.get("n_periods") or 0) >= th.min_periods),
        ("deflated Sharpe (K=%d)" % CUMULATIVE_TRIALS, metrics.get("deflated_sharpe"),
         ">= 0.95", (metrics.get("deflated_sharpe") or 0) >= 0.95),
    ]
    if cost_sensitivity:
        checks.append(("survives %gx costs" % th.survives_cost_multiple,
                       cost_sensitivity.get("sharpe_at_multiple"),
                       "> 0", (cost_sensitivity.get("sharpe_at_multiple") or -9) > 0))

    df = pd.DataFrame(checks, columns=["criterion", "value", "requirement", "pass"])
    df["value"] = df["value"].apply(
        lambda v: round(v, 4) if isinstance(v, (int, float)) and v is not None else v)
    return df


def verdict(sc: pd.DataFrame) -> str:
    n_fail = int((~sc["pass"]).sum())
    if n_fail == 0:
        return "PROMOTE"
    if n_fail <= 2:
        return f"HOLD ({n_fail} criteria failed)"
    return f"REJECT ({n_fail} criteria failed)"
