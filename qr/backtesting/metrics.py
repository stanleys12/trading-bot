"""
Performance and risk metrics.

Every metric here is computed from the NET return series (after costs). The
period-return series is per-rebalance, not per-day, so annualisation uses the
observed rebalance frequency rather than an assumed 252.

Deliberate omissions and cautions
---------------------------------
- We do NOT report an annualised Sharpe computed as mean/std*sqrt(252) on a
  per-trade series. That was how the previous research produced "Sharpe 8.79"
  on a strategy trading four times a month. Annualisation uses periods/year
  derived from the actual timestamps.
- Sharpe on a short, fat-tailed sample is badly biased upward. We report the
  probabilistic Sharpe ratio (PSR) alongside it, which corrects for skew and
  kurtosis, and the deflated Sharpe ratio (DSR) which additionally corrects
  for the number of strategies tried.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _periods_per_year(dates: pd.Series) -> float:
    if len(dates) < 3:
        return 252.0
    span_days = (dates.max() - dates.min()).days
    if span_days <= 0:
        return 252.0
    return len(dates) / (span_days / 365.25)


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0)).cumprod()


def max_drawdown(eq: pd.Series) -> float:
    peak = eq.cummax()
    return float(((eq - peak) / peak).min())


def sharpe(returns: pd.Series, ppy: float, rf: float = 0.0) -> float:
    r = returns.dropna() - rf / ppy
    if len(r) < 3 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ppy))


def sortino(returns: pd.Series, ppy: float) -> float:
    r = returns.dropna()
    downside = r[r < 0]
    if len(downside) < 2 or downside.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / downside.std(ddof=1) * np.sqrt(ppy))


def probabilistic_sharpe(returns: pd.Series, ppy: float,
                         benchmark_sr: float = 0.0) -> float:
    """
    Bailey & Lopez de Prado: P(true Sharpe > benchmark), correcting for
    non-normality and sample length. Values below ~0.95 mean the observed
    Sharpe is not distinguishable from the benchmark.
    """
    r = returns.dropna()
    n = len(r)
    if n < 8 or r.std(ddof=1) == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)                 # per-period
    sr_bench = benchmark_sr / np.sqrt(ppy)
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    z = (sr - sr_bench) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe(returns: pd.Series, ppy: float, n_trials: int,
                    trial_sr_std: float | None = None) -> float:
    """
    Deflated Sharpe Ratio: PSR against a benchmark set to the Sharpe you would
    EXPECT to see as the maximum of `n_trials` independent random strategies.
    This is the honest way to report "I tested 40 things and this was best".
    """
    r = returns.dropna()
    if len(r) < 8 or n_trials < 1:
        return np.nan
    if trial_sr_std is None:
        trial_sr_std = r.std(ddof=1) / np.sqrt(len(r))  # crude fallback
    e = np.euler_gamma
    if n_trials == 1:
        sr0 = 0.0
    else:
        sr0 = trial_sr_std * (
            (1 - e) * stats.norm.ppf(1 - 1 / n_trials)
            + e * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )
    return probabilistic_sharpe(r, ppy, benchmark_sr=sr0 * np.sqrt(ppy))


def summarize(daily: pd.DataFrame, capital: float = 100_000.0,
              n_trials: int = 1) -> dict:
    """Full metric set from an engine result's `daily` frame."""
    if daily is None or daily.empty:
        return {"n_periods": 0}

    d = daily.sort_values("date").copy()
    ppy = _periods_per_year(d["date"])
    net, gross = d["net_ret"], d["gross_ret"]

    eq = equity_curve(net)
    years = max((d["date"].max() - d["date"].min()).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else np.nan

    monthly = (d.set_index("date")["net_ret"]
               .resample("ME").apply(lambda x: (1 + x).prod() - 1))
    yearly = (d.set_index("date")["net_ret"]
              .resample("YE").apply(lambda x: (1 + x).prod() - 1))

    wins, losses = net[net > 0], net[net < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())

    turnover_ann = float(d["turnover_dollars"].sum() / capital / years)

    return {
        "n_periods": int(len(d)),
        "periods_per_year": round(ppy, 2),
        "years": round(years, 2),
        "total_return": float(eq.iloc[-1] - 1),
        "cagr": cagr,
        "vol_annual": float(net.std(ddof=1) * np.sqrt(ppy)),
        "sharpe": sharpe(net, ppy),
        "sharpe_gross": sharpe(gross, ppy),
        "sortino": sortino(net, ppy),
        "psr": probabilistic_sharpe(net, ppy),
        "deflated_sharpe": deflated_sharpe(net, ppy, n_trials),
        "max_drawdown": max_drawdown(eq),
        "win_rate": float((net > 0).mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.inf,
        "mean_period_ret": float(net.mean()),
        "median_period_ret": float(net.median()),
        "t_stat": float(net.mean() / (net.std(ddof=1) / np.sqrt(len(net))))
                  if net.std(ddof=1) > 0 else np.nan,
        "p_value_one_sided": float(
            1 - stats.t.cdf(net.mean() / (net.std(ddof=1) / np.sqrt(len(net))),
                            df=len(net) - 1)) if net.std(ddof=1) > 0 else np.nan,
        "worst_month": float(monthly.min()) if len(monthly) else np.nan,
        "worst_year": float(yearly.min()) if len(yearly) else np.nan,
        "pct_months_positive": float((monthly > 0).mean()) if len(monthly) else np.nan,
        "cvar_5pct": float(net[net <= net.quantile(0.05)].mean())
                     if len(net) > 20 else np.nan,
        "skew": float(stats.skew(net)) if len(net) > 3 else np.nan,
        "kurtosis": float(stats.kurtosis(net, fisher=False)) if len(net) > 3 else np.nan,
        "turnover_annual_x": turnover_ann,
        "avg_gross_exposure": float(d["gross_notional"].mean() / capital),
        "cost_drag_annual": float(d["cost_dollars"].sum() / capital / years),
        "gross_minus_net_bps_per_period": float(
            (gross - net).mean() * 1e4),
        "avg_cost_bps": float(d["avg_cost_bps"].mean()),
        "avg_spread_bps": float(d["avg_spread_bps"].mean()),
        "avg_n_long": float(d["n_long"].mean()),
        "avg_n_short": float(d["n_short"].mean()),
    }


def benchmark_series(panel: pd.DataFrame, symbol: str,
                     dates: pd.Series) -> pd.Series:
    """Buy-and-hold return of `symbol` sampled on the strategy's rebalance dates."""
    b = (panel[panel["symbol"] == symbol]
         .sort_values("date")
         .set_index("date")["adj_close"])
    if b.empty:
        return pd.Series(dtype=float)
    aligned = b.reindex(pd.DatetimeIndex(dates), method="ffill")
    return aligned.pct_change(fill_method=None).fillna(0)
