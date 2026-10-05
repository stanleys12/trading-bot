"""
Portfolio-construction strategies on liquid ETFs.
=================================================

These are allocation rules, not stock-selection signals. They do not claim to
know which asset will outperform; they claim that HOW you size exposure
matters independently of what you pick. That is a weaker claim than alpha, and
correspondingly easier to make true.

THE COMPARISON THAT DECIDES EVERYTHING

Volatility targeting changes portfolio volatility by construction, so
comparing its raw return to buy-and-hold is meaningless -- you can always
raise return by raising leverage. Two valid comparisons:

  1. SHARPE, which is already volatility-normalised. If a rule raises Sharpe,
     that is a genuine improvement in return per unit of risk.
  2. VOL-MATCHED RETURN: lever the strategy to the benchmark's volatility and
     compare returns directly. This is the honest version because it prices
     the leverage you would actually need.

We report both, and we charge FINANCING on leverage above 1x. A strategy that
only wins by being levered, without paying for the leverage, is an accounting
error rather than a strategy.

STRATEGIES
  vol_target        scale exposure inversely to trailing realised volatility
  tsmom             time-series momentum: hold if trailing return positive
  risk_parity       weight inversely to volatility across assets
  tsmom_rp          trend gate + risk-parity sizing (the classic managed-
                    futures construction, long-only variant)
  buy_hold          benchmark

ALL SIGNALS ARE LAGGED. Volatility and trend at date t use data through t-1
and the resulting weight is applied to t's return. There is no same-bar
information anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AllocConfig:
    target_vol: float = 0.10          # annualised
    vol_lookback: int = 63            # trading days
    trend_lookback: int = 252
    max_leverage: float = 2.0
    rebalance_days: int = 21          # monthly
    cost_bps: float = 3.0             # one-way, generous for liquid ETFs
    financing_spread: float = 0.01    # over cash, charged on leverage > 1


def realised_vol(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Annualised trailing vol, LAGGED one day."""
    return returns.rolling(lookback).std().shift(1) * np.sqrt(252)


def trend_signal(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """1 if trailing cumulative return positive, else 0. Lagged."""
    cum = (1 + returns).rolling(lookback).apply(np.prod, raw=True) - 1
    return (cum.shift(1) > 0).astype(float)


def _apply_rebalance(weights: pd.DataFrame, every: int) -> pd.DataFrame:
    """Hold weights constant between rebalance dates."""
    idx = weights.index
    keep = pd.Series(False, index=idx)
    keep.iloc[::every] = True
    return weights.where(keep, np.nan).ffill()


def _portfolio_return(weights: pd.DataFrame, returns: pd.DataFrame,
                      cfg: AllocConfig, cash: pd.Series | None = None
                      ) -> tuple[pd.Series, pd.Series]:
    """Net return and turnover. Costs on weight changes, financing on leverage."""
    w = weights.reindex(returns.index).fillna(0.0)
    gross = (w * returns.fillna(0)).sum(axis=1)

    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    cost = turn * cfg.cost_bps / 1e4

    lev = w.sum(axis=1)
    excess_lev = (lev - 1.0).clip(lower=0)
    if cash is not None:
        fin = excess_lev * (cash.reindex(returns.index).fillna(0)
                            + cfg.financing_spread / 252)
    else:
        fin = excess_lev * cfg.financing_spread / 252
    # unlevered residual earns cash
    under = (1.0 - lev).clip(lower=0)
    cash_earn = under * (cash.reindex(returns.index).fillna(0) if cash is not None else 0.0)

    return gross - cost - fin + cash_earn, turn


def buy_hold(returns: pd.DataFrame, asset: str, cfg: AllocConfig,
             cash: pd.Series | None = None):
    w = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    w[asset] = 1.0
    return _portfolio_return(w, returns, cfg, cash)


def vol_target(returns: pd.DataFrame, asset: str, cfg: AllocConfig,
               cash: pd.Series | None = None):
    """
    Scale a single asset's exposure to hit a constant volatility target.

    Hypothesis: volatility is persistent (it clusters) while returns are not,
    so trailing vol predicts future vol and sizing on it produces a more
    stable risk profile. The Sharpe gain, if any, comes from not being fully
    invested going INTO high-volatility regimes, which are historically also
    low-return regimes.
    """
    vol = realised_vol(returns[[asset]], cfg.vol_lookback)
    w = (cfg.target_vol / vol).clip(upper=cfg.max_leverage).fillna(0.0)
    w = _apply_rebalance(w, cfg.rebalance_days)
    full = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    full[asset] = w[asset]
    return _portfolio_return(full, returns, cfg, cash)


def tsmom(returns: pd.DataFrame, assets: list[str], cfg: AllocConfig,
          cash: pd.Series | None = None):
    """
    Time-series momentum: hold each asset only when its own trailing return
    is positive; otherwise hold cash. Equal weight across held assets.

    Hypothesis (Moskowitz-Ooi-Pedersen): under-reaction to news then
    over-reaction produces persistent own-asset autocorrelation at 1-12m.
    """
    sub = returns[assets]
    sig = trend_signal(sub, cfg.trend_lookback)
    w = sig.div(len(assets))
    w = _apply_rebalance(w, cfg.rebalance_days)
    full = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for a in assets:
        full[a] = w[a]
    return _portfolio_return(full, returns, cfg, cash)


def risk_parity(returns: pd.DataFrame, assets: list[str], cfg: AllocConfig,
                cash: pd.Series | None = None):
    """
    Weight inversely to trailing volatility, scaled to the vol target.

    Hypothesis: equal-weighting dollars concentrates risk in the most volatile
    asset. Equalising risk contribution produces better diversification, which
    is the only free lunch in finance.
    """
    sub = returns[assets]
    vol = realised_vol(sub, cfg.vol_lookback)
    inv = 1.0 / vol.replace(0, np.nan)
    w = inv.div(inv.sum(axis=1), axis=0)

    port_vol = (w * vol).sum(axis=1)
    scale = (cfg.target_vol / port_vol).clip(upper=cfg.max_leverage)
    w = w.mul(scale, axis=0).fillna(0.0)
    w = _apply_rebalance(w, cfg.rebalance_days)
    full = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for a in assets:
        full[a] = w[a]
    return _portfolio_return(full, returns, cfg, cash)


def tsmom_rp(returns: pd.DataFrame, assets: list[str], cfg: AllocConfig,
             cash: pd.Series | None = None):
    """Trend gate on top of risk-parity sizing -- the managed-futures shape."""
    sub = returns[assets]
    vol = realised_vol(sub, cfg.vol_lookback)
    sig = trend_signal(sub, cfg.trend_lookback)
    inv = (1.0 / vol.replace(0, np.nan)) * sig
    tot = inv.sum(axis=1).replace(0, np.nan)
    w = inv.div(tot, axis=0)

    port_vol = (w * vol).sum(axis=1)
    scale = (cfg.target_vol / port_vol).clip(upper=cfg.max_leverage)
    w = w.mul(scale, axis=0).fillna(0.0)
    w = _apply_rebalance(w, cfg.rebalance_days)
    full = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for a in assets:
        full[a] = w[a]
    return _portfolio_return(full, returns, cfg, cash)


# ------------------------------------------------------------------ metrics
def vol_matched_return(strategy: pd.Series, benchmark: pd.Series,
                       financing: float = 0.01) -> dict:
    """
    Lever the strategy to the benchmark's volatility and compare returns.

    This is the comparison that cannot be gamed: it prices the leverage a
    strategy would actually need to be interesting, including its cost.
    """
    idx = strategy.dropna().index.intersection(benchmark.dropna().index)
    s, b = strategy.loc[idx], benchmark.loc[idx]
    s_vol, b_vol = s.std(ddof=1), b.std(ddof=1)
    if s_vol == 0:
        return {}
    lev = b_vol / s_vol
    levered = s * lev - max(lev - 1.0, 0) * financing / 252
    ppy = 252
    return {
        "leverage_needed": float(lev),
        "strategy_vol": float(s_vol * np.sqrt(ppy)),
        "benchmark_vol": float(b_vol * np.sqrt(ppy)),
        "levered_strategy_return": float(levered.mean() * ppy),
        "benchmark_return": float(b.mean() * ppy),
        "excess_at_matched_vol": float((levered.mean() - b.mean()) * ppy),
        "wins": bool(levered.mean() > b.mean()),
    }
