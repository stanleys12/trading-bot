"""
Portfolio construction.
=======================

Turns a cross-sectional signal into dollar targets. Kept separate from the
engine so that sizing rules can be tested and swapped without touching either
the signal or the execution path.

The rule that matters most for a small account
-----------------------------------------------
`min_position_dollars` exists because of a result from this project's own
screen: at $100k of capital spread across ~85 names per side, each position is
about $590, and a $1 minimum commission is then 17bps one way before the spread
is even considered. Fixed costs do not scale down. A small account cannot run a
wide book, and pretending otherwise is how a backtest quietly becomes fiction.

`max_adv_participation` is the other binding constraint, in the opposite
direction: it caps how large any position can be relative to the name's
liquidity, and it is what makes capacity estimates honest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SizingConfig:
    capital: float = 100_000.0
    gross_leverage: float = 1.0
    long_only: bool = False
    max_position_pct: float = 0.10        # of capital, per name
    max_adv_participation: float = 0.02   # of the name's 21d ADV
    min_position_dollars: float = 2_000.0 # below this, fixed costs dominate
    vol_target_annual: float | None = None


def equal_weight(selected: pd.DataFrame, side: str,
                 cfg: SizingConfig) -> pd.Series:
    """Equal dollar weight, capped by both capital share and ADV."""
    if selected.empty:
        return pd.Series(dtype=float)
    n_sides = 1 if cfg.long_only else 2
    side_capital = cfg.capital * cfg.gross_leverage / n_sides
    per = side_capital / len(selected)

    cap_capital = cfg.capital * cfg.max_position_pct
    cap_adv = selected["adv_dollar"].fillna(0) * cfg.max_adv_participation
    dollars = np.minimum(np.minimum(per, cap_capital), cap_adv)

    sign = 1.0 if side == "long" else -1.0
    out = pd.Series(sign * dollars.values, index=selected["symbol"].values)
    return out[out.abs() >= cfg.min_position_dollars]


def inverse_vol_weight(selected: pd.DataFrame, side: str,
                       cfg: SizingConfig) -> pd.Series:
    """
    Risk-parity within the leg: weight inversely to trailing volatility so a
    handful of highly volatile microcaps do not dominate the book's risk.
    """
    if selected.empty:
        return pd.Series(dtype=float)
    vol = selected["vol_est"].replace(0, np.nan)
    vol = vol.fillna(vol.median())
    if not np.isfinite(vol).any():
        return equal_weight(selected, side, cfg)

    inv = 1.0 / vol
    w = inv / inv.sum()
    n_sides = 1 if cfg.long_only else 2
    side_capital = cfg.capital * cfg.gross_leverage / n_sides
    dollars = w.values * side_capital

    cap_capital = cfg.capital * cfg.max_position_pct
    cap_adv = (selected["adv_dollar"].fillna(0) * cfg.max_adv_participation).values
    dollars = np.minimum(np.minimum(dollars, cap_capital), cap_adv)

    sign = 1.0 if side == "long" else -1.0
    out = pd.Series(sign * dollars, index=selected["symbol"].values)
    return out[out.abs() >= cfg.min_position_dollars]


def max_deployable_capital(selected: pd.DataFrame, cfg: SizingConfig) -> float:
    """
    Capacity: the largest book this selection could support before breaching
    the ADV participation cap. This is the number that decides whether a
    strategy is worth running at all, and it is usually the one that is
    quietly omitted.
    """
    if selected.empty:
        return 0.0
    per_name_cap = selected["adv_dollar"].fillna(0) * cfg.max_adv_participation
    n_sides = 1 if cfg.long_only else 2
    return float(per_name_cap.sum() * n_sides / cfg.gross_leverage)


def turnover(prev: pd.Series, new: pd.Series) -> float:
    """One-way turnover in dollars between two target books."""
    idx = prev.index.union(new.index)
    return float((new.reindex(idx, fill_value=0.0)
                  - prev.reindex(idx, fill_value=0.0)).abs().sum())
