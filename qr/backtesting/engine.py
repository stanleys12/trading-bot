"""
Cross-sectional backtest engine.
================================

Design contract
---------------
The engine never sees a strategy's internals. A strategy is a function

    signal(panel_asof) -> Series indexed by symbol

that is called ONCE PER REBALANCE DATE and is handed a frame containing only
rows with date <= the information cutoff. It is structurally impossible for a
strategy to read the future, because the future is not in the frame it is
given. This is the single most important design decision in the project: the
previous research leaked because features and outcomes lived in the same row
of the same dataframe, and nothing stopped a filter from touching a column it
should not have.

Timing convention (the thing that actually caused the 86% mirage)
-----------------------------------------------------------------
    decision_date  D   : strategy sees data through D's CLOSE
    entry              : D+1 OPEN         (you cannot trade D's close on D)
    exit               : D+1+H OPEN
So the return earned is open-to-open, and no part of the entry bar is used to
form the decision. For gap strategies that legitimately observe D+1's open
before deciding, set `use_entry_open_in_signal=True`, which passes the open of
the entry bar (and nothing else from it) to the strategy.

Portfolio construction is explicit
----------------------------------
Signals are ranked cross-sectionally, converted to target weights, capped by
ADV participation, and costed on turnover. Nothing is implicit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .costs import (CostConfig, add_cost_features, shortable_mask,
                    size_cap_dollars, trade_cost_bps)

logger = logging.getLogger(__name__)

SignalFn = Callable[[pd.DataFrame], pd.Series]


@dataclass
class BacktestConfig:
    name: str = "unnamed"
    hypothesis: str = ""
    holding_days: int = 5
    rebalance_days: int = 5
    long_quantile: float = 0.1          # top decile
    short_quantile: float = 0.1         # bottom decile
    long_only: bool = False
    gross_leverage: float = 1.0
    capital: float = 100_000.0
    min_names_per_side: int = 5
    min_price: float = 1.0
    max_price: float = 1e9
    min_adv_dollar: float = 1e6
    price_cap: float = 1_000.0        # above this = reverse-split artefact
    max_price_range: float = 50.0     # trailing 252d max/min; >this = artefact
    lookback_days: int = 252            # how much history the signal may see
    use_entry_open_in_signal: bool = False
    costs: CostConfig = field(default_factory=CostConfig)
    seed: int = 0


@dataclass
class BacktestResult:
    config: BacktestConfig
    daily: pd.DataFrame                 # date, gross_ret, net_ret, turnover, ...
    trades: pd.DataFrame
    diagnostics: dict


class LookaheadError(RuntimeError):
    """Raised when a strategy is handed, or returns, information it cannot have."""


def prepare_panel(panel: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Attach cost features and forward returns. Called once."""
    df = add_cost_features(panel, cfg.costs)

    parts = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        # Returns are computed from ADJUSTED prices (splits are not returns),
        # but every FILTER downstream uses raw close.
        g["adj_open"] = g["open"] * g["adj_factor"]
        g["next_open"] = g["adj_open"].shift(-1)
        g["exit_open"] = g["adj_open"].shift(-(1 + cfg.holding_days))
        g["fwd_ret"] = g["exit_open"] / g["next_open"] - 1.0
        g["raw_next_open"] = g["open"].shift(-1)
        # Trailing-only split-contamination detector (see _eligible).
        roll = g["close"].rolling(252, min_periods=60)
        g["range_ratio_252"] = (roll.max() / roll.min().replace(0, np.nan)).shift(1)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _eligible(day: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Point-in-time tradeability filter. Uses only lagged/observable fields.

    SPLIT CONTAMINATION GUARD. Yahoo's `Close` is split-adjusted even when
    `auto_adjust=False` -- that flag disables only the DIVIDEND adjustment.
    Reverse splits are therefore back-propagated into history: a stock that
    did 1:100 shows a pre-split price 100x too high. ADTX, which reverse-split
    seven times, appears in this panel at a nominal $115,450 and, at the
    extreme, $1.49bn per share.

    That breaks any absolute price filter, which is exactly the defect this
    project was built to catch, so it is caught here rather than trusted:

      price_cap        no name in this universe legitimately trades above
                       $1,000; anything that does is an artefact of a later
                       reverse split.
      range_ratio_252  a trailing 252-day max/min ratio above `max_price_range`
                       is the reverse-split death-spiral signature. Computed
                       from trailing data only, so it stays point-in-time.

    Both are heuristics. The correct fix is a split-factor time series, which
    this data source does not provide and Polygon does. Until then these
    filters are deliberately aggressive: they discard real names to be sure of
    discarding fake ones.
    """
    m = (
        day["close"].between(cfg.min_price, cfg.max_price)
        & (day["close"] <= cfg.price_cap)
        & (day["adv_dollar"] >= cfg.min_adv_dollar)
        & day["adv_dollar"].notna()
        & day["spread_est"].notna()
    )
    if "range_ratio_252" in day.columns:
        m &= (day["range_ratio_252"].fillna(np.inf) <= cfg.max_price_range)
    return day[m]


def run_backtest(panel: pd.DataFrame, signal_fn: SignalFn,
                 cfg: BacktestConfig) -> BacktestResult:
    """
    Execute one strategy over the panel.

    `panel` must already have been through prepare_panel() for this cfg.
    """
    rng = np.random.default_rng(cfg.seed)
    dates = np.sort(panel["date"].unique())
    if len(dates) < cfg.lookback_days + cfg.holding_days + 10:
        raise ValueError("not enough history")

    by_date = {d: g for d, g in panel.groupby("date", sort=True)}
    rebal_dates = dates[cfg.lookback_days::cfg.rebalance_days]

    rows, trade_rows = [], []
    prev_weights: dict[str, float] = {}
    n_signal_nan = 0

    for d in rebal_dates:
        # ---- information cutoff: the strategy sees NOTHING dated after d ----
        lo = d - np.timedelta64(cfg.lookback_days * 2, "D")
        hist = panel[(panel["date"] <= d) & (panel["date"] >= lo)]
        if hist.empty:
            continue

        # Hand the strategy a frame stripped of every forward-looking column.
        forbidden = ["fwd_ret", "next_open", "exit_open", "raw_next_open"]
        visible = hist.drop(columns=[c for c in forbidden if c in hist.columns])

        if cfg.use_entry_open_in_signal:
            # Legitimately observable before the entry decision: the entry
            # bar's OPEN only. Nothing else from that bar.
            nxt = by_date.get(dates[np.searchsorted(dates, d) + 1]) \
                if np.searchsorted(dates, d) + 1 < len(dates) else None
            if nxt is not None:
                visible = visible.merge(
                    nxt[["symbol", "open"]].rename(columns={"open": "entry_open"}),
                    on="symbol", how="left")

        sig = signal_fn(visible)
        if sig is None or len(sig) == 0:
            continue
        sig = sig.replace([np.inf, -np.inf], np.nan).dropna()
        n_signal_nan += int(sig.isna().sum())

        day = by_date[d]
        day = _eligible(day, cfg)
        day = day[day["symbol"].isin(sig.index)]
        if len(day) < 2 * cfg.min_names_per_side:
            continue
        day = day.assign(signal=day["symbol"].map(sig)).dropna(subset=["signal"])
        if len(day) < 2 * cfg.min_names_per_side:
            continue

        # ---------------- portfolio construction ----------------
        r = day["signal"].rank(pct=True, method="first")
        longs = day[r >= 1 - cfg.long_quantile]
        shorts = day[r <= cfg.short_quantile]

        if not cfg.long_only:
            shorts = shorts[shortable_mask(shorts, cfg.costs)]
        else:
            shorts = shorts.iloc[0:0]

        if len(longs) < cfg.min_names_per_side:
            continue
        if not cfg.long_only and len(shorts) < cfg.min_names_per_side:
            continue

        side_capital = cfg.capital * cfg.gross_leverage / (1 if cfg.long_only else 2)

        def _weights(sub: pd.DataFrame, sign: float) -> pd.DataFrame:
            if sub.empty:
                return sub.assign(target_dollars=[], w=[])
            per = side_capital / len(sub)
            cap = size_cap_dollars(sub["adv_dollar"], cfg.costs)
            dollars = np.minimum(per, cap)
            return sub.assign(target_dollars=sign * dollars)

        book = pd.concat([_weights(longs, 1.0), _weights(shorts, -1.0)])
        if book.empty or book["target_dollars"].abs().sum() == 0:
            continue

        # ---------------- costs on turnover ----------------
        new_w = dict(zip(book["symbol"], book["target_dollars"]))
        all_syms = set(new_w) | set(prev_weights)
        traded = pd.Series({s: new_w.get(s, 0.0) - prev_weights.get(s, 0.0)
                            for s in all_syms})
        traded = traded[traded.abs() > 1e-9]

        cost_ref = (panel[(panel["date"] == d) &
                          (panel["symbol"].isin(traded.index))]
                    .set_index("symbol")
                    .reindex(traded.index))
        one_way_bps = trade_cost_bps(
            price=cost_ref["close"], spread=cost_ref["spread_est"],
            vol=cost_ref["vol_est"], trade_dollars=traded,
            adv_dollar=cost_ref["adv_dollar"], cfg=cfg.costs,
        ).fillna(50.0)
        cost_dollars = float((traded.abs() * one_way_bps / 1e4).sum())

        # ---------------- realised P&L ----------------
        fwd = book["fwd_ret"]
        valid = fwd.notna()
        pnl_gross = float((book.loc[valid, "target_dollars"] * fwd[valid]).sum())

        # Borrow cost on the short leg for the holding period.
        short_notional = float(book.loc[book["target_dollars"] < 0,
                                        "target_dollars"].abs().sum())
        borrow = short_notional * cfg.costs.borrow_rate_annual * cfg.holding_days / 252

        gross_notional = float(book["target_dollars"].abs().sum())
        rows.append({
            "date": pd.Timestamp(d),
            "n_long": int((book["target_dollars"] > 0).sum()),
            "n_short": int((book["target_dollars"] < 0).sum()),
            "gross_notional": gross_notional,
            "turnover_dollars": float(traded.abs().sum()),
            "cost_dollars": cost_dollars + borrow,
            "pnl_gross": pnl_gross,
            "pnl_net": pnl_gross - cost_dollars - borrow,
            "gross_ret": pnl_gross / cfg.capital,
            "net_ret": (pnl_gross - cost_dollars - borrow) / cfg.capital,
            "avg_cost_bps": float(one_way_bps.mean()) if len(one_way_bps) else np.nan,
            "avg_spread_bps": float(cost_ref["spread_est"].mean() * 1e4)
                              if len(cost_ref) else np.nan,
            "coverage": float(valid.mean()),
        })

        for sym, row in book.iterrows():
            trade_rows.append({
                "date": pd.Timestamp(d), "symbol": row["symbol"],
                "side": "long" if row["target_dollars"] > 0 else "short",
                "dollars": row["target_dollars"], "fwd_ret": row["fwd_ret"],
                "close": row["close"], "adv_dollar": row["adv_dollar"],
                "spread_bps": row["spread_est"] * 1e4,
            })

        prev_weights = new_w

    daily = pd.DataFrame(rows)
    trades = pd.DataFrame(trade_rows)
    diag = {
        "n_rebalances": len(daily),
        "signal_nans": n_signal_nan,
        "mean_coverage": float(daily["coverage"].mean()) if len(daily) else np.nan,
        "mean_cost_bps": float(daily["avg_cost_bps"].mean()) if len(daily) else np.nan,
        "mean_spread_bps": float(daily["avg_spread_bps"].mean()) if len(daily) else np.nan,
    }
    return BacktestResult(cfg, daily, trades, diag)
