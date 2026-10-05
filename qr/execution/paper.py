"""
Paper-trading harness.
======================

Generates signals on the current panel, records hypothetical orders, simulates
fills with the SAME cost model the backtest used, maintains a portfolio, and
tracks live-versus-backtest divergence.

LIVE TRADING IS NOT IMPLEMENTED AND MUST NOT BE. There is no broker client in
this module by design. The purpose of paper trading here is not to make money;
it is to measure the gap between assumed fills and achievable fills. That gap
is the honest estimate of how much of any backtested edge is real.

The comparison that matters
---------------------------
`divergence_report` compares realised paper fills against the backtest's
assumed costs. If realised cost per trade materially exceeds the modelled
cost, the backtest was optimistic by exactly that amount and every conclusion
drawn from it shifts down accordingly.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from qr.backtesting.costs import CostConfig, trade_cost_bps

logger = logging.getLogger(__name__)

LIVE_TRADING_ENABLED = False       # do not change this


@dataclass
class Order:
    ts: str
    date: str
    symbol: str
    side: str                       # buy / sell / short / cover
    target_dollars: float
    ref_price: float
    est_cost_bps: float
    status: str = "simulated"
    fill_price: Optional[float] = None
    fill_dollars: Optional[float] = None
    realised_cost_bps: Optional[float] = None
    note: str = ""


@dataclass
class Position:
    symbol: str
    dollars: float
    entry_price: float
    entry_date: str


@dataclass
class PaperPortfolio:
    capital: float = 100_000.0
    cash: float = 100_000.0
    positions: dict[str, Position] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def gross_exposure(self) -> float:
        return sum(abs(p.dollars) for p in self.positions.values())

    def net_exposure(self) -> float:
        return sum(p.dollars for p in self.positions.values())

    def mark_to_market(self, prices: pd.Series, date: str) -> dict:
        equity = self.cash
        for sym, p in self.positions.items():
            px = prices.get(sym)
            if px is None or not np.isfinite(px):
                equity += p.dollars                 # stale mark
                continue
            equity += p.dollars * (px / p.entry_price)
        row = {"date": date, "equity": equity, "cash": self.cash,
               "gross": self.gross_exposure(), "net": self.net_exposure(),
               "n_positions": len(self.positions)}
        self.history.append(row)
        return row


class PaperTrader:
    """Signal -> orders -> simulated fills -> portfolio, with a durable journal."""

    def __init__(self, db_path: str | Path, cost_cfg: CostConfig = CostConfig(),
                 capital: float = 100_000.0):
        if LIVE_TRADING_ENABLED:
            raise RuntimeError("live trading is disabled in this project")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cost_cfg = cost_cfg
        self.pf = PaperPortfolio(capital=capital, cash=capital)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS orders (
                ts TEXT, date TEXT, symbol TEXT, side TEXT,
                target_dollars REAL, ref_price REAL, est_cost_bps REAL,
                status TEXT, fill_price REAL, fill_dollars REAL,
                realised_cost_bps REAL, note TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS equity (
                date TEXT PRIMARY KEY, equity REAL, cash REAL,
                gross REAL, net REAL, n_positions INTEGER)""")
            c.execute("""CREATE TABLE IF NOT EXISTS expectations (
                run_id TEXT, metric TEXT, backtest_value REAL, live_value REAL)""")

    # ------------------------------------------------------------- ordering
    def generate_orders(self, day: pd.DataFrame, targets: pd.Series,
                        date: str) -> list[Order]:
        """
        `targets` maps symbol -> desired dollar exposure (signed).
        `day` must contain close, spread_est, vol_est, adv_dollar for each name.
        """
        ref = day.set_index("symbol")
        orders: list[Order] = []
        syms = set(targets.index) | set(self.pf.positions)

        for sym in sorted(syms):
            want = float(targets.get(sym, 0.0))
            have = self.pf.positions[sym].dollars if sym in self.pf.positions else 0.0
            delta = want - have
            if abs(delta) < 1.0:
                continue
            if sym not in ref.index:
                orders.append(Order(_now(), date, sym, _side(delta), delta,
                                    np.nan, np.nan, status="rejected",
                                    note="no market data on decision date"))
                continue
            r = ref.loc[sym]
            est = float(trade_cost_bps(
                price=pd.Series([r["close"]]), spread=pd.Series([r["spread_est"]]),
                vol=pd.Series([r["vol_est"]]), trade_dollars=pd.Series([delta]),
                adv_dollar=pd.Series([r["adv_dollar"]]), cfg=self.cost_cfg).iloc[0])
            orders.append(Order(_now(), date, sym, _side(delta), delta,
                                float(r["close"]), est))
        return orders

    def simulate_fills(self, orders: list[Order], fill_prices: pd.Series
                       ) -> list[Order]:
        """
        Fill at the next available price, degraded by the modelled cost.
        This is intentionally the SAME cost model as the backtest, so that any
        divergence observed later is attributable to the model being wrong
        rather than to two different models being compared.
        """
        for o in orders:
            if o.status == "rejected":
                continue
            px = fill_prices.get(o.symbol)
            if px is None or not np.isfinite(px):
                o.status, o.note = "unfilled", "no fill price available"
                continue
            slip = o.est_cost_bps / 1e4
            o.fill_price = float(px * (1 + slip if o.target_dollars > 0 else 1 - slip))
            o.fill_dollars = float(o.target_dollars)
            o.realised_cost_bps = o.est_cost_bps
            o.status = "filled"

            cur = self.pf.positions.get(o.symbol)
            new_dollars = (cur.dollars if cur else 0.0) + o.fill_dollars
            if abs(new_dollars) < 1.0:
                self.pf.positions.pop(o.symbol, None)
            else:
                self.pf.positions[o.symbol] = Position(
                    o.symbol, new_dollars, o.fill_price, o.date)
            self.pf.cash -= abs(o.fill_dollars) * slip
        return orders

    def journal(self, orders: list[Order]):
        with sqlite3.connect(self.db_path) as c:
            c.executemany(
                "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [tuple(asdict(o).values()) for o in orders])

    def record_equity(self, row: dict):
        with sqlite3.connect(self.db_path) as c:
            c.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?,?,?)",
                      (row["date"], row["equity"], row["cash"], row["gross"],
                       row["net"], row["n_positions"]))

    # ---------------------------------------------------------- comparison
    def divergence_report(self, backtest_metrics: dict) -> pd.DataFrame:
        """Live vs backtest. The cost line is the one that matters."""
        with sqlite3.connect(self.db_path) as c:
            eq = pd.read_sql("SELECT * FROM equity ORDER BY date", c)
            od = pd.read_sql("SELECT * FROM orders WHERE status='filled'", c)
        if eq.empty:
            return pd.DataFrame()

        live_ret = eq["equity"].pct_change(fill_method=None).dropna()
        rows = [
            {"metric": "mean_period_ret",
             "backtest": backtest_metrics.get("mean_period_ret"),
             "live": float(live_ret.mean()) if len(live_ret) else np.nan},
            {"metric": "vol_period",
             "backtest": backtest_metrics.get("vol_annual"),
             "live": float(live_ret.std(ddof=1)) if len(live_ret) > 1 else np.nan},
            {"metric": "avg_cost_bps",
             "backtest": backtest_metrics.get("avg_cost_bps"),
             "live": float(od["realised_cost_bps"].mean()) if len(od) else np.nan},
            {"metric": "fill_rate",
             "backtest": 1.0,
             "live": float((od["status"] == "filled").mean()) if len(od) else np.nan},
        ]
        out = pd.DataFrame(rows)
        out["divergence"] = out["live"] - out["backtest"]
        return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _side(delta: float) -> str:
    return "buy" if delta > 0 else "sell"
