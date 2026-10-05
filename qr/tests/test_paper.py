"""Tests for the paper-trading harness."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.execution import paper
from qr.execution.paper import PaperTrader


def _day(symbols, price=20.0):
    return pd.DataFrame({
        "symbol": symbols,
        "close": price,
        "spread_est": 0.004,
        "vol_est": 0.03,
        "adv_dollar": 5e7,
    })


def test_live_trading_is_disabled():
    assert paper.LIVE_TRADING_ENABLED is False
    assert not hasattr(paper, "submit_live_order")


def test_orders_move_the_portfolio_and_charge_cost():
    with tempfile.TemporaryDirectory() as d:
        pt = PaperTrader(Path(d) / "p.db", capital=100_000.0)
        day = _day(["AAA", "BBB", "CCC"])
        targets = pd.Series({"AAA": 10_000.0, "BBB": -10_000.0})

        orders = pt.generate_orders(day, targets, "2026-01-05")
        assert len(orders) == 2
        assert all(o.est_cost_bps > 0 for o in orders)

        pt.simulate_fills(orders, pd.Series({"AAA": 20.0, "BBB": 20.0, "CCC": 20.0}))
        assert set(pt.pf.positions) == {"AAA", "BBB"}
        assert pt.pf.positions["AAA"].dollars == pytest.approx(10_000.0)
        assert pt.pf.positions["BBB"].dollars == pytest.approx(-10_000.0)
        assert pt.pf.cash < 100_000.0, "no transaction cost was charged"
        assert pt.pf.gross_exposure() == pytest.approx(20_000.0)
        assert pt.pf.net_exposure() == pytest.approx(0.0)


def test_closing_a_position_removes_it():
    with tempfile.TemporaryDirectory() as d:
        pt = PaperTrader(Path(d) / "p.db")
        day = _day(["AAA"])
        px = pd.Series({"AAA": 20.0})
        pt.simulate_fills(
            pt.generate_orders(day, pd.Series({"AAA": 5_000.0}), "2026-01-05"), px)
        assert "AAA" in pt.pf.positions
        pt.simulate_fills(
            pt.generate_orders(day, pd.Series(dtype=float), "2026-01-06"), px)
        assert "AAA" not in pt.pf.positions


def test_missing_market_data_rejects_rather_than_guesses():
    with tempfile.TemporaryDirectory() as d:
        pt = PaperTrader(Path(d) / "p.db")
        day = _day(["AAA"])
        orders = pt.generate_orders(day, pd.Series({"ZZZ": 5_000.0}), "2026-01-05")
        assert orders[0].status == "rejected"
        assert "no market data" in orders[0].note


def test_journal_and_divergence_report():
    with tempfile.TemporaryDirectory() as d:
        pt = PaperTrader(Path(d) / "p.db")
        day = _day(["AAA", "BBB"])
        px = pd.Series({"AAA": 20.0, "BBB": 20.0})
        for i, date in enumerate(["2026-01-05", "2026-01-12", "2026-01-20"]):
            tgt = pd.Series({"AAA": 10_000.0 * (1 + i * 0.1), "BBB": -8_000.0})
            o = pt.simulate_fills(pt.generate_orders(day, tgt, date), px)
            pt.journal(o)
            pt.record_equity(pt.pf.mark_to_market(px, date))

        rep = pt.divergence_report({"mean_period_ret": 0.01, "avg_cost_bps": 20.0,
                                    "vol_annual": 0.15})
        assert not rep.empty
        assert set(rep["metric"]) >= {"avg_cost_bps", "fill_rate"}
        assert rep["divergence"].notna().any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
