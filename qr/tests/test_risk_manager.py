"""Tests for the legacy src/ risk manager."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from risk_manager import PositionRequest, RiskManager


def test_consecutive_losses_trip_circuit_breaker():
    rm = RiskManager()
    for _ in range(rm.config.risk.max_consecutive_losses):
        rm.record_trade_result(-1.0)

    rm._trigger_circuit_breaker()

    assert rm.daily_state.is_circuit_broken
    assert rm.daily_state.circuit_break_until is not None
    assert rm.daily_state.consecutive_losses == 0

    request = rm.evaluate_trade(PositionRequest(
        symbol="ABC", direction="long", strategy="rvol_rocket",
        entry_price=10.0, stop_price=9.5,
    ))
    assert not request.approved
    assert request.rejection_reason.startswith("Circuit breaker active")
