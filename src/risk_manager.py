"""
Risk Manager Module
====================
Enforces position sizing, daily loss limits, and circuit breakers.
This module has VETO POWER over all trades — no trade executes without approval.
"""

import logging
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from typing import Optional

from config import BotConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


@dataclass
class PositionRequest:
    """A request to open a position (pending risk approval)."""
    symbol: str
    direction: str          # "long" or "short"
    strategy: str           # "rvol_rocket", "dead_cat_fade", "panic_bounce"
    entry_price: float
    stop_price: float
    target_price: Optional[float] = None
    setup_score: float = 0.0

    # Calculated by risk manager
    approved: bool = False
    position_size_shares: int = 0
    position_size_dollars: float = 0.0
    risk_amount: float = 0.0
    rejection_reason: Optional[str] = None


@dataclass
class DailyState:
    """Tracks the bot's daily P&L and activity."""
    date: date = field(default_factory=lambda: datetime.now(ET).date())
    starting_capital: float = 0.0
    current_capital: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades_taken: int = 0
    consecutive_losses: int = 0
    is_circuit_broken: bool = False
    circuit_break_until: Optional[datetime] = None
    open_positions: int = 0


class RiskManager:
    """
    Risk management engine.

    Every trade request goes through here before execution.
    The risk manager can REJECT any trade that violates risk rules.
    """

    def __init__(self, config: BotConfig = DEFAULT_CONFIG):
        self.config = config
        self.daily_state = DailyState(
            starting_capital=config.risk.starting_capital,
            current_capital=config.risk.starting_capital,
        )
        self._trade_results = []  # Track for Kelly calculation

    def evaluate_trade(self, request: PositionRequest) -> PositionRequest:
        """
        Evaluate a trade request against all risk rules.
        Returns the request with approved=True/False and calculated sizing.
        """
        cfg = self.config.risk

        # === CHECK 1: Circuit breaker ===
        if self.daily_state.is_circuit_broken:
            now = datetime.now(ET)
            if (self.daily_state.circuit_break_until
                    and now < self.daily_state.circuit_break_until):
                request.rejection_reason = (
                    f"Circuit breaker active until "
                    f"{self.daily_state.circuit_break_until.strftime('%H:%M')}"
                )
                logger.warning(f"REJECTED {request.symbol}: {request.rejection_reason}")
                return request
            else:
                # Circuit breaker expired
                self.daily_state.is_circuit_broken = False
                self.daily_state.circuit_break_until = None
                logger.info("Circuit breaker expired — resuming trading")

        # === CHECK 2: Daily loss limit ===
        daily_loss_pct = (self.daily_state.realized_pnl / self.daily_state.starting_capital) * 100
        if daily_loss_pct <= -cfg.max_daily_loss_pct:
            request.rejection_reason = (
                f"Daily loss limit hit: {daily_loss_pct:.1f}% "
                f"(limit: -{cfg.max_daily_loss_pct}%)"
            )
            logger.warning(f"REJECTED {request.symbol}: {request.rejection_reason}")
            return request

        # === CHECK 3: Max trades per day ===
        if self.daily_state.trades_taken >= cfg.max_trades_per_day:
            request.rejection_reason = (
                f"Max trades reached: {self.daily_state.trades_taken}/{cfg.max_trades_per_day}"
            )
            logger.warning(f"REJECTED {request.symbol}: {request.rejection_reason}")
            return request

        # === CHECK 4: Max concurrent positions ===
        if self.daily_state.open_positions >= cfg.max_concurrent_positions:
            request.rejection_reason = (
                f"Max concurrent positions: {self.daily_state.open_positions}/"
                f"{cfg.max_concurrent_positions}"
            )
            logger.warning(f"REJECTED {request.symbol}: {request.rejection_reason}")
            return request

        # === CHECK 5: Trading hours ===
        now = datetime.now(ET)
        close_time = now.replace(
            hour=cfg.close_all_by_hour,
            minute=cfg.close_all_by_minute,
            second=0,
        )
        if now >= close_time:
            request.rejection_reason = (
                f"Past closing time ({cfg.close_all_by_hour}:"
                f"{cfg.close_all_by_minute:02d} ET)"
            )
            logger.warning(f"REJECTED {request.symbol}: {request.rejection_reason}")
            return request

        # === CHECK 6: Consecutive losses ===
        if self.daily_state.consecutive_losses >= cfg.max_consecutive_losses:
            self._trigger_circuit_breaker()
            request.rejection_reason = (
                f"Consecutive losses: {self.daily_state.consecutive_losses}"
            )
            logger.warning(f"REJECTED {request.symbol}: {request.rejection_reason}")
            return request

        # === POSITION SIZING ===
        position_size = self._calculate_position_size(request)
        if position_size <= 0:
            request.rejection_reason = "Position size calculated as zero"
            return request

        request.position_size_shares = position_size
        request.position_size_dollars = position_size * request.entry_price
        request.risk_amount = abs(request.entry_price - request.stop_price) * position_size
        request.approved = True

        logger.info(
            f"APPROVED {request.symbol} ({request.direction}): "
            f"{position_size} shares @ ${request.entry_price:.2f}, "
            f"stop=${request.stop_price:.2f}, "
            f"risk=${request.risk_amount:.2f} "
            f"({request.risk_amount/self.daily_state.current_capital*100:.2f}%)"
        )

        return request

    def _calculate_position_size(self, request: PositionRequest) -> int:
        """
        Calculate position size using ATR-based sizing + fractional Kelly.

        Position Size = Risk Amount / Stop Distance
        Risk Amount = min(Kelly fraction, hard cap) * Account Equity
        """
        cfg = self.config.risk
        capital = self.daily_state.current_capital

        # Determine risk percentage based on direction
        if request.direction == "short":
            base_risk_pct = cfg.max_risk_per_trade_short_pct / 100
        else:
            base_risk_pct = cfg.max_risk_per_trade_pct / 100

        # Try Kelly criterion if we have enough history
        kelly_risk_pct = self._get_kelly_risk()
        if kelly_risk_pct is not None:
            risk_pct = min(kelly_risk_pct, base_risk_pct)
        else:
            risk_pct = base_risk_pct

        # Scale by setup score (higher conviction = larger position)
        score_multiplier = 0.5 + (request.setup_score / 100) * 0.5  # 0.5x to 1.0x
        risk_pct *= score_multiplier

        risk_amount = capital * risk_pct

        # Stop distance
        stop_distance = abs(request.entry_price - request.stop_price)
        if stop_distance <= 0:
            return 0

        # Raw position size
        shares = int(risk_amount / stop_distance)

        # === HARD CAPS ===

        # Cap 1: Max % of capital
        max_dollars = capital * (cfg.max_position_pct / 100)
        max_shares_by_capital = int(max_dollars / request.entry_price) if request.entry_price > 0 else 0
        shares = min(shares, max_shares_by_capital)

        # Cap 2: Minimum 1 share
        shares = max(shares, 0)

        return shares

    def _get_kelly_risk(self) -> Optional[float]:
        """
        Calculate fractional Kelly criterion from trade history.
        Returns None if not enough history.
        """
        cfg = self.config.risk

        if len(self._trade_results) < cfg.kelly_min_trades:
            return None

        # Use most recent N trades
        recent = self._trade_results[-cfg.kelly_lookback:]

        wins = [r for r in recent if r > 0]
        losses = [r for r in recent if r <= 0]

        if not losses:
            return cfg.max_risk_per_trade_pct / 100

        win_rate = len(wins) / len(recent)
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 1

        if avg_loss == 0:
            return cfg.max_risk_per_trade_pct / 100

        # Kelly formula
        reward_risk_ratio = avg_win / avg_loss
        kelly = (win_rate * (reward_risk_ratio + 1) - 1) / reward_risk_ratio

        # Apply fraction (Quarter-Kelly by default)
        fractional_kelly = max(0, kelly * cfg.kelly_fraction)

        # Hard cap
        return min(fractional_kelly, cfg.max_risk_per_trade_pct / 100)

    def record_trade_result(self, pnl_pct: float):
        """Record a trade result for Kelly calculation and daily tracking."""
        self._trade_results.append(pnl_pct)

        if pnl_pct > 0:
            self.daily_state.consecutive_losses = 0
        else:
            self.daily_state.consecutive_losses += 1

        self.daily_state.realized_pnl += pnl_pct
        self.daily_state.trades_taken += 1

    def update_capital(self, new_capital: float):
        """Update current capital (called after trade execution)."""
        self.daily_state.current_capital = new_capital

    def position_opened(self):
        """Track that a new position was opened."""
        self.daily_state.open_positions += 1

    def position_closed(self):
        """Track that a position was closed."""
        self.daily_state.open_positions = max(0, self.daily_state.open_positions - 1)

    def _trigger_circuit_breaker(self):
        """Activate the circuit breaker — pause trading."""
        cfg = self.config.risk
        now = datetime.now(ET)
        resume_time = now + timedelta(minutes=cfg.pause_minutes_after_breaker)

        self.daily_state.is_circuit_broken = True
        self.daily_state.circuit_break_until = resume_time
        self.daily_state.consecutive_losses = 0  # Reset counter

        logger.warning(
            f"🚨 CIRCUIT BREAKER ACTIVATED — "
            f"Pausing until {resume_time.strftime('%H:%M ET')}"
        )

    def reset_daily(self):
        """Reset daily state for a new trading day."""
        capital = self.daily_state.current_capital
        self.daily_state = DailyState(
            starting_capital=capital,
            current_capital=capital,
        )
        logger.info(f"Daily state reset. Capital: ${capital:,.2f}")

    def should_close_all(self) -> bool:
        """Check if it's time to close all positions."""
        now = datetime.now(ET)
        cfg = self.config.risk

        close_time = now.replace(
            hour=cfg.close_all_by_hour,
            minute=cfg.close_all_by_minute,
            second=0,
        )
        return now >= close_time

    def get_daily_summary(self) -> dict:
        """Get a summary of today's trading activity."""
        state = self.daily_state
        return {
            'date': str(state.date),
            'starting_capital': state.starting_capital,
            'current_capital': state.current_capital,
            'realized_pnl': state.realized_pnl,
            'trades_taken': state.trades_taken,
            'pnl_pct': (state.realized_pnl / state.starting_capital * 100)
                       if state.starting_capital > 0 else 0,
            'circuit_broken': state.is_circuit_broken,
        }
