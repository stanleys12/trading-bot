"""
Order Executor Module
======================
Handles all Alpaca order submission and management.
Supports paper trading, extended hours, bracket orders.
"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from config import BotConfig, DEFAULT_CONFIG
from risk_manager import PositionRequest

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class ExecutedOrder:
    """Record of an executed order."""
    symbol: str
    direction: str
    strategy: str
    order_id: str
    status: OrderStatus
    shares: int
    entry_price: float
    stop_price: float
    target_price: Optional[float]
    fill_price: Optional[float] = None
    filled_at: Optional[datetime] = None
    stop_order_id: Optional[str] = None
    target_order_id: Optional[str] = None
    error: Optional[str] = None


class Executor:
    """
    Handles order submission to Alpaca.

    Features:
    - Paper trading support
    - Bracket orders (entry + stop-loss + take-profit)
    - Extended hours limit orders
    - Order status tracking
    """

    def __init__(self, config: BotConfig = DEFAULT_CONFIG,
                 trading_client=None):
        self.config = config
        self.trading_client = trading_client
        self.active_orders: dict[str, ExecutedOrder] = {}

    def execute_trade(self, request: PositionRequest) -> ExecutedOrder:
        """
        Execute an approved trade via Alpaca.

        For longs: submits a bracket order (entry + stop + optional target)
        For shorts: submits entry + separate stop order
        """
        if not request.approved:
            return ExecutedOrder(
                symbol=request.symbol,
                direction=request.direction,
                strategy=request.strategy,
                order_id="",
                status=OrderStatus.REJECTED,
                shares=0,
                entry_price=request.entry_price,
                stop_price=request.stop_price,
                target_price=request.target_price,
                error="Trade not approved by risk manager",
            )

        if self.trading_client is None:
            return self._simulate_execution(request)

        try:
            return self._execute_alpaca(request)
        except Exception as e:
            logger.error(f"Execution failed for {request.symbol}: {e}")
            return ExecutedOrder(
                symbol=request.symbol,
                direction=request.direction,
                strategy=request.strategy,
                order_id="",
                status=OrderStatus.FAILED,
                shares=request.position_size_shares,
                entry_price=request.entry_price,
                stop_price=request.stop_price,
                target_price=request.target_price,
                error=str(e),
            )

    def _execute_alpaca(self, request: PositionRequest) -> ExecutedOrder:
        """Execute via Alpaca API."""
        from alpaca.trading.requests import (
            MarketOrderRequest, LimitOrderRequest,
            StopLossRequest, TakeProfitRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

        side = OrderSide.BUY if request.direction == "long" else OrderSide.SELL

        # Determine if we're in extended hours
        now = datetime.now(ET)
        is_extended = (now.hour < 9 or (now.hour == 9 and now.minute < 30)
                       or now.hour >= 16)

        if is_extended:
            # Extended hours: must use limit orders
            order_data = LimitOrderRequest(
                symbol=request.symbol,
                qty=request.position_size_shares,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(request.entry_price, 2),
                extended_hours=True,
            )
        else:
            # Regular hours: use bracket order (entry + stop + optional target)
            stop_loss = StopLossRequest(stop_price=round(request.stop_price, 2))

            take_profit = None
            if request.target_price:
                take_profit = TakeProfitRequest(
                    limit_price=round(request.target_price, 2)
                )

            if take_profit:
                order_data = MarketOrderRequest(
                    symbol=request.symbol,
                    qty=request.position_size_shares,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.BRACKET,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )
            else:
                order_data = MarketOrderRequest(
                    symbol=request.symbol,
                    qty=request.position_size_shares,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.OTO,
                    stop_loss=stop_loss,
                )

        order = self.trading_client.submit_order(order_data)

        result = ExecutedOrder(
            symbol=request.symbol,
            direction=request.direction,
            strategy=request.strategy,
            order_id=str(order.id),
            status=OrderStatus.PENDING,
            shares=request.position_size_shares,
            entry_price=request.entry_price,
            stop_price=request.stop_price,
            target_price=request.target_price,
        )

        self.active_orders[request.symbol] = result
        logger.info(f"Order submitted: {request.direction} {request.position_size_shares} "
                     f"{request.symbol} | order_id={order.id}")

        return result

    def _simulate_execution(self, request: PositionRequest) -> ExecutedOrder:
        """Simulate execution for testing without Alpaca connection."""
        import uuid

        order_id = str(uuid.uuid4())[:8]

        result = ExecutedOrder(
            symbol=request.symbol,
            direction=request.direction,
            strategy=request.strategy,
            order_id=f"sim-{order_id}",
            status=OrderStatus.FILLED,
            shares=request.position_size_shares,
            entry_price=request.entry_price,
            stop_price=request.stop_price,
            target_price=request.target_price,
            fill_price=request.entry_price,
            filled_at=datetime.now(ET),
        )

        self.active_orders[request.symbol] = result
        logger.info(
            f"[SIMULATED] {request.direction.upper()} {request.position_size_shares} "
            f"{request.symbol} @ ${request.entry_price:.2f} | "
            f"stop=${request.stop_price:.2f}"
        )

        return result

    def close_position(self, symbol: str) -> bool:
        """Close an open position."""
        if self.trading_client is None:
            logger.info(f"[SIMULATED] Closed position: {symbol}")
            if symbol in self.active_orders:
                del self.active_orders[symbol]
            return True

        try:
            self.trading_client.close_position(symbol)
            logger.info(f"Closed position: {symbol}")
            if symbol in self.active_orders:
                del self.active_orders[symbol]
            return True
        except Exception as e:
            logger.error(f"Failed to close {symbol}: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Close all open positions (EOD or emergency)."""
        if self.trading_client is None:
            logger.info("[SIMULATED] Closed all positions")
            self.active_orders.clear()
            return True

        try:
            self.trading_client.close_all_positions(cancel_orders=True)
            logger.info("Closed all positions and cancelled open orders")
            self.active_orders.clear()
            return True
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return False

    def get_account_equity(self) -> float:
        """Get current account equity."""
        if self.trading_client is None:
            return self.config.risk.starting_capital

        try:
            account = self.trading_client.get_account()
            return float(account.equity)
        except Exception as e:
            logger.error(f"Failed to get account equity: {e}")
            return self.config.risk.starting_capital

    def get_open_positions(self) -> list:
        """Get all currently open positions."""
        if self.trading_client is None:
            return []

        try:
            return self.trading_client.get_all_positions()
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
