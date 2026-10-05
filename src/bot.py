"""
Trading Bot - Main Orchestrator
=================================
The brain that coordinates all modules into a fully autonomous trading system.

Daily lifecycle:
  7:00 AM  → Pre-market scan begins
  9:25 AM  → Final scan + catalyst check + strategy evaluation
  9:30 AM  → Market opens, execute approved trades
  9:30-11  → Monitor positions, trail stops, manage risk
  11:00 AM → Close all positions (configurable)
  4:00 PM  → End-of-day summary, log performance, adapt

Adaptive features:
  - Detects when strategies stop working and disables them
  - Adjusts position sizing based on live performance
  - Optimizes RVOL thresholds from real execution data
  - Logs everything for future ML model training
"""

import logging
import sys
import time
import signal
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from enum import Enum
from pathlib import Path

from config import BotConfig, DEFAULT_CONFIG
from scanner import Scanner, GapCandidate
from catalyst import CatalystDetector
from strategy import StrategyEngine, TradeSignal
from risk_manager import RiskManager, PositionRequest
from executor import Executor
from journal import TradeJournal
from indicators import calc_spy_trend, calc_sma

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


class BotState(Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    ANALYZING = "analyzing"
    READY = "ready"
    TRADING = "trading"
    MONITORING = "monitoring"
    CLOSING = "closing"
    SHUTDOWN = "shutdown"


class TradingBot:
    """
    Fully autonomous trading bot.

    Implements a state machine:
    IDLE → SCANNING → ANALYZING → READY → TRADING → MONITORING → CLOSING → IDLE
    """

    def __init__(self, config: BotConfig = DEFAULT_CONFIG):
        self.config = config
        self.state = BotState.IDLE

        # Initialize all modules
        self.scanner = Scanner(config)
        self.catalyst = CatalystDetector(config)
        self.strategy = StrategyEngine(config)
        self.risk_mgr = RiskManager(config)
        self.executor = Executor(config)
        self.journal = TradeJournal(config)

        # Alpaca clients (initialized in connect())
        self.trading_client = None
        self.data_client = None

        # Daily state
        self.today_candidates: list[GapCandidate] = []
        self.today_signals: list[TradeSignal] = []
        self.today_trades: list[dict] = []
        self.spy_bullish = True
        self.is_running = False

        # Adaptive parameters (updated daily from journal)
        self.adaptive_params = {}

        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        log_dir = Path(self.config.logging.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.config.logging.log_level),
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(self.config.logging.log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def connect(self):
        """Connect to Alpaca API."""
        api_key = self.config.alpaca.api_key
        api_secret = self.config.alpaca.api_secret

        if not api_key or not api_secret:
            logger.warning(
                "No Alpaca API keys found — running in SIMULATION mode. "
                "Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables."
            )
            return

        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical.stock import StockHistoricalDataClient

            self.trading_client = TradingClient(
                api_key, api_secret,
                paper=self.config.alpaca.paper
            )
            self.data_client = StockHistoricalDataClient(api_key, api_secret)

            # Pass clients to modules
            self.scanner.trading_client = self.trading_client
            self.scanner.data_client = self.data_client
            self.executor.trading_client = self.trading_client

            account = self.trading_client.get_account()
            logger.info(
                f"Connected to Alpaca ({'PAPER' if self.config.alpaca.paper else 'LIVE'}). "
                f"Equity: ${float(account.equity):,.2f}"
            )

            # Update capital from actual account
            self.risk_mgr.daily_state.current_capital = float(account.equity)
            self.risk_mgr.daily_state.starting_capital = float(account.equity)

        except ImportError:
            logger.warning("alpaca-py not installed. Running in simulation mode.")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")

    def run(self):
        """
        Main run loop. Call this to start the bot.

        For paper trading / testing, this runs a single trading day cycle.
        For production, wrap this in a scheduler (APScheduler / cron).
        """
        self.is_running = True

        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("=" * 60)
        logger.info("🤖 TRADING BOT STARTED")
        logger.info(f"   Mode: {'PAPER' if self.config.alpaca.paper else '🔴 LIVE'}")
        logger.info(f"   Capital: ${self.risk_mgr.daily_state.current_capital:,.2f}")
        logger.info("=" * 60)

        try:
            self._load_adaptive_params()
            self._run_daily_cycle()
        except KeyboardInterrupt:
            logger.info("Bot interrupted by user")
        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
        finally:
            self._shutdown()

    def _run_daily_cycle(self):
        """Execute one full trading day cycle."""
        now = datetime.now(ET)
        logger.info(f"\n📅 Daily cycle starting: {now.strftime('%Y-%m-%d %H:%M ET')}")

        # Reset daily state
        self.risk_mgr.reset_daily()
        self.today_candidates = []
        self.today_signals = []
        self.today_trades = []

        # === PHASE 1: PRE-MARKET SCAN ===
        self.state = BotState.SCANNING
        logger.info("\n🔍 PHASE 1: Pre-market scan...")
        self.today_candidates = self._run_scan()

        if not self.today_candidates:
            logger.info("No candidates found. Nothing to trade today.")
            self._end_of_day()
            return

        # === PHASE 2: CATALYST & ANALYSIS ===
        self.state = BotState.ANALYZING
        logger.info(f"\n📰 PHASE 2: Analyzing {len(self.today_candidates)} candidates...")
        self._analyze_candidates()

        # === PHASE 3: STRATEGY EVALUATION ===
        logger.info("\n🎯 PHASE 3: Generating trade signals...")
        self.today_signals = self.strategy.evaluate_candidates(
            self.today_candidates, self.spy_bullish
        )

        if not self.today_signals:
            logger.info("No signals generated. Waiting for better setups.")
            self._end_of_day()
            return

        # === PHASE 4: RISK CHECK & EXECUTION ===
        self.state = BotState.TRADING
        logger.info(f"\n🚀 PHASE 4: Executing {len(self.today_signals)} signals...")
        self._execute_signals()

        # === PHASE 5: MONITORING ===
        self.state = BotState.MONITORING
        logger.info("\n📈 PHASE 5: Monitoring positions...")
        self._monitor_positions()

        # === PHASE 6: END OF DAY ===
        self.state = BotState.CLOSING
        self._end_of_day()

    def _load_adaptive_params(self):
        """Load adaptive parameters from trade journal."""
        self.adaptive_params = self.journal.get_adaptive_parameters()

        for strategy, params in self.adaptive_params.items():
            if isinstance(params, dict) and 'enabled' in params:
                if not params['enabled']:
                    logger.warning(
                        f"⚠️ Strategy '{strategy}' DISABLED by adaptive system: "
                        f"{params.get('reason', 'unknown')}"
                    )
                elif params.get('size_multiplier', 1.0) != 1.0:
                    logger.info(
                        f"📊 Strategy '{strategy}' size adjusted to "
                        f"{params['size_multiplier']:.1f}x: {params.get('reason', '')}"
                    )

        # Update RVOL threshold if adaptive system found a better one
        opt_rvol = self.adaptive_params.get('optimal_rvol_threshold')
        if opt_rvol and opt_rvol != self.config.scanner.min_rvol_long:
            logger.info(
                f"📊 RVOL threshold adapted: {self.config.scanner.min_rvol_long}x "
                f"→ {opt_rvol}x (from live performance)"
            )
            self.config.scanner.min_rvol_long = opt_rvol
            self.config.strategy.rvol_rocket_min_rvol = opt_rvol

    def _run_scan(self) -> list[GapCandidate]:
        """Run the pre-market scanner."""
        try:
            candidates = self.scanner.scan_for_gaps()

            # Enrich with fundamentals (market cap, float)
            if candidates:
                candidates = self.scanner.enrich_with_fundamentals(candidates)

            # Check SPY trend
            self._update_spy_trend()

            return candidates
        except Exception as e:
            logger.error(f"Scanner error: {e}", exc_info=True)
            return []

    def _update_spy_trend(self):
        """Check if SPY is in a bullish or bearish trend."""
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY").history(period="30d", auto_adjust=True)
            if len(spy) >= 20:
                sma20 = spy['Close'].rolling(20).mean().iloc[-1]
                self.spy_bullish = spy['Close'].iloc[-1] > sma20
                logger.info(
                    f"SPY trend: {'🟢 BULLISH' if self.spy_bullish else '🔴 BEARISH'} "
                    f"(${spy['Close'].iloc[-1]:.2f} vs SMA20=${sma20:.2f})"
                )
        except Exception as e:
            logger.warning(f"Could not determine SPY trend: {e}")
            self.spy_bullish = True  # Default to bullish

    def _analyze_candidates(self):
        """Run catalyst detection on all candidates."""
        for candidate in self.today_candidates:
            try:
                self.catalyst.enrich_candidate(candidate)
                logger.info(
                    f"  {candidate.symbol}: "
                    f"catalyst={'✅ ' + candidate.catalyst_type if candidate.has_catalyst else '❌ none'}, "
                    f"sentiment={candidate.sentiment_score:+.2f}"
                )
            except Exception as e:
                logger.error(f"Catalyst analysis failed for {candidate.symbol}: {e}")

        # Filter: check adaptive system for strategy health
        filtered = []
        for c in self.today_candidates:
            if c.strategy in self.adaptive_params:
                params = self.adaptive_params[c.strategy]
                if isinstance(params, dict) and not params.get('enabled', True):
                    logger.info(
                        f"  {c.symbol} skipped: {c.strategy} disabled by adaptive system"
                    )
                    continue
            filtered.append(c)

        self.today_candidates = filtered

    def _execute_signals(self):
        """Execute trade signals through risk manager and executor."""
        for signal in self.today_signals:
            if not self.is_running:
                break

            # Log signal (even if not traded)
            self.journal.log_signal({
                'symbol': signal.symbol,
                'strategy': signal.strategy,
                'direction': signal.direction,
                'gap_pct': signal.candidate.gap_pct,
                'relative_volume': signal.candidate.relative_volume,
                'setup_score': signal.setup_score,
                'was_traded': False,  # Updated below if traded
            })

            # Convert to position request
            request = self.strategy.signal_to_position_request(signal)

            # Apply adaptive size multiplier
            if signal.strategy in self.adaptive_params:
                params = self.adaptive_params[signal.strategy]
                if isinstance(params, dict):
                    multiplier = params.get('size_multiplier', 1.0)
                    if multiplier != 1.0:
                        request.setup_score *= multiplier

            # Risk check
            request = self.risk_mgr.evaluate_trade(request)

            if not request.approved:
                logger.info(
                    f"  ❌ {signal.symbol} rejected: {request.rejection_reason}"
                )
                continue

            # Execute
            result = self.executor.execute_trade(request)

            if result.status.value in ('filled', 'pending'):
                self.risk_mgr.position_opened()
                self.today_trades.append({
                    'signal': signal,
                    'request': request,
                    'result': result,
                    'entry_time': datetime.now(ET),
                })
                logger.info(
                    f"  ✅ {signal.direction.upper()} {signal.symbol}: "
                    f"{request.position_size_shares} shares @ ${request.entry_price:.2f} "
                    f"| {signal.reasoning[:80]}..."
                )

    def _monitor_positions(self):
        """
        Monitor open positions.

        In live mode, this would run in a loop checking prices via WebSocket.
        For this version, we do a simplified check.
        """
        if not self.today_trades:
            return

        logger.info(f"Monitoring {len(self.today_trades)} positions...")

        # Check if it's time to close
        if self.risk_mgr.should_close_all():
            logger.info("⏰ Time to close all positions")
            self._close_all_and_log()
            return

        # In simulation mode, positions are already closed at EOD
        # In live mode, this would be a continuous monitoring loop

    def _close_all_and_log(self):
        """Close all positions and log the results."""
        self.executor.close_all_positions()

        for trade in self.today_trades:
            signal = trade['signal']
            request = trade['request']
            result = trade['result']

            # Get actual P&L (in live mode, from Alpaca; in sim, estimate)
            entry_price = result.fill_price or request.entry_price
            exit_price = entry_price  # In simulation, would be actual exit

            # Log the trade
            self.journal.log_trade({
                'symbol': signal.symbol,
                'strategy': signal.strategy,
                'direction': signal.direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'stop_price': request.stop_price,
                'target_price': request.target_price,
                'shares': request.position_size_shares,
                'pnl_dollars': 0,  # Would be calculated from actual fill
                'pnl_pct': 0,
                'gap_pct': signal.candidate.gap_pct,
                'relative_volume': signal.candidate.relative_volume,
                'atr_pct': signal.candidate.atr_pct,
                'prior_5d_return': signal.candidate.prior_5d_return,
                'rsi': signal.candidate.rsi,
                'spy_bullish': self.spy_bullish,
                'has_catalyst': signal.candidate.has_catalyst,
                'catalyst_type': signal.candidate.catalyst_type,
                'catalyst_weight': signal.candidate.catalyst_weight,
                'sentiment_score': signal.candidate.sentiment_score,
                'setup_score': signal.setup_score,
                'day_of_week': datetime.now(ET).weekday(),
                'reasoning': signal.reasoning,
            })

            self.risk_mgr.position_closed()

    def _end_of_day(self):
        """End-of-day procedures."""
        logger.info("\n📊 END OF DAY SUMMARY")
        logger.info("=" * 50)

        # Close any remaining positions
        if self.executor.active_orders:
            self._close_all_and_log()

        # Get summary
        summary = self.risk_mgr.get_daily_summary()
        cum_stats = self.journal.get_cumulative_stats()

        logger.info(f"  Date: {summary['date']}")
        logger.info(f"  Trades: {summary['trades_taken']}")
        logger.info(f"  P&L: ${summary['realized_pnl']:+.2f} ({summary['pnl_pct']:+.2f}%)")
        logger.info(f"  Capital: ${summary['current_capital']:,.2f}")

        if cum_stats['total_trades'] > 0:
            logger.info(f"\n  📈 ALL-TIME STATS:")
            logger.info(f"    Total trades: {cum_stats['total_trades']}")
            logger.info(f"    Win rate: {cum_stats['win_rate']:.1f}%")
            logger.info(f"    Total P&L: ${cum_stats['total_pnl']:+,.2f}")

        # Check strategy health
        logger.info(f"\n  🏥 STRATEGY HEALTH CHECK:")
        for strat in ['rvol_rocket', 'dead_cat_fade', 'panic_bounce']:
            health = self.journal.get_strategy_health(strat)
            status_emoji = {
                'no_data': '⬜', 'active': '🟢', 'disabled': '🔴'
            }.get(health.get('status', 'no_data'), '⬜')

            if health.get('win_rate') is not None:
                logger.info(
                    f"    {status_emoji} {strat}: "
                    f"WR={health['win_rate']*100:.1f}%, "
                    f"Avg={health['avg_return']:.2f}%, "
                    f"PF={health.get('profit_factor', 0):.2f}, "
                    f"{'⚠️ DECAYING' if health.get('is_decaying') else '✅ stable'}"
                )
            else:
                logger.info(f"    {status_emoji} {strat}: No data yet")

        # Log daily summary to DB
        self.journal.log_daily_summary(summary)

        # Check if ML training data threshold reached
        total_trades = self.journal.get_trade_count()
        if total_trades >= 200 and total_trades % 50 == 0:
            logger.info(
                f"\n  🧠 ML TRAINING READY: {total_trades} trades logged. "
                f"Consider training the ML signal filter (Phase C)."
            )

        logger.info("=" * 50)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("\n⚠️ Shutdown signal received — closing all positions...")
        self.is_running = False
        self.state = BotState.SHUTDOWN

    def _shutdown(self):
        """Clean shutdown."""
        if self.executor.active_orders:
            logger.info("Closing remaining positions...")
            self.executor.close_all_positions()

        logger.info("🤖 Bot shutdown complete.")
        self.state = BotState.IDLE


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Small-Cap Gap Trading Bot")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run scanner only, don't execute trades")
    parser.add_argument("--paper", action="store_true", default=True,
                        help="Use paper trading (default)")
    parser.add_argument("--capital", type=float, default=30000,
                        help="Starting capital (default: $30,000)")
    args = parser.parse_args()

    # Configure
    config = BotConfig()
    config.alpaca.paper = args.paper
    config.risk.starting_capital = args.capital

    # Create and run bot
    bot = TradingBot(config)
    bot.connect()

    if args.dry_run:
        logger.info("DRY RUN MODE — scanning only, no trades")
        candidates = bot._run_scan()
        bot._update_spy_trend()
        for c in candidates:
            bot.catalyst.enrich_candidate(c)
        signals = bot.strategy.evaluate_candidates(candidates, bot.spy_bullish)
        logger.info(f"\nWould have generated {len(signals)} signals")
        for s in signals:
            logger.info(f"  {s.direction.upper()} {s.symbol} | {s.reasoning}")
    else:
        bot.run()


if __name__ == "__main__":
    main()
