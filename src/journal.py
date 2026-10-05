"""
Trade Journal Module
=====================
SQLite-backed trade journal that tracks every decision.
Powers the adaptive learning system — the bot gets smarter over time.

This is the BRAIN of the adaptive system:
- Logs every trade with full context (features + outcome)
- Calculates rolling statistics to detect edge decay
- Recalculates Kelly criterion from live results
- Triggers strategy parameter adjustments when performance drifts
"""

import sqlite3
import logging
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional

import numpy as np

from config import BotConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


class TradeJournal:
    """
    Persistent trade journal with adaptive learning capabilities.

    The journal doesn't just record — it LEARNS:
    1. Tracks rolling win rates per strategy
    2. Detects when an edge is decaying (below threshold)
    3. Calculates optimal position sizing from real results
    4. Logs feature vectors for future ML training (Phase C)
    """

    def __init__(self, config: BotConfig = DEFAULT_CONFIG):
        self.config = config
        db_path = Path(config.logging.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main trade log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                stop_price REAL,
                target_price REAL,
                shares INTEGER,
                pnl_dollars REAL,
                pnl_pct REAL,
                won INTEGER,  -- 1 = win, 0 = loss
                
                -- Context features (for ML training later)
                gap_pct REAL,
                relative_volume REAL,
                atr_pct REAL,
                prior_5d_return REAL,
                rsi REAL,
                spy_bullish INTEGER,
                has_catalyst INTEGER,
                catalyst_type TEXT,
                catalyst_weight INTEGER,
                sentiment_score REAL,
                setup_score REAL,
                day_of_week INTEGER,
                
                -- Execution quality
                slippage_pct REAL,
                hold_time_minutes REAL,
                
                -- Meta
                reasoning TEXT,
                notes TEXT
            )
        """)

        # Daily performance summary
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT PRIMARY KEY,
                starting_capital REAL,
                ending_capital REAL,
                realized_pnl REAL,
                trades_taken INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                best_trade REAL,
                worst_trade REAL,
                strategies_used TEXT  -- JSON list
            )
        """)

        # Strategy performance tracking (rolling windows)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy TEXT NOT NULL,
                window_size INTEGER,  -- number of trades in window
                win_rate REAL,
                avg_return REAL,
                avg_win REAL,
                avg_loss REAL,
                profit_factor REAL,
                kelly_fraction REAL,
                edge_score REAL,  -- composite score for edge strength
                is_edge_decaying INTEGER  -- 1 if below threshold
            )
        """)

        # Signals log (ALL signals, even those not traded)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                direction TEXT NOT NULL,
                gap_pct REAL,
                relative_volume REAL,
                setup_score REAL,
                was_traded INTEGER,  -- 1 if actually traded
                rejection_reason TEXT,
                
                -- Outcome (even for non-traded signals, track what would have happened)
                hypothetical_return REAL
            )
        """)

        conn.commit()
        conn.close()

    def log_trade(self, trade_data: dict):
        """Log a completed trade with full context."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO trades (
                timestamp, symbol, strategy, direction,
                entry_price, exit_price, stop_price, target_price,
                shares, pnl_dollars, pnl_pct, won,
                gap_pct, relative_volume, atr_pct, prior_5d_return,
                rsi, spy_bullish, has_catalyst, catalyst_type,
                catalyst_weight, sentiment_score, setup_score,
                day_of_week, slippage_pct, hold_time_minutes,
                reasoning, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_data.get('timestamp', datetime.now(ET).isoformat()),
            trade_data['symbol'],
            trade_data['strategy'],
            trade_data['direction'],
            trade_data.get('entry_price'),
            trade_data.get('exit_price'),
            trade_data.get('stop_price'),
            trade_data.get('target_price'),
            trade_data.get('shares'),
            trade_data.get('pnl_dollars', 0),
            trade_data.get('pnl_pct', 0),
            1 if trade_data.get('pnl_dollars', 0) > 0 else 0,
            trade_data.get('gap_pct'),
            trade_data.get('relative_volume'),
            trade_data.get('atr_pct'),
            trade_data.get('prior_5d_return'),
            trade_data.get('rsi'),
            1 if trade_data.get('spy_bullish') else 0,
            1 if trade_data.get('has_catalyst') else 0,
            trade_data.get('catalyst_type'),
            trade_data.get('catalyst_weight', 0),
            trade_data.get('sentiment_score', 0),
            trade_data.get('setup_score', 0),
            trade_data.get('day_of_week'),
            trade_data.get('slippage_pct'),
            trade_data.get('hold_time_minutes'),
            trade_data.get('reasoning'),
            trade_data.get('notes'),
        ))

        conn.commit()
        conn.close()

        logger.info(
            f"Trade logged: {trade_data['direction']} {trade_data['symbol']} "
            f"via {trade_data['strategy']} | "
            f"P&L: ${trade_data.get('pnl_dollars', 0):+.2f} "
            f"({trade_data.get('pnl_pct', 0):+.2f}%)"
        )

        # Update rolling performance after each trade
        self._update_strategy_performance(trade_data['strategy'])

    def log_signal(self, signal_data: dict):
        """Log a signal (traded or not) for complete record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO signals (
                timestamp, symbol, strategy, direction,
                gap_pct, relative_volume, setup_score,
                was_traded, rejection_reason, hypothetical_return
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_data.get('timestamp', datetime.now(ET).isoformat()),
            signal_data['symbol'],
            signal_data['strategy'],
            signal_data['direction'],
            signal_data.get('gap_pct'),
            signal_data.get('relative_volume'),
            signal_data.get('setup_score'),
            1 if signal_data.get('was_traded') else 0,
            signal_data.get('rejection_reason'),
            signal_data.get('hypothetical_return'),
        ))

        conn.commit()
        conn.close()

    def _update_strategy_performance(self, strategy: str):
        """Calculate and store rolling performance metrics for a strategy."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get recent trades for this strategy
        cursor.execute("""
            SELECT pnl_pct, won FROM trades
            WHERE strategy = ?
            ORDER BY timestamp DESC
            LIMIT 50
        """, (strategy,))

        rows = cursor.fetchall()
        if len(rows) < 10:  # Need minimum sample
            conn.close()
            return

        pnl_pcts = [r[0] for r in rows if r[0] is not None]
        wins = [r[0] for r in rows if r[0] is not None and r[0] > 0]
        losses = [r[0] for r in rows if r[0] is not None and r[0] <= 0]

        win_rate = len(wins) / len(pnl_pcts) if pnl_pcts else 0
        avg_return = np.mean(pnl_pcts) if pnl_pcts else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Kelly fraction
        if avg_loss != 0 and len(pnl_pcts) >= 20:
            rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            kelly = (win_rate * (rr_ratio + 1) - 1) / rr_ratio if rr_ratio > 0 else 0
            kelly = max(0, kelly * 0.25)  # Quarter-Kelly
        else:
            kelly = 0

        # Edge score: composite metric
        # Combines win rate, return magnitude, and consistency
        edge_score = win_rate * avg_return * np.sqrt(len(pnl_pcts))

        # Is the edge decaying?
        # Compare last 20 trades to the 20 before that
        is_decaying = False
        if len(pnl_pcts) >= 40:
            recent_20 = pnl_pcts[:20]
            older_20 = pnl_pcts[20:40]
            recent_wr = sum(1 for x in recent_20 if x > 0) / len(recent_20)
            older_wr = sum(1 for x in older_20 if x > 0) / len(older_20)

            # Edge is decaying if recent win rate dropped significantly
            if recent_wr < older_wr - 0.15:  # 15% drop
                is_decaying = True
                logger.warning(
                    f"⚠️ EDGE DECAY DETECTED for {strategy}: "
                    f"Recent WR={recent_wr*100:.1f}% vs Prior WR={older_wr*100:.1f}%"
                )

        cursor.execute("""
            INSERT INTO strategy_performance (
                timestamp, strategy, window_size,
                win_rate, avg_return, avg_win, avg_loss,
                profit_factor, kelly_fraction, edge_score,
                is_edge_decaying
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(ET).isoformat(),
            strategy, len(pnl_pcts),
            win_rate, avg_return, avg_win, avg_loss,
            profit_factor, kelly, edge_score,
            1 if is_decaying else 0,
        ))

        conn.commit()
        conn.close()

    def get_strategy_health(self, strategy: str) -> dict:
        """
        Get the current health of a strategy.
        Used by the bot to decide whether to keep trading a strategy.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT win_rate, avg_return, profit_factor,
                   kelly_fraction, edge_score, is_edge_decaying, window_size
            FROM strategy_performance
            WHERE strategy = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (strategy,))

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return {
                'status': 'no_data',
                'win_rate': None,
                'avg_return': None,
                'should_trade': True,  # No data = use research defaults
                'confidence': 'low',
            }

        win_rate, avg_return, profit_factor, kelly, edge_score, is_decaying, n = row

        # Determine if strategy should still be traded
        should_trade = True
        confidence = 'high'

        if is_decaying:
            confidence = 'degrading'
            # Don't stop immediately — reduce position sizing instead
            should_trade = True

        if win_rate < 0.40 and n >= 30:
            # Win rate dropped below 40% with enough sample = stop trading
            should_trade = False
            confidence = 'dead'
            logger.warning(f"🛑 Strategy {strategy} DISABLED: WR={win_rate*100:.1f}% < 40%")

        if avg_return < -2.0 and n >= 20:
            # Consistently losing = stop
            should_trade = False
            confidence = 'dead'
            logger.warning(f"🛑 Strategy {strategy} DISABLED: avg return={avg_return:.2f}%")

        return {
            'status': 'active' if should_trade else 'disabled',
            'win_rate': win_rate,
            'avg_return': avg_return,
            'profit_factor': profit_factor,
            'kelly_fraction': kelly,
            'edge_score': edge_score,
            'is_decaying': bool(is_decaying),
            'should_trade': should_trade,
            'confidence': confidence,
            'sample_size': n,
        }

    def get_adaptive_parameters(self) -> dict:
        """
        Calculate adaptive parameter adjustments based on live performance.

        This is the KEY adaptive mechanism:
        - If a strategy is winning more than expected → increase sizing
        - If a strategy is underperforming → decrease sizing
        - If edge has decayed → disable strategy
        - Adjust RVOL threshold based on what's actually working
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        adaptations = {}

        for strategy in ['rvol_rocket', 'dead_cat_fade', 'panic_bounce']:
            health = self.get_strategy_health(strategy)

            if health['status'] == 'no_data':
                adaptations[strategy] = {
                    'size_multiplier': 1.0,
                    'enabled': True,
                    'reason': 'Using research defaults (no live data yet)',
                }
                continue

            # Size multiplier based on live performance vs expectations
            if health['confidence'] == 'dead':
                adaptations[strategy] = {
                    'size_multiplier': 0.0,
                    'enabled': False,
                    'reason': f"Edge dead: WR={health['win_rate']*100:.1f}%, "
                              f"Avg={health['avg_return']:.2f}%",
                }
            elif health['is_decaying']:
                adaptations[strategy] = {
                    'size_multiplier': 0.5,  # Half size while decaying
                    'enabled': True,
                    'reason': f"Edge decaying: reducing size 50%. "
                              f"WR={health['win_rate']*100:.1f}%",
                }
            elif health['win_rate'] and health['win_rate'] > 0.70:
                # Performing great — can slightly increase
                adaptations[strategy] = {
                    'size_multiplier': 1.2,
                    'enabled': True,
                    'reason': f"Strong performance: WR={health['win_rate']*100:.1f}%",
                }
            else:
                adaptations[strategy] = {
                    'size_multiplier': 1.0,
                    'enabled': True,
                    'reason': 'Normal performance',
                }

        # Find optimal RVOL threshold from live trades
        cursor.execute("""
            SELECT relative_volume, pnl_pct, won
            FROM trades
            WHERE strategy = 'rvol_rocket'
            AND relative_volume IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        rvol_trades = cursor.fetchall()

        if len(rvol_trades) >= 30:
            # Test different RVOL thresholds
            best_threshold = 5.0
            best_edge = 0

            for threshold in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
                filtered = [t for t in rvol_trades if t[0] >= threshold]
                if len(filtered) >= 10:
                    wr = sum(1 for t in filtered if t[2] == 1) / len(filtered)
                    avg_r = np.mean([t[1] for t in filtered])
                    edge = wr * avg_r * np.sqrt(len(filtered))
                    if edge > best_edge:
                        best_edge = edge
                        best_threshold = threshold

            adaptations['optimal_rvol_threshold'] = best_threshold
            logger.info(f"Adaptive RVOL threshold: {best_threshold}x "
                       f"(from live data, {len(rvol_trades)} trades)")
        else:
            adaptations['optimal_rvol_threshold'] = 5.0  # Research default

        conn.close()
        return adaptations

    def get_ml_training_data(self) -> list[dict]:
        """
        Export trade data formatted for ML model training (Phase C).
        Returns feature vectors with outcome labels.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT gap_pct, relative_volume, atr_pct, prior_5d_return,
                   rsi, spy_bullish, has_catalyst, catalyst_weight,
                   sentiment_score, setup_score, day_of_week,
                   pnl_pct, won
            FROM trades
            WHERE gap_pct IS NOT NULL
            AND relative_volume IS NOT NULL
            ORDER BY timestamp
        """)

        columns = [
            'gap_pct', 'relative_volume', 'atr_pct', 'prior_5d_return',
            'rsi', 'spy_bullish', 'has_catalyst', 'catalyst_weight',
            'sentiment_score', 'setup_score', 'day_of_week',
            'pnl_pct', 'won'
        ]

        rows = cursor.fetchall()
        conn.close()

        return [dict(zip(columns, row)) for row in rows]

    def get_trade_count(self, strategy: str = None) -> int:
        """Get total number of trades (optionally filtered by strategy)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if strategy:
            cursor.execute("SELECT COUNT(*) FROM trades WHERE strategy = ?", (strategy,))
        else:
            cursor.execute("SELECT COUNT(*) FROM trades")

        count = cursor.fetchone()[0]
        conn.close()
        return count

    def log_daily_summary(self, summary: dict):
        """Save end-of-day summary."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO daily_summary (
                date, starting_capital, ending_capital, realized_pnl,
                trades_taken, wins, losses, win_rate,
                best_trade, worst_trade, strategies_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary['date'],
            summary.get('starting_capital', 0),
            summary.get('ending_capital', 0),
            summary.get('realized_pnl', 0),
            summary.get('trades_taken', 0),
            summary.get('wins', 0),
            summary.get('losses', 0),
            summary.get('win_rate', 0),
            summary.get('best_trade', 0),
            summary.get('worst_trade', 0),
            json.dumps(summary.get('strategies_used', [])),
        ))

        conn.commit()
        conn.close()

    def get_cumulative_stats(self) -> dict:
        """Get all-time cumulative statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*), 
                   SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END),
                   AVG(pnl_pct),
                   SUM(pnl_dollars),
                   MAX(pnl_dollars),
                   MIN(pnl_dollars)
            FROM trades
        """)

        row = cursor.fetchone()
        conn.close()

        if row[0] == 0:
            return {'total_trades': 0}

        return {
            'total_trades': row[0],
            'total_wins': row[1],
            'win_rate': row[1] / row[0] * 100 if row[0] > 0 else 0,
            'avg_return_pct': row[2] or 0,
            'total_pnl': row[3] or 0,
            'best_trade': row[4] or 0,
            'worst_trade': row[5] or 0,
        }
