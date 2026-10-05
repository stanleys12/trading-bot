"""
Trading Bot Configuration
=========================
All configurable parameters in one place.
Environment variables for secrets, hardcoded defaults for strategy params.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""
    api_key: str = field(default_factory=lambda: os.environ.get("ALPACA_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.environ.get("ALPACA_API_SECRET", ""))
    paper: bool = True  # ALWAYS start with paper trading
    data_feed: str = "iex"  # "iex" (free) or "sip" (paid, all exchanges)


@dataclass
class ScannerConfig:
    """Pre-market scanner filter thresholds (from research findings)."""
    # Price range
    min_price: float = 2.0
    max_price: float = 20.0

    # Gap thresholds
    min_gap_pct: float = 3.0       # Minimum gap % to consider
    max_gap_pct_long: float = 12.0  # Max gap for RVOL Rocket (long)
    min_gap_pct_short: float = 10.0 # Min gap for Dead Cat Fade (short)
    max_gap_pct_short: float = 15.0 # Max gap for Dead Cat Fade
    min_gap_down_pct: float = 15.0  # Min gap down for Panic Bounce
    max_gap_down_pct: float = 25.0  # Max gap down for Panic Bounce

    # Volume thresholds (THE key edge)
    min_rvol_long: float = 5.0      # RVOL≥5x for longs (research: 86% WR)
    max_rvol_short: float = 2.0     # RVOL<2x for shorts (research: 79.7% WR)
    max_rvol_bounce: float = 3.0    # RVOL<3x for gap-down bounce
    min_premarket_volume: int = 10000  # Minimum pre-market shares traded

    # Universe filters
    max_market_cap: float = 2e9     # $2B max (small cap territory)
    max_float: float = 20e6         # 20M shares max float

    # How many candidates to track
    max_watchlist_size: int = 10


@dataclass
class StrategyConfig:
    """Strategy parameters derived from research."""
    # RVOL Rocket (Primary Long)
    rvol_rocket_enabled: bool = True
    rvol_rocket_min_gap: float = 3.0
    rvol_rocket_max_gap: float = 12.0
    rvol_rocket_min_rvol: float = 5.0
    rvol_rocket_prefer_spy_bullish: bool = True
    rvol_rocket_prefer_low_atr: bool = True  # ATR < 6% preferred
    rvol_rocket_max_atr_pct: float = 6.0

    # Dead Cat Fade (Short)
    dead_cat_enabled: bool = True
    dead_cat_min_gap: float = 10.0
    dead_cat_max_gap: float = 15.0
    dead_cat_max_rvol: float = 2.0
    dead_cat_stop_pct: float = 8.0  # Hard stop 8% above entry

    # Panic Bounce (Mean Reversion Long)
    panic_bounce_enabled: bool = True
    panic_bounce_min_gap_down: float = 15.0
    panic_bounce_max_gap_down: float = 25.0
    panic_bounce_max_rvol: float = 3.0
    panic_bounce_stop_pct: float = 8.0  # Hard stop 8% below entry


@dataclass
class RiskConfig:
    """Risk management parameters (non-negotiable safety rails)."""
    starting_capital: float = 30000.0

    # Per-trade risk
    max_risk_per_trade_pct: float = 1.0     # 1% of account per trade (long)
    max_risk_per_trade_short_pct: float = 0.5  # 0.5% for shorts (more conservative)
    max_position_pct: float = 15.0          # Max 15% of capital in one position
    max_position_adv_pct: float = 5.0       # Max 5% of avg daily volume

    # Daily limits
    max_daily_loss_pct: float = 3.0         # Shut down after 3% daily loss
    max_trades_per_day: int = 5             # Cap on total trades per day
    max_concurrent_positions: int = 3       # Max open positions at once

    # Time limits
    close_all_by_hour: int = 11             # Close everything by 11 AM ET
    close_all_by_minute: int = 0
    hard_close_by_hour: int = 15            # Absolute latest: 3 PM ET
    hard_close_by_minute: int = 45

    # Kelly Criterion
    kelly_fraction: float = 0.25            # Quarter-Kelly (conservative)
    kelly_min_trades: int = 50              # Min trades before using Kelly
    kelly_lookback: int = 200               # Rolling window for Kelly calc

    # Circuit breaker
    max_consecutive_losses: int = 3         # Pause after 3 consecutive losses
    pause_minutes_after_breaker: int = 30   # Wait 30 min before resuming


@dataclass
class TradingHours:
    """Market hours (all in ET)."""
    premarket_scan_start_hour: int = 7
    premarket_scan_start_minute: int = 0
    premarket_scan_end_hour: int = 9
    premarket_scan_end_minute: int = 25

    market_open_hour: int = 9
    market_open_minute: int = 30

    # Wait N seconds after open before first trade (avoid the chaos)
    open_delay_seconds: int = 30

    market_close_hour: int = 16
    market_close_minute: int = 0

    timezone: str = "America/New_York"


@dataclass
class CatalystConfig:
    """News/catalyst detection settings."""
    # FinBERT
    finbert_model: str = "ProsusAI/finbert"
    min_sentiment_score: float = 0.6  # Minimum positive sentiment to proceed

    # Catalyst keywords (weighted by impact)
    catalyst_keywords: dict = field(default_factory=lambda: {
        # High impact (weight 3)
        "fda approval": 3, "fda approved": 3, "phase 3": 3,
        "phase iii": 3, "breakthrough therapy": 3,
        "earnings beat": 3, "revenue beat": 3, "guidance raised": 3,
        "acquisition": 3, "merger": 3, "buyout": 3,
        "short squeeze": 3, "gamma squeeze": 3,

        # Medium impact (weight 2)
        "contract awarded": 2, "partnership": 2, "collaboration": 2,
        "new product": 2, "fda filing": 2, "clinical trial": 2,
        "phase 2": 2, "phase ii": 2, "insider buying": 2,
        "stock split": 2, "buyback": 2, "share repurchase": 2,
        "upgrade": 2, "price target raised": 2,

        # Lower impact (weight 1)
        "positive results": 1, "beat estimates": 1, "raised guidance": 1,
        "new ceo": 1, "restructuring": 1, "cost cutting": 1,
        "analyst coverage": 1, "initiated": 1,
    })

    # Anti-catalyst keywords (avoid these)
    anti_keywords: list = field(default_factory=lambda: [
        "dilution", "offering", "shelf registration", "delisting",
        "sec investigation", "fraud", "bankruptcy", "chapter 11",
        "reverse split", "going concern",
    ])


@dataclass
class LoggingConfig:
    """Logging and trade journal settings."""
    db_path: str = "data/trades.db"
    log_level: str = "INFO"
    log_file: str = "data/bot.log"
    save_scanner_results: bool = True
    scanner_results_dir: str = "data/scanner_history"


@dataclass
class BotConfig:
    """Master configuration combining all sub-configs."""
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    hours: TradingHours = field(default_factory=TradingHours)
    catalyst: CatalystConfig = field(default_factory=CatalystConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Singleton default config
DEFAULT_CONFIG = BotConfig()
