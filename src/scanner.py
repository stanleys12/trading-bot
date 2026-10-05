"""
Pre-Market Scanner Module
==========================
Scans for gap-up and gap-down candidates before market open.
Uses Alpaca data API for prices/volume and yfinance for float/market cap.
"""

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from config import BotConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


@dataclass
class GapCandidate:
    """A stock that passed the scanner filters."""
    symbol: str
    current_price: float
    prev_close: float
    gap_pct: float
    premarket_volume: int
    avg_volume_20d: float
    relative_volume: float
    premarket_high: float
    premarket_low: float

    # Optional fundamental data
    market_cap: Optional[float] = None
    float_shares: Optional[float] = None

    # Indicators
    atr_pct: Optional[float] = None
    prior_5d_return: Optional[float] = None
    rsi: Optional[float] = None

    # Catalyst info (filled by catalyst module)
    has_catalyst: bool = False
    catalyst_type: Optional[str] = None
    catalyst_weight: int = 0
    sentiment_score: float = 0.0
    news_headline: Optional[str] = None

    # Scoring
    setup_score: float = 0.0
    strategy: Optional[str] = None  # "rvol_rocket", "dead_cat_fade", "panic_bounce"

    # Direction
    direction: str = "long"  # "long" or "short"


class Scanner:
    """
    Pre-market gap scanner.

    Workflow:
    1. Get universe of tradeable stocks from Alpaca
    2. Fetch yesterday's close and today's pre-market price
    3. Calculate gap % and RVOL
    4. Filter by scanner criteria
    5. Return ranked list of candidates
    """

    def __init__(self, config: BotConfig = DEFAULT_CONFIG,
                 trading_client=None, data_client=None):
        self.config = config
        self.trading_client = trading_client
        self.data_client = data_client
        self._universe_cache = None
        self._universe_cache_date = None

    def get_tradeable_universe(self) -> list[str]:
        """Get all tradeable US equity symbols from Alpaca."""
        if self.trading_client is None:
            logger.warning("No trading client — using fallback universe")
            return self._get_fallback_universe()

        # Cache for the day
        today = datetime.now(ET).date()
        if self._universe_cache and self._universe_cache_date == today:
            return self._universe_cache

        try:
            from alpaca.trading.requests import GetAssetsRequest
            from alpaca.trading.enums import AssetClass, AssetStatus

            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE
            )
            assets = self.trading_client.get_all_assets(request)

            symbols = [
                a.symbol for a in assets
                if a.tradable and a.shortable and not a.symbol.endswith('W')
                and '.' not in a.symbol  # skip warrants and units
            ]

            self._universe_cache = symbols
            self._universe_cache_date = today
            logger.info(f"Loaded {len(symbols)} tradeable symbols from Alpaca")
            return symbols

        except Exception as e:
            logger.error(f"Failed to get Alpaca universe: {e}")
            return self._get_fallback_universe()

    def _get_fallback_universe(self) -> list[str]:
        """Fallback list of known high-gap-frequency small caps."""
        return [
            "SOFI", "PLTR", "NIO", "XPEV", "LCID", "RIVN", "AMC", "MARA",
            "RIOT", "COIN", "HOOD", "UPST", "AFRM", "DKNG", "FUBO", "IONQ",
            "RKLB", "ASTS", "HIMS", "DNA", "QS", "SNDL", "TLRY", "CGC",
            "SPCE", "GSAT", "MVST", "MVIS", "WKHS", "CLOV", "SKLZ", "OPEN",
            "DOCS", "SKIN", "BKKT", "JOBY", "BARK", "REAL", "PAYO", "BNGO",
            "EDIT", "BEAM", "CRSP", "NTLA", "IOVA", "HUT", "BB", "ACB",
            "GENI", "RBLX", "LI", "BABA", "JD", "PDD", "BIDU", "IQ",
            "TIGR", "FUTU", "RIG", "AR", "BORR", "ABCL", "PRAX",
        ]

    def scan_for_gaps(self, symbols: list[str] = None) -> list[GapCandidate]:
        """
        Main scanning function. Identifies stocks gapping up or down.

        Returns a list of GapCandidate objects sorted by setup quality.
        """
        if symbols is None:
            symbols = self.get_tradeable_universe()

        now = datetime.now(ET)
        candidates = []

        # Process in batches to respect API limits
        batch_size = 100
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                batch_candidates = self._scan_batch(batch, now)
                candidates.extend(batch_candidates)
            except Exception as e:
                logger.error(f"Error scanning batch {i}: {e}")
                continue

        # Sort by absolute gap % (largest gaps first for initial ranking)
        candidates.sort(key=lambda c: abs(c.gap_pct), reverse=True)

        # Trim to max watchlist size
        max_size = self.config.scanner.max_watchlist_size
        candidates = candidates[:max_size]

        logger.info(f"Scanner found {len(candidates)} candidates")
        for c in candidates:
            logger.info(
                f"  {c.symbol}: gap={c.gap_pct:+.1f}%, "
                f"rvol={c.relative_volume:.1f}x, "
                f"price=${c.current_price:.2f}, "
                f"strategy={c.strategy}"
            )

        return candidates

    def _scan_batch(self, symbols: list[str],
                    now: datetime) -> list[GapCandidate]:
        """Scan a batch of symbols for gaps."""
        if self.data_client is None:
            return self._scan_batch_yfinance(symbols, now)

        candidates = []

        try:
            from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
            from alpaca.data.timeframe import TimeFrame

            # Get previous day's bars (for close price and avg volume)
            hist_request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=now - timedelta(days=30),
                end=now - timedelta(days=1),
            )
            hist_bars = self.data_client.get_stock_bars(hist_request)

            # Get latest bars (current/pre-market)
            latest_request = StockLatestBarRequest(symbol_or_symbols=symbols)
            latest_bars = self.data_client.get_stock_latest_bar(latest_request)

            for symbol in symbols:
                try:
                    candidate = self._evaluate_symbol_alpaca(
                        symbol, hist_bars, latest_bars
                    )
                    if candidate:
                        candidates.append(candidate)
                except Exception as e:
                    continue

        except Exception as e:
            logger.error(f"Alpaca batch scan failed: {e}")
            # Fallback to yfinance
            return self._scan_batch_yfinance(symbols, now)

        return candidates

    def _evaluate_symbol_alpaca(self, symbol: str, hist_bars, latest_bars) -> Optional[GapCandidate]:
        """Evaluate a single symbol using Alpaca data."""
        try:
            # Get historical data
            symbol_bars = hist_bars.df.loc[symbol] if symbol in hist_bars.df.index.get_level_values(0) else None
            if symbol_bars is None or len(symbol_bars) < 20:
                return None

            latest = latest_bars.get(symbol)
            if latest is None:
                return None

            prev_close = symbol_bars.iloc[-1]['close']
            current_price = latest.close
            premarket_volume = latest.volume

            # Gap calculation
            gap_pct = ((current_price - prev_close) / prev_close) * 100

            # Volume analysis
            avg_vol_20d = symbol_bars['volume'].tail(20).mean()
            rvol = premarket_volume / avg_vol_20d if avg_vol_20d > 0 else 0

            return self._build_candidate(
                symbol=symbol,
                current_price=current_price,
                prev_close=prev_close,
                gap_pct=gap_pct,
                premarket_volume=premarket_volume,
                avg_volume_20d=avg_vol_20d,
                rvol=rvol,
                premarket_high=latest.high,
                premarket_low=latest.low,
                hist_bars=symbol_bars,
            )

        except Exception:
            return None

    def _scan_batch_yfinance(self, symbols: list[str],
                             now: datetime) -> list[GapCandidate]:
        """Fallback: scan using yfinance (for development/testing)."""
        import yfinance as yf

        candidates = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="30d", auto_adjust=True)
                if len(hist) < 20:
                    continue

                prev_close = hist.iloc[-2]['Close']
                current_price = hist.iloc[-1]['Open']
                gap_pct = ((current_price - prev_close) / prev_close) * 100

                avg_vol_20d = hist['Volume'].tail(20).mean()
                today_vol = hist.iloc[-1]['Volume']
                rvol = today_vol / avg_vol_20d if avg_vol_20d > 0 else 0

                candidate = self._build_candidate(
                    symbol=symbol,
                    current_price=current_price,
                    prev_close=prev_close,
                    gap_pct=gap_pct,
                    premarket_volume=int(today_vol),
                    avg_volume_20d=avg_vol_20d,
                    rvol=rvol,
                    premarket_high=hist.iloc[-1]['High'],
                    premarket_low=hist.iloc[-1]['Low'],
                    hist_bars=hist,
                )
                if candidate:
                    candidates.append(candidate)
            except Exception:
                continue

        return candidates

    def _build_candidate(self, symbol: str, current_price: float,
                         prev_close: float, gap_pct: float,
                         premarket_volume: int, avg_volume_20d: float,
                         rvol: float, premarket_high: float,
                         premarket_low: float,
                         hist_bars=None) -> Optional[GapCandidate]:
        """
        Build a GapCandidate if it passes filters.
        This is where the research-derived filter logic lives.
        """
        cfg = self.config.scanner

        # === BASIC FILTERS ===
        if current_price < cfg.min_price or current_price > cfg.max_price:
            return None

        if premarket_volume < cfg.min_premarket_volume:
            return None

        # Calculate additional indicators from history
        atr_pct = None
        prior_5d_return = None
        rsi_val = None

        if hist_bars is not None and len(hist_bars) >= 20:
            from indicators import calc_atr_pct, calc_rsi, calc_prior_trend

            closes = hist_bars['Close'] if 'Close' in hist_bars.columns else hist_bars['close']
            highs = hist_bars['High'] if 'High' in hist_bars.columns else hist_bars['high']
            lows = hist_bars['Low'] if 'Low' in hist_bars.columns else hist_bars['low']

            atr_series = calc_atr_pct(highs, lows, closes)
            atr_pct = atr_series.iloc[-1] if len(atr_series) > 0 else None

            rsi_series = calc_rsi(closes)
            rsi_val = rsi_series.iloc[-1] if len(rsi_series) > 0 else None

            prior_5d_return = calc_prior_trend(closes, lookback=5)

        # === STRATEGY CLASSIFICATION ===
        strategy = None
        direction = "long"

        # Strategy 1: RVOL Rocket (LONG)
        if (gap_pct >= cfg.min_gap_pct and gap_pct <= cfg.max_gap_pct_long
                and rvol >= cfg.min_rvol_long):
            strategy = "rvol_rocket"
            direction = "long"

        # Strategy 2: Dead Cat Fade (SHORT)
        elif (gap_pct >= cfg.min_gap_pct_short
              and gap_pct <= cfg.max_gap_pct_short
              and rvol < cfg.max_rvol_short):
            strategy = "dead_cat_fade"
            direction = "short"

        # Strategy 3: Panic Bounce (LONG on gap down)
        elif (gap_pct <= -cfg.min_gap_down_pct
              and gap_pct >= -cfg.max_gap_down_pct
              and rvol < cfg.max_rvol_bounce):
            strategy = "panic_bounce"
            direction = "long"

        # Doesn't match any strategy
        if strategy is None:
            return None

        candidate = GapCandidate(
            symbol=symbol,
            current_price=current_price,
            prev_close=prev_close,
            gap_pct=gap_pct,
            premarket_volume=premarket_volume,
            avg_volume_20d=avg_volume_20d,
            relative_volume=rvol,
            premarket_high=premarket_high,
            premarket_low=premarket_low,
            atr_pct=atr_pct,
            prior_5d_return=prior_5d_return,
            rsi=rsi_val,
            strategy=strategy,
            direction=direction,
        )

        return candidate

    def enrich_with_fundamentals(self, candidates: list[GapCandidate]) -> list[GapCandidate]:
        """
        Add market cap and float data from yfinance.
        Filters out stocks that exceed market cap / float limits.
        """
        import yfinance as yf

        enriched = []
        for c in candidates:
            try:
                info = yf.Ticker(c.symbol).info
                c.market_cap = info.get('marketCap')
                c.float_shares = info.get('floatShares')

                # Filter by market cap
                if c.market_cap and c.market_cap > self.config.scanner.max_market_cap:
                    logger.debug(f"  {c.symbol} filtered: market cap ${c.market_cap/1e9:.1f}B > ${self.config.scanner.max_market_cap/1e9:.1f}B")
                    continue

                # Filter by float
                if c.float_shares and c.float_shares > self.config.scanner.max_float:
                    logger.debug(f"  {c.symbol} filtered: float {c.float_shares/1e6:.1f}M > {self.config.scanner.max_float/1e6:.1f}M")
                    continue

                enriched.append(c)
            except Exception:
                # If we can't get fundamentals, include anyway (conservative)
                enriched.append(c)

        return enriched
