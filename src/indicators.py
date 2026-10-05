"""
Technical Indicators Module
============================
All indicator calculations used by the strategies.
Pure functions — no API calls, no side effects.
"""

import numpy as np
import pandas as pd
from typing import Optional


def calc_vwap(highs: pd.Series, lows: pd.Series, closes: pd.Series,
              volumes: pd.Series) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).
    Resets each trading day.
    """
    typical_price = (highs + lows + closes) / 3
    cumulative_tp_vol = (typical_price * volumes).cumsum()
    cumulative_vol = volumes.cumsum()
    vwap = cumulative_tp_vol / cumulative_vol
    return vwap


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calc_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=period).mean()


def calc_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_atr(highs: pd.Series, lows: pd.Series, closes: pd.Series,
             period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    prev_close = closes.shift(1)
    tr = pd.concat([
        highs - lows,
        (highs - prev_close).abs(),
        (lows - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calc_atr_pct(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                 period: int = 14) -> pd.Series:
    """ATR as percentage of price (for volatility comparison across price levels)."""
    atr = calc_atr(highs, lows, closes, period)
    return (atr / closes) * 100


def calc_relative_volume(volumes: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Relative Volume (RVOL).
    This is THE key indicator from our research.

    RVOL = Current Volume / Average Volume over N days

    Research findings:
        RVOL < 1x  → 22.6% continuation, -2.38% avg return
        RVOL 3-5x  → 57.4% continuation, +5.67% avg return
        RVOL 5-10x → 64.8% continuation, +6.54% avg return
        RVOL 10x+  → 53.9% continuation, +10.23% avg return
    """
    avg_volume = volumes.rolling(window=period).mean()
    rvol = volumes / avg_volume
    return rvol


def calc_gap_pct(current_open: float, prev_close: float) -> float:
    """
    Calculate gap percentage.
    Positive = gap up, Negative = gap down.
    """
    if prev_close <= 0:
        return 0.0
    return ((current_open - prev_close) / prev_close) * 100


def calc_opening_range(bars_df: pd.DataFrame, minutes: int = 5) -> dict:
    """
    Calculate the Opening Range (first N minutes of trading).

    Args:
        bars_df: DataFrame with columns [open, high, low, close, volume]
                 indexed by timestamp, containing 1-minute bars
        minutes: Opening range period (5, 15, or 30)

    Returns:
        dict with 'high', 'low', 'range_pct', 'volume'
    """
    if len(bars_df) < minutes:
        return None

    or_bars = bars_df.iloc[:minutes]
    or_high = or_bars['high'].max()
    or_low = or_bars['low'].min()
    or_volume = or_bars['volume'].sum()
    or_open = or_bars.iloc[0]['open']

    range_pct = ((or_high - or_low) / or_open) * 100 if or_open > 0 else 0

    return {
        'high': or_high,
        'low': or_low,
        'open': or_open,
        'range_pct': range_pct,
        'volume': or_volume,
        'midpoint': (or_high + or_low) / 2,
    }


def calc_close_position_in_range(high: float, low: float, close: float) -> float:
    """
    Where did the stock close within its daily range?
    0.0 = closed at the low (bearish)
    1.0 = closed at the high (bullish)
    0.5 = closed at midpoint
    """
    if high == low:
        return 0.5
    return (close - low) / (high - low)


def is_above_vwap(price: float, vwap: float) -> bool:
    """Check if price is above VWAP (bullish)."""
    return price > vwap


def calc_prior_trend(closes: pd.Series, lookback: int = 5) -> float:
    """
    Calculate prior N-day return.
    Used to check if stock was already running before the gap.

    Research finding: stocks already up >20% in 5 days have WORST
    gap continuation (38.7% WR).
    """
    if len(closes) < lookback + 1:
        return 0.0
    return (closes.iloc[-1] / closes.iloc[-(lookback + 1)] - 1) * 100


def calc_spy_trend(spy_close: float, spy_sma20: float) -> bool:
    """
    Determine if SPY is in a bullish trend.
    Research finding: bullish SPY → 43.5% continuation vs 34.1% bearish.
    """
    return spy_close > spy_sma20


def score_setup(gap_pct: float, rvol: float, atr_pct: float,
                spy_bullish: bool, prior_5d_return: float,
                sentiment_score: float = 0.0,
                catalyst_weight: int = 0) -> float:
    """
    Score a trading setup from 0-100 based on all research factors.
    Higher score = higher conviction = larger position.

    Weights derived from research findings.
    """
    score = 0.0

    # RVOL is the #1 factor (research: most predictive)
    if rvol >= 10:
        score += 30
    elif rvol >= 7:
        score += 28
    elif rvol >= 5:
        score += 25
    elif rvol >= 3:
        score += 15
    elif rvol >= 2:
        score += 5

    # Gap size sweet spot (3-12% is optimal from research)
    if 3 <= gap_pct <= 12:
        score += 20
    elif 12 < gap_pct <= 20:
        score += 10
    elif gap_pct > 20:
        score += 5  # Too large = more risk

    # Low ATR = normally quiet stock suddenly gapping (research: 77.8% WR)
    if atr_pct < 3:
        score += 15
    elif atr_pct < 6:
        score += 10
    elif atr_pct < 10:
        score += 5
    # atr >= 10: no bonus (research: 30.1% WR for extreme vol)

    # SPY context (research: +10 pct points continuation in bull market)
    if spy_bullish:
        score += 10

    # Prior trend: penalize if already extended
    if -5 <= prior_5d_return <= 5:
        score += 10  # Flat base = best setup
    elif prior_5d_return > 20:
        score -= 10  # Already running = exhaustion risk

    # Catalyst quality
    if catalyst_weight >= 3:
        score += 10
    elif catalyst_weight >= 2:
        score += 7
    elif catalyst_weight >= 1:
        score += 3

    # Sentiment
    if sentiment_score >= 0.8:
        score += 5
    elif sentiment_score >= 0.6:
        score += 3

    return max(0, min(100, score))
