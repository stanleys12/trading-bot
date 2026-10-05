"""
Strategy Module
================
Implements the 3 research-validated trading strategies.
Each strategy takes a GapCandidate and produces trade signals.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from config import BotConfig, DEFAULT_CONFIG
from scanner import GapCandidate
from indicators import score_setup
from risk_manager import PositionRequest

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """A fully-formed trade signal ready for risk approval."""
    symbol: str
    strategy: str           # "rvol_rocket", "dead_cat_fade", "panic_bounce"
    direction: str          # "long" or "short"
    entry_price: float
    stop_price: float
    target_price: Optional[float]
    setup_score: float
    reasoning: str          # Human-readable explanation
    candidate: GapCandidate  # Original scanner data


class StrategyEngine:
    """
    Evaluates gap candidates and generates trade signals.

    Three strategies, each validated by our research:
    1. RVOL Rocket    — Long  (86% WR, +15.3% avg, p=0.0000)
    2. Dead Cat Fade  — Short (79.7% WR, +4.3% avg)
    3. Panic Bounce   — Long  (60% WR, +2.2% avg)
    """

    def __init__(self, config: BotConfig = DEFAULT_CONFIG):
        self.config = config

    def evaluate_candidates(self, candidates: list[GapCandidate],
                            spy_bullish: bool = True) -> list[TradeSignal]:
        """
        Evaluate all scanner candidates and generate trade signals.
        Returns signals sorted by setup_score (best first).
        """
        signals = []

        for candidate in candidates:
            signal = self._evaluate_single(candidate, spy_bullish)
            if signal:
                signals.append(signal)

        # Sort by score descending (best setups first)
        signals.sort(key=lambda s: s.setup_score, reverse=True)

        for sig in signals:
            logger.info(
                f"SIGNAL: {sig.direction.upper()} {sig.symbol} "
                f"via {sig.strategy} | score={sig.setup_score:.0f} | "
                f"entry=${sig.entry_price:.2f} stop=${sig.stop_price:.2f}"
            )

        return signals

    def _evaluate_single(self, c: GapCandidate,
                         spy_bullish: bool) -> Optional[TradeSignal]:
        """Evaluate a single candidate based on its assigned strategy."""
        if c.strategy == "rvol_rocket":
            return self._rvol_rocket(c, spy_bullish)
        elif c.strategy == "dead_cat_fade":
            return self._dead_cat_fade(c, spy_bullish)
        elif c.strategy == "panic_bounce":
            return self._panic_bounce(c, spy_bullish)
        return None

    def _rvol_rocket(self, c: GapCandidate,
                     spy_bullish: bool) -> Optional[TradeSignal]:
        """
        RVOL Rocket Strategy (Primary Long)

        Research: 86% win rate, +15.3% avg return
        Best when: gap 3-12%, RVOL≥5x, price $2-$20, SPY bullish

        Entry:  Buy at open (or first pullback to VWAP)
        Stop:   Pre-market low or -5% from entry (whichever is tighter)
        Target: Trail with 9 EMA, or close at EOD
        """
        cfg = self.config.strategy

        if not cfg.rvol_rocket_enabled:
            return None

        # Score the setup
        setup_score = score_setup(
            gap_pct=c.gap_pct,
            rvol=c.relative_volume,
            atr_pct=c.atr_pct or 5.0,
            spy_bullish=spy_bullish,
            prior_5d_return=c.prior_5d_return or 0.0,
            sentiment_score=c.sentiment_score,
            catalyst_weight=c.catalyst_weight,
        )

        # Minimum score threshold
        if setup_score < 30:
            return None

        # Calculate stop price
        # Use pre-market low or 5% below entry (whichever gives tighter stop)
        stop_from_premarket = c.premarket_low
        stop_from_pct = c.current_price * 0.95  # 5% stop

        # Use the HIGHER of the two (tighter stop = less risk)
        stop_price = max(stop_from_premarket, stop_from_pct)

        # Ensure stop makes sense (at least 1% below entry)
        if stop_price >= c.current_price * 0.99:
            stop_price = c.current_price * 0.95

        # Target: no fixed target (trail with 9 EMA or close at EOD)
        target_price = None

        reasoning = (
            f"RVOL Rocket: gap +{c.gap_pct:.1f}%, RVOL {c.relative_volume:.1f}x "
            f"(research: 86% WR when ≥5x). "
            f"Price ${c.current_price:.2f} in sweet spot $2-$20. "
            f"{'SPY bullish ✓' if spy_bullish else 'SPY bearish ⚠️'}. "
            f"{'Catalyst: ' + c.news_headline if c.news_headline else 'No catalyst detected'}."
        )

        return TradeSignal(
            symbol=c.symbol,
            strategy="rvol_rocket",
            direction="long",
            entry_price=c.current_price,
            stop_price=stop_price,
            target_price=target_price,
            setup_score=setup_score,
            reasoning=reasoning,
            candidate=c,
        )

    def _dead_cat_fade(self, c: GapCandidate,
                       spy_bullish: bool) -> Optional[TradeSignal]:
        """
        Dead Cat Fade Strategy (Short)

        Research: 79.7% win rate, +4.3% avg return
        Best when: gap 10-15%, RVOL < 2x (LOW volume = no conviction)

        Entry:  Short at open
        Stop:   8% above entry (hard stop, non-negotiable)
        Target: Close at EOD or when stock touches VWAP from above
        """
        cfg = self.config.strategy

        if not cfg.dead_cat_enabled:
            return None

        # Shorts need extra caution
        # Skip if volume is actually high (potential squeeze)
        if c.relative_volume >= cfg.dead_cat_max_rvol:
            return None

        # Score: use base score but adjust for short-specific factors
        setup_score = 50.0  # Base score for short setups

        # Low RVOL is GOOD for shorts (that's the whole thesis)
        if c.relative_volume < 1.0:
            setup_score += 20
        elif c.relative_volume < 1.5:
            setup_score += 10

        # Higher gap = more fade potential (but more risk too)
        if 12 <= c.gap_pct <= 15:
            setup_score += 15
        elif 10 <= c.gap_pct < 12:
            setup_score += 10

        # SPY bearish is GOOD for shorts
        if not spy_bullish:
            setup_score += 10

        # No catalyst = better for short (gap has no backing)
        if not c.has_catalyst:
            setup_score += 10
        elif c.catalyst_weight <= 1:
            setup_score += 5

        if setup_score < 40:
            return None

        # Stop: 8% above entry (hard stop)
        stop_price = c.current_price * (1 + cfg.dead_cat_stop_pct / 100)

        # Target: previous close (full gap fill)
        target_price = c.prev_close

        reasoning = (
            f"Dead Cat Fade: gap +{c.gap_pct:.1f}% but RVOL only {c.relative_volume:.1f}x "
            f"(research: 79.7% WR when RVOL<2x). "
            f"Low volume = no conviction behind the gap. "
            f"{'No catalyst ✓' if not c.has_catalyst else 'Weak catalyst ⚠️'}. "
            f"Target: gap fill to ${c.prev_close:.2f}."
        )

        return TradeSignal(
            symbol=c.symbol,
            strategy="dead_cat_fade",
            direction="short",
            entry_price=c.current_price,
            stop_price=stop_price,
            target_price=target_price,
            setup_score=setup_score,
            reasoning=reasoning,
            candidate=c,
        )

    def _panic_bounce(self, c: GapCandidate,
                      spy_bullish: bool) -> Optional[TradeSignal]:
        """
        Panic Bounce Strategy (Mean Reversion Long)

        Research: 60% win rate, +2.2% avg return
        Best when: gap down 15-25%, RVOL < 3x, price ≥ $2

        Entry:  Buy at open (catching the oversold bounce)
        Stop:   8% below entry
        Target: Close at EOD
        """
        cfg = self.config.strategy

        if not cfg.panic_bounce_enabled:
            return None

        # HIGH RVOL on gap downs = real capitulation, avoid!
        if c.relative_volume >= cfg.panic_bounce_max_rvol:
            return None

        setup_score = 40.0

        # Gap down size scoring
        gap_down = abs(c.gap_pct)
        if 15 <= gap_down <= 20:
            setup_score += 15
        elif 20 < gap_down <= 25:
            setup_score += 10

        # Low RVOL is good (panic without institutional selling)
        if c.relative_volume < 1.5:
            setup_score += 15
        elif c.relative_volume < 2.0:
            setup_score += 10

        # SPY bullish helps bounces
        if spy_bullish:
            setup_score += 10

        if setup_score < 40:
            return None

        # Stop: 8% below entry
        stop_price = c.current_price * (1 - cfg.panic_bounce_stop_pct / 100)

        # Target: close at EOD (no fixed target)
        target_price = None

        reasoning = (
            f"Panic Bounce: gap {c.gap_pct:.1f}% with RVOL {c.relative_volume:.1f}x. "
            f"Research: 60% WR on 15-25% gap downs with RVOL<3x. "
            f"Oversold bounce play — panic selling tends to overshoot. "
            f"{'SPY bullish ✓ (helps bounce)' if spy_bullish else 'SPY bearish ⚠️'}."
        )

        return TradeSignal(
            symbol=c.symbol,
            strategy="panic_bounce",
            direction="long",
            entry_price=c.current_price,
            stop_price=stop_price,
            target_price=target_price,
            setup_score=setup_score,
            reasoning=reasoning,
            candidate=c,
        )

    def signal_to_position_request(self, signal: TradeSignal) -> PositionRequest:
        """Convert a TradeSignal to a PositionRequest for risk approval."""
        return PositionRequest(
            symbol=signal.symbol,
            direction=signal.direction,
            strategy=signal.strategy,
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            setup_score=signal.setup_score,
        )
