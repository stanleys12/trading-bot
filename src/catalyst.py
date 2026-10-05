"""
News & Catalyst Detection Module
==================================
Fetches news headlines and classifies sentiment + catalyst type.
Uses FinBERT (pre-trained, no custom training needed).
"""

import logging
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional

from config import BotConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


class CatalystDetector:
    """
    Detects news catalysts for gap moves.

    Two-stage pipeline:
    1. Keyword detection (fast, no dependencies)
    2. FinBERT sentiment scoring (optional, requires transformers)

    A gap WITHOUT a catalyst is often a pump-and-dump trap.
    """

    def __init__(self, config: BotConfig = DEFAULT_CONFIG):
        self.config = config
        self._finbert = None
        self._finbert_loaded = False

    def _load_finbert(self):
        """Lazy-load FinBERT model (heavy, only load when needed)."""
        if self._finbert_loaded:
            return

        try:
            from transformers import pipeline
            self._finbert = pipeline(
                "sentiment-analysis",
                model=self.config.catalyst.finbert_model,
                truncation=True,
            )
            self._finbert_loaded = True
            logger.info("FinBERT model loaded successfully")
        except ImportError:
            logger.warning(
                "transformers not installed — running without FinBERT. "
                "Install with: pip install transformers torch"
            )
            self._finbert_loaded = True  # Don't retry
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            self._finbert_loaded = True

    def analyze_news(self, symbol: str, headlines: list[str] = None) -> dict:
        """
        Analyze news for a symbol.

        Returns:
            dict with:
                has_catalyst: bool
                catalyst_type: str or None
                catalyst_weight: int (1-3)
                sentiment_score: float (-1 to 1)
                top_headline: str or None
                anti_catalyst: bool (offering, dilution, etc.)
        """
        if headlines is None:
            headlines = self._fetch_headlines(symbol)

        if not headlines:
            return {
                'has_catalyst': False,
                'catalyst_type': None,
                'catalyst_weight': 0,
                'sentiment_score': 0.0,
                'top_headline': None,
                'anti_catalyst': False,
            }

        # Stage 1: Keyword detection
        catalyst_result = self._detect_catalyst_keywords(headlines)

        # Stage 2: FinBERT sentiment (if available)
        sentiment_score = self._get_sentiment(headlines)

        # Stage 3: Anti-catalyst check
        anti_catalyst = self._check_anti_catalyst(headlines)

        return {
            'has_catalyst': catalyst_result['found'],
            'catalyst_type': catalyst_result['type'],
            'catalyst_weight': catalyst_result['weight'],
            'sentiment_score': sentiment_score,
            'top_headline': headlines[0] if headlines else None,
            'anti_catalyst': anti_catalyst,
        }

    def _detect_catalyst_keywords(self, headlines: list[str]) -> dict:
        """Scan headlines for catalyst keywords."""
        best_match = {'found': False, 'type': None, 'weight': 0}
        keywords = self.config.catalyst.catalyst_keywords

        combined_text = ' '.join(headlines).lower()

        for keyword, weight in keywords.items():
            if keyword in combined_text:
                if weight > best_match['weight']:
                    best_match = {
                        'found': True,
                        'type': keyword,
                        'weight': weight,
                    }

        return best_match

    def _get_sentiment(self, headlines: list[str]) -> float:
        """
        Get FinBERT sentiment score.
        Returns: float from -1 (very negative) to +1 (very positive)
        """
        self._load_finbert()

        if self._finbert is None:
            return 0.0

        try:
            # Analyze top 5 headlines (most relevant)
            texts = headlines[:5]
            results = self._finbert(texts)

            scores = []
            for r in results:
                if r['label'] == 'positive':
                    scores.append(r['score'])
                elif r['label'] == 'negative':
                    scores.append(-r['score'])
                else:  # neutral
                    scores.append(0.0)

            # Return weighted average (more weight on first headline)
            if not scores:
                return 0.0

            weights = [1.0 / (i + 1) for i in range(len(scores))]
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            total_weight = sum(weights)

            return weighted_sum / total_weight

        except Exception as e:
            logger.error(f"FinBERT error: {e}")
            return 0.0

    def _check_anti_catalyst(self, headlines: list[str]) -> bool:
        """Check if any headline contains negative catalysts (avoid!)."""
        combined = ' '.join(headlines).lower()
        for keyword in self.config.catalyst.anti_keywords:
            if keyword in combined:
                logger.warning(f"Anti-catalyst detected: '{keyword}'")
                return True
        return False

    def _fetch_headlines(self, symbol: str) -> list[str]:
        """
        Fetch recent news headlines for a symbol.
        Uses Alpaca News API if available, falls back to yfinance.
        """
        headlines = []

        # Try yfinance first (always available, no API key needed)
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news = ticker.news
            if news:
                for article in news[:10]:
                    title = article.get('title', '')
                    if title:
                        headlines.append(title)
        except Exception as e:
            logger.debug(f"yfinance news fetch failed for {symbol}: {e}")

        return headlines

    def enrich_candidate(self, candidate) -> None:
        """
        Enrich a GapCandidate with catalyst data.
        Modifies the candidate in-place.
        """
        result = self.analyze_news(candidate.symbol)

        candidate.has_catalyst = result['has_catalyst']
        candidate.catalyst_type = result['catalyst_type']
        candidate.catalyst_weight = result['catalyst_weight']
        candidate.sentiment_score = result['sentiment_score']
        candidate.news_headline = result['top_headline']

        # If anti-catalyst detected, zero out the catalyst
        if result['anti_catalyst']:
            candidate.has_catalyst = False
            candidate.catalyst_weight = 0
            logger.warning(
                f"{candidate.symbol}: Anti-catalyst detected — "
                f"flagging as no-catalyst"
            )
