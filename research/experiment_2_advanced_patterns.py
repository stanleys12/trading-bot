"""
EXPERIMENT 2: Consecutive Gap & Momentum Decay Patterns
========================================================
This experiment looks for patterns the daily gap analysis might miss:

1. Consecutive gap patterns (back-to-back gaps)
2. Gap size "sweet spots" with tighter statistical rigor
3. Volume profile decay analysis
4. Mean reversion timing (when do fades accelerate?)
5. The "2nd day play" - is day 2 after a big gap more reliable?
6. Gap + prior trend analysis (gap from a base vs gap from a run)
7. Intraday range ratio (is the open near the high or low?)
8. Statistical edge calculation with Monte Carlo validation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Focused universe - stocks with the most gap activity
FOCUS_UNIVERSE = [
    # High-gap-frequency small caps
    "SOFI", "PLTR", "NIO", "XPEV", "LCID", "RIVN", "AMC", "MARA", "RIOT",
    "COIN", "HOOD", "UPST", "AFRM", "DKNG", "FUBO", "IONQ", "RKLB",
    "ASTS", "HIMS", "DNA", "QS", "NKLA", "GOEV", "MULN", "SNDL",
    "TLRY", "CGC", "SPCE", "BBBY", "GSAT", "MVST", "MVIS", "WKHS",
    "CLOV", "WISH", "SKLZ", "OPEN", "RDFN", "DOCS", "SKIN",
    "BKKT", "JOBY", "LILM", "BARK", "REAL", "PAYO",
    "SAVA", "BNGO", "EDIT", "BEAM", "CRSP", "NTLA", "IOVA",
    "BITF", "HUT", "BB", "ACB", "GENI", "RBLX",
    "LI", "BABA", "JD", "PDD", "BIDU", "IQ", "TIGR", "FUTU",
    "BGFV", "VNET", "CENN", "PHUN", "RIG", "AR", "CTRA", "BORR",
    "VERV", "ABCL", "PRAX",
    # Market context
    "SPY", "QQQ", "IWM",  # IWM = Russell 2000 (small cap index)
]

FOCUS_UNIVERSE = list(set(FOCUS_UNIVERSE))


def download_all(symbols, period="5y"):
    """Download data."""
    print(f"Downloading {len(symbols)} symbols...")
    data = {}
    for i, sym in enumerate(symbols):
        try:
            df = yf.Ticker(sym).history(period=period, auto_adjust=True)
            if len(df) > 60:
                data[sym] = df
        except:
            pass
        if (i+1) % 25 == 0:
            print(f"  {i+1}/{len(symbols)} done")
    print(f"Downloaded {len(data)} symbols successfully")
    return data


def build_features(df, symbol):
    """Build a rich feature set for each trading day."""
    df = df.copy()
    
    # Basic gap
    df['prev_close'] = df['Close'].shift(1)
    df['gap_pct'] = (df['Open'] - df['prev_close']) / df['prev_close'] * 100
    
    # Intraday metrics
    df['intraday_return'] = (df['Close'] - df['Open']) / df['Open'] * 100
    df['high_from_open_pct'] = (df['High'] - df['Open']) / df['Open'] * 100
    df['low_from_open_pct'] = (df['Low'] - df['Open']) / df['Open'] * 100
    df['full_range_pct'] = (df['High'] - df['Low']) / df['Open'] * 100
    
    # Where did it close in the day's range? (0 = closed at low, 1 = closed at high)
    df['close_position'] = np.where(
        df['High'] != df['Low'],
        (df['Close'] - df['Low']) / (df['High'] - df['Low']),
        0.5
    )
    
    # Where did it open in the day's range?
    df['open_position'] = np.where(
        df['High'] != df['Low'],
        (df['Open'] - df['Low']) / (df['High'] - df['Low']),
        0.5
    )
    
    # Volume analysis
    df['vol_sma20'] = df['Volume'].rolling(20).mean()
    df['relative_volume'] = df['Volume'] / df['vol_sma20']
    df['vol_change'] = df['Volume'] / df['Volume'].shift(1)
    
    # Trend context (was the stock already trending before the gap?)
    df['sma5'] = df['Close'].rolling(5).mean()
    df['sma10'] = df['Close'].rolling(10).mean()
    df['sma20'] = df['Close'].rolling(20).mean()
    
    df['above_sma5'] = df['prev_close'] > df['sma5'].shift(1)
    df['above_sma10'] = df['prev_close'] > df['sma10'].shift(1)
    df['above_sma20'] = df['prev_close'] > df['sma20'].shift(1)
    
    # Prior 5-day return (was it already running?)
    df['prior_5d_return'] = df['Close'].shift(1) / df['Close'].shift(6) - 1
    df['prior_5d_return_pct'] = df['prior_5d_return'] * 100
    
    # Prior 10-day return
    df['prior_10d_return'] = df['Close'].shift(1) / df['Close'].shift(11) - 1
    df['prior_10d_return_pct'] = df['prior_10d_return'] * 100
    
    # ATR for volatility context
    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['prev_close']), abs(df['Low'] - df['prev_close']))
    )
    df['atr14'] = df['tr'].rolling(14).mean()
    df['atr_pct'] = df['atr14'] / df['Close'] * 100
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['prev_rsi'] = df['rsi'].shift(1)
    
    # Consecutive gaps
    df['prev_gap'] = df['gap_pct'].shift(1)
    df['prev_prev_gap'] = df['gap_pct'].shift(2)
    df['consecutive_gap_up'] = (df['gap_pct'] > 3) & (df['prev_gap'] > 3)
    
    # Previous day's characteristics
    df['prev_intraday_return'] = df['intraday_return'].shift(1)
    df['prev_close_position'] = df['close_position'].shift(1)
    df['prev_relative_volume'] = df['relative_volume'].shift(1)
    
    # Next day (for multi-day analysis)
    df['next_open'] = df['Open'].shift(-1)
    df['next_close'] = df['Close'].shift(-1)
    df['next_high'] = df['High'].shift(-1)
    df['next_low'] = df['Low'].shift(-1)
    df['next_day_return'] = (df['next_close'] - df['next_open']) / df['next_open'] * 100
    df['next_day_gap'] = (df['next_open'] - df['Close']) / df['Close'] * 100
    df['next_day_total_return'] = (df['next_close'] - df['Close']) / df['Close'] * 100
    
    # Day 2 (two days after the gap)
    df['day2_return'] = df['next_day_total_return'].shift(-1)
    
    # Day of week
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = df['day_of_week'] == 0
    df['is_friday'] = df['day_of_week'] == 4
    
    # Price bucket
    df['price'] = df['Open']
    
    df['symbol'] = symbol
    return df.dropna(subset=['gap_pct', 'prev_close', 'atr14'])


def experiment_consecutive_gaps(df):
    """EXPERIMENT: What happens on consecutive gap days?"""
    print("\n" + "="*70)
    print("EXPERIMENT 2A: CONSECUTIVE GAP PATTERNS")
    print("="*70)
    
    results = {}
    
    # Day 1 gap up → Day 2 gap up (consecutive)
    consec_up = df[(df['gap_pct'] >= 5) & (df['prev_gap'] >= 5)]
    single_up = df[(df['gap_pct'] >= 5) & (df['prev_gap'] < 5) & (df['prev_gap'] > -5)]
    
    if len(consec_up) > 10 and len(single_up) > 10:
        print(f"\n  Consecutive gap ups (both days ≥5%): n={len(consec_up)}")
        print(f"    Avg intraday return: {consec_up['intraday_return'].mean():.2f}%")
        print(f"    Win rate (close > open): {(consec_up['intraday_return'] > 0).mean()*100:.1f}%")
        print(f"    Avg high from open: +{consec_up['high_from_open_pct'].mean():.2f}%")
        
        print(f"\n  Single gap ups (≥5%, no prior gap): n={len(single_up)}")
        print(f"    Avg intraday return: {single_up['intraday_return'].mean():.2f}%")
        print(f"    Win rate (close > open): {(single_up['intraday_return'] > 0).mean()*100:.1f}%")
        print(f"    Avg high from open: +{single_up['high_from_open_pct'].mean():.2f}%")
        
        # Statistical test
        t_stat, p_val = stats.ttest_ind(consec_up['intraday_return'], single_up['intraday_return'])
        print(f"\n    T-test p-value: {p_val:.4f} {'*** SIGNIFICANT ***' if p_val < 0.05 else '(not significant)'}")
        
        results['consecutive_vs_single'] = {
            'consecutive_count': len(consec_up),
            'consecutive_avg_return': consec_up['intraday_return'].mean(),
            'consecutive_win_rate': (consec_up['intraday_return'] > 0).mean() * 100,
            'single_count': len(single_up),
            'single_avg_return': single_up['intraday_return'].mean(),
            'single_win_rate': (single_up['intraday_return'] > 0).mean() * 100,
            'p_value': p_val,
        }
    
    # Gap up after gap DOWN (reversal pattern)
    reversal = df[(df['gap_pct'] >= 5) & (df['prev_gap'] <= -3)]
    if len(reversal) > 10:
        print(f"\n  Gap UP after prior gap DOWN (reversal): n={len(reversal)}")
        print(f"    Avg intraday return: {reversal['intraday_return'].mean():.2f}%")
        print(f"    Win rate: {(reversal['intraday_return'] > 0).mean()*100:.1f}%")
        results['reversal_gap'] = {
            'count': len(reversal),
            'avg_return': reversal['intraday_return'].mean(),
            'win_rate': (reversal['intraday_return'] > 0).mean() * 100,
        }
    
    return results


def experiment_prior_trend(df):
    """EXPERIMENT: Does the prior trend affect gap outcomes?"""
    print("\n" + "="*70)
    print("EXPERIMENT 2B: PRIOR TREND vs GAP OUTCOME")
    print("  Gap from a base vs. gap from an existing run")
    print("="*70)
    
    results = {}
    gap_ups = df[df['gap_pct'] >= 5]
    
    # Bucket by prior 5-day return
    trend_buckets = [
        ('Already down >10%', -100, -10),
        ('Down 5-10%', -10, -5),
        ('Flat (-5% to +5%)', -5, 5),
        ('Already up 5-10%', 5, 10),
        ('Already up 10-20%', 10, 20),
        ('Already running >20%', 20, 100),
    ]
    
    for label, low, high in trend_buckets:
        mask = (gap_ups['prior_5d_return_pct'] >= low) & (gap_ups['prior_5d_return_pct'] < high)
        subset = gap_ups[mask]
        if len(subset) > 10:
            win_rate = (subset['intraday_return'] > 0).mean() * 100
            avg_ret = subset['intraday_return'].mean()
            print(f"\n  {label}: n={len(subset)}")
            print(f"    Win rate: {win_rate:.1f}%")
            print(f"    Avg intraday return: {avg_ret:.2f}%")
            print(f"    Avg high from open: +{subset['high_from_open_pct'].mean():.2f}%")
            
            results[label] = {
                'count': len(subset),
                'win_rate': win_rate,
                'avg_return': avg_ret,
            }
    
    return results


def experiment_close_position(df):
    """EXPERIMENT: Where the stock closes in its range as a predictor."""
    print("\n" + "="*70)
    print("EXPERIMENT 2C: CLOSE POSITION IN RANGE (Day 1 → Day 2 Predictor)")
    print("  If a gap up stock closes near its HIGH, what happens next day?")
    print("="*70)
    
    results = {}
    gap_ups = df[(df['gap_pct'] >= 5)]
    
    # If gap up stock closed near high of day → strong close
    strong_close = gap_ups[gap_ups['close_position'] >= 0.7]
    weak_close = gap_ups[gap_ups['close_position'] <= 0.3]
    mid_close = gap_ups[(gap_ups['close_position'] > 0.3) & (gap_ups['close_position'] < 0.7)]
    
    for label, subset in [('Strong close (top 30% of range)', strong_close), 
                           ('Weak close (bottom 30% of range)', weak_close),
                           ('Mid close (middle 40%)', mid_close)]:
        if len(subset) > 10:
            next_day = subset['next_day_total_return'].dropna()
            next_gap = subset['next_day_gap'].dropna()
            
            print(f"\n  {label}: n={len(subset)}")
            print(f"    Avg NEXT DAY total return: {next_day.mean():.2f}%")
            print(f"    Avg NEXT DAY gap: {next_gap.mean():.2f}%")
            print(f"    Next day positive rate: {(next_day > 0).mean()*100:.1f}%")
            print(f"    Day 1 avg return: {subset['intraday_return'].mean():.2f}%")
            
            results[label] = {
                'count': len(subset),
                'next_day_avg_return': next_day.mean(),
                'next_day_positive_rate': (next_day > 0).mean() * 100,
                'next_day_avg_gap': next_gap.mean(),
            }
    
    if len(strong_close) > 10 and len(weak_close) > 10:
        t_stat, p_val = stats.ttest_ind(
            strong_close['next_day_total_return'].dropna(),
            weak_close['next_day_total_return'].dropna()
        )
        print(f"\n    Strong vs Weak close → Next day t-test p={p_val:.4f} {'*** SIGNIFICANT ***' if p_val < 0.05 else ''}")
        results['p_value_strong_vs_weak'] = p_val
    
    return results


def experiment_rsi_context(df):
    """EXPERIMENT: RSI before the gap as a filter."""
    print("\n" + "="*70)
    print("EXPERIMENT 2D: PRE-GAP RSI as FILTER")
    print("="*70)
    
    results = {}
    gap_ups = df[df['gap_pct'] >= 5]
    
    rsi_buckets = [
        ('Oversold RSI (<30)', 0, 30),
        ('Low RSI (30-40)', 30, 40),
        ('Neutral RSI (40-60)', 40, 60),
        ('High RSI (60-70)', 60, 70),
        ('Overbought RSI (>70)', 70, 100),
    ]
    
    for label, low, high in rsi_buckets:
        mask = (gap_ups['prev_rsi'] >= low) & (gap_ups['prev_rsi'] < high)
        subset = gap_ups[mask]
        if len(subset) > 10:
            win_rate = (subset['intraday_return'] > 0).mean() * 100
            avg_ret = subset['intraday_return'].mean()
            print(f"\n  {label}: n={len(subset)}")
            print(f"    Win rate: {win_rate:.1f}%")
            print(f"    Avg return: {avg_ret:.2f}%")
            
            results[label] = {
                'count': len(subset),
                'win_rate': win_rate,
                'avg_return': avg_ret,
            }
    
    return results


def experiment_gap_fade_timing(df):
    """EXPERIMENT: Optimal timing for gap fade shorts."""
    print("\n" + "="*70)
    print("EXPERIMENT 2E: GAP FADE CHARACTERISTICS")
    print("  When gaps fade, how much of the move is given back?")
    print("="*70)
    
    results = {}
    
    for gap_label, gap_min, gap_max in [('5-10%', 5, 10), ('10-20%', 10, 20), ('20%+', 20, 100)]:
        gap_ups = df[(df['gap_pct'] >= gap_min) & (df['gap_pct'] < gap_max)]
        if len(gap_ups) < 10:
            continue
            
        faders = gap_ups[gap_ups['intraday_return'] < 0]
        runners = gap_ups[gap_ups['intraday_return'] >= 0]
        
        fade_rate = len(faders) / len(gap_ups) * 100
        
        print(f"\n  Gap UP {gap_label}: n={len(gap_ups)}")
        print(f"    Fade rate: {fade_rate:.1f}%")
        
        if len(faders) > 5:
            # What % of the gap gets filled on fades?
            avg_fade = abs(faders['intraday_return'].mean())
            avg_gap = faders['gap_pct'].mean()
            fade_ratio = avg_fade / avg_gap * 100 if avg_gap > 0 else 0
            
            print(f"    When it fades:")
            print(f"      Avg fade magnitude: -{avg_fade:.2f}%")
            print(f"      Avg gap size: +{avg_gap:.2f}%")
            print(f"      Fade recaptures: {fade_ratio:.0f}% of the gap")
            print(f"      Avg low from open: {faders['low_from_open_pct'].mean():.2f}%")
            
            results[gap_label] = {
                'total': len(gap_ups),
                'fade_rate': fade_rate,
                'avg_fade_magnitude': avg_fade,
                'fade_ratio': fade_ratio,
            }
        
        if len(runners) > 5:
            print(f"    When it runs:")
            print(f"      Avg continuation: +{runners['intraday_return'].mean():.2f}%")
            print(f"      Avg high from open: +{runners['high_from_open_pct'].mean():.2f}%")
    
    return results


def experiment_day2_play(df):
    """EXPERIMENT: Is day 2 after a big gap more predictable?"""
    print("\n" + "="*70)
    print("EXPERIMENT 2F: THE DAY-2 PLAY")
    print("  After a big gap up, is day 2's behavior predictable?")
    print("="*70)
    
    results = {}
    big_gaps = df[df['gap_pct'] >= 10]
    
    # Segment by day 1 outcome
    d1_winners = big_gaps[big_gaps['intraday_return'] > 0]
    d1_losers = big_gaps[big_gaps['intraday_return'] <= 0]
    
    for label, subset in [('Day 1 was GREEN (continuation)', d1_winners),
                           ('Day 1 was RED (faded)', d1_losers)]:
        if len(subset) > 10:
            d2 = subset['next_day_total_return'].dropna()
            d2_gap = subset['next_day_gap'].dropna()
            
            print(f"\n  {label}: n={len(subset)}")
            print(f"    Day 2 avg return: {d2.mean():.2f}%")
            print(f"    Day 2 positive rate: {(d2 > 0).mean()*100:.1f}%")
            print(f"    Day 2 avg gap: {d2_gap.mean():.2f}%")
            
            # Is day 2 mean reverting or continuing?
            print(f"    Day 2 median return: {d2.median():.2f}%")
            
            results[label] = {
                'count': len(subset),
                'day2_avg_return': d2.mean(),
                'day2_positive_rate': (d2 > 0).mean() * 100,
                'day2_median_return': d2.median(),
            }
    
    # Strong close on day 1 + day 2
    if len(d1_winners) > 10:
        strong = d1_winners[d1_winners['close_position'] >= 0.7]
        if len(strong) > 5:
            d2_strong = strong['next_day_total_return'].dropna()
            print(f"\n  Day 1 GREEN + STRONG CLOSE → Day 2: n={len(strong)}")
            print(f"    Day 2 avg return: {d2_strong.mean():.2f}%")
            print(f"    Day 2 positive rate: {(d2_strong > 0).mean()*100:.1f}%")
            results['d1_green_strong_close'] = {
                'count': len(strong),
                'day2_avg_return': d2_strong.mean(),
                'day2_positive_rate': (d2_strong > 0).mean() * 100,
            }
    
    return results


def experiment_volume_profile(df):
    """EXPERIMENT: Volume characteristics as predictors."""
    print("\n" + "="*70)
    print("EXPERIMENT 2G: VOLUME PROFILE ANALYSIS")
    print("="*70)
    
    results = {}
    gap_ups = df[df['gap_pct'] >= 5]
    
    # Volume acceleration: is today's volume increasing vs yesterday?
    vol_accel = gap_ups[gap_ups['vol_change'] >= 2]  # Volume doubled
    vol_decel = gap_ups[gap_ups['vol_change'] < 1]   # Volume decreased
    
    if len(vol_accel) > 10:
        print(f"\n  Volume DOUBLED+ vs prior day: n={len(vol_accel)}")
        print(f"    Win rate: {(vol_accel['intraday_return'] > 0).mean()*100:.1f}%")
        print(f"    Avg return: {vol_accel['intraday_return'].mean():.2f}%")
    
    if len(vol_decel) > 10:
        print(f"\n  Volume DECREASED vs prior day: n={len(vol_decel)}")
        print(f"    Win rate: {(vol_decel['intraday_return'] > 0).mean()*100:.1f}%")
        print(f"    Avg return: {vol_decel['intraday_return'].mean():.2f}%")
    
    # Extreme relative volume
    for rvol_label, rvol_min, rvol_max in [('3-5x', 3, 5), ('5-10x', 5, 10), ('10x+', 10, 1000)]:
        mask = (gap_ups['relative_volume'] >= rvol_min) & (gap_ups['relative_volume'] < rvol_max)
        subset = gap_ups[mask]
        if len(subset) > 10:
            wr = (subset['intraday_return'] > 0).mean() * 100
            avg_r = subset['intraday_return'].mean()
            print(f"\n  RVOL {rvol_label}: n={len(subset)}")
            print(f"    Win rate: {wr:.1f}%")
            print(f"    Avg return: {avg_r:.2f}%")
            print(f"    Avg high from open: +{subset['high_from_open_pct'].mean():.2f}%")
            
            results[rvol_label] = {
                'count': len(subset),
                'win_rate': wr,
                'avg_return': avg_r,
            }
    
    return results


def experiment_monte_carlo(df, n_simulations=10000):
    """EXPERIMENT: Monte Carlo validation of the best strategy."""
    print("\n" + "="*70)
    print("EXPERIMENT 2H: MONTE CARLO VALIDATION")
    print("  Are the patterns real or just noise?")
    print("="*70)
    
    results = {}
    
    # Define the "best" strategy based on prior experiments
    # Gap up 5-15%, RVOL >= 3x, Price $2-20
    strategy_mask = (
        (df['gap_pct'] >= 5) & (df['gap_pct'] < 15) &
        (df['relative_volume'] >= 3) &
        (df['price'] >= 2) & (df['price'] <= 20)
    )
    strategy_trades = df[strategy_mask]['intraday_return'].dropna()
    
    if len(strategy_trades) < 30:
        print("  Not enough trades for Monte Carlo. Relaxing filters...")
        strategy_mask = (
            (df['gap_pct'] >= 5) &
            (df['relative_volume'] >= 2) &
            (df['price'] >= 2) & (df['price'] <= 50)
        )
        strategy_trades = df[strategy_mask]['intraday_return'].dropna()
    
    if len(strategy_trades) < 20:
        print("  Still not enough data for Monte Carlo validation.")
        return results
    
    observed_mean = strategy_trades.mean()
    observed_win_rate = (strategy_trades > 0).mean()
    n_trades = len(strategy_trades)
    
    print(f"\n  Strategy trades: n={n_trades}")
    print(f"  Observed avg return: {observed_mean:.3f}%")
    print(f"  Observed win rate: {observed_win_rate*100:.1f}%")
    
    # Monte Carlo: randomly sample same number of trades from ALL days
    all_returns = df['intraday_return'].dropna()
    
    random_means = []
    random_win_rates = []
    for _ in range(n_simulations):
        sample = np.random.choice(all_returns, size=n_trades, replace=True)
        random_means.append(sample.mean())
        random_win_rates.append((sample > 0).mean())
    
    random_means = np.array(random_means)
    random_win_rates = np.array(random_win_rates)
    
    # What percentile is our strategy vs random?
    mean_percentile = (random_means < observed_mean).mean() * 100
    wr_percentile = (random_win_rates < observed_win_rate).mean() * 100
    
    print(f"\n  Monte Carlo Results ({n_simulations} simulations):")
    print(f"    Strategy avg return percentile: {mean_percentile:.1f}th percentile")
    print(f"    Strategy win rate percentile: {wr_percentile:.1f}th percentile")
    print(f"    Random avg return: {random_means.mean():.3f}% (±{random_means.std():.3f}%)")
    print(f"    Random win rate: {random_win_rates.mean()*100:.1f}%")
    
    p_value = 1 - (mean_percentile / 100)
    print(f"    P-value (strategy is better than random): {p_value:.4f}")
    
    if p_value < 0.01:
        print("    *** HIGHLY SIGNIFICANT: Strategy edge is REAL (p < 0.01) ***")
    elif p_value < 0.05:
        print("    ** SIGNIFICANT: Strategy edge appears real (p < 0.05) **")
    elif p_value < 0.10:
        print("    * MARGINALLY SIGNIFICANT (p < 0.10) *")
    else:
        print("    NOT SIGNIFICANT: Edge may be noise")
    
    results['monte_carlo'] = {
        'n_trades': n_trades,
        'observed_mean': observed_mean,
        'observed_win_rate': observed_win_rate * 100,
        'mean_percentile': mean_percentile,
        'wr_percentile': wr_percentile,
        'p_value': p_value,
        'random_mean': random_means.mean(),
        'random_std': random_means.std(),
    }
    
    # Also test the SHORT strategy
    print("\n  --- SHORT STRATEGY Monte Carlo ---")
    short_mask = (
        (df['gap_pct'] >= 15) &
        (df['relative_volume'] >= 2) &
        (df['price'] >= 2)
    )
    short_trades = -df[short_mask]['intraday_return'].dropna()  # flip for short
    
    if len(short_trades) > 15:
        short_mean = short_trades.mean()
        short_wr = (short_trades > 0).mean()
        
        short_random_means = []
        for _ in range(n_simulations):
            sample = np.random.choice(-all_returns, size=len(short_trades), replace=True)
            short_random_means.append(sample.mean())
        
        short_random_means = np.array(short_random_means)
        short_percentile = (short_random_means < short_mean).mean() * 100
        short_p = 1 - (short_percentile / 100)
        
        print(f"  Short strategy trades: n={len(short_trades)}")
        print(f"  Short avg return: {short_mean:.3f}%")
        print(f"  Short win rate: {short_wr*100:.1f}%")
        print(f"  Short percentile: {short_percentile:.1f}th")
        print(f"  Short p-value: {short_p:.4f}")
        
        results['short_monte_carlo'] = {
            'n_trades': len(short_trades),
            'observed_mean': short_mean,
            'win_rate': short_wr * 100,
            'percentile': short_percentile,
            'p_value': short_p,
        }
    
    return results


def run_experiment_2():
    """Main runner."""
    print("="*70)
    print("EXPERIMENT 2: ADVANCED PATTERN DISCOVERY")
    print("="*70)
    
    data = download_all(FOCUS_UNIVERSE, period="5y")
    
    # Build feature-rich dataset
    all_dfs = []
    for sym, raw_df in data.items():
        if sym in ("SPY", "QQQ", "IWM"):
            continue
        featured = build_features(raw_df, sym)
        all_dfs.append(featured)
    
    df = pd.concat(all_dfs, ignore_index=False)
    print(f"\nTotal trading days in dataset: {len(df)}")
    print(f"Unique symbols: {df['symbol'].nunique()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    all_results = {}
    
    # Run all experiments
    all_results['2A_consecutive'] = experiment_consecutive_gaps(df)
    all_results['2B_prior_trend'] = experiment_prior_trend(df)
    all_results['2C_close_position'] = experiment_close_position(df)
    all_results['2D_rsi_context'] = experiment_rsi_context(df)
    all_results['2E_gap_fade'] = experiment_gap_fade_timing(df)
    all_results['2F_day2_play'] = experiment_day2_play(df)
    all_results['2G_volume_profile'] = experiment_volume_profile(df)
    all_results['2H_monte_carlo'] = experiment_monte_carlo(df)
    
    # Save results
    results_file = os.path.join(RESULTS_DIR, 'experiment_2_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")
    
    return all_results


if __name__ == "__main__":
    run_experiment_2()
