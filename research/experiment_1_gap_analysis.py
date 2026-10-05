"""
EXPERIMENT 1: Deep Gap Analysis Engine
=======================================
Pulls years of daily data for a broad universe of small-cap stocks.
Identifies every gap event (open vs. previous close) and tracks what happened:
- Did the gap continue (momentum)?
- Did it fade (mean reversion)?
- What was the optimal exit time?
- How does gap size affect outcome?
- Day-of-week effects?
- Market context (SPY trend) impact?

This is the foundational research for the trading bot.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# UNIVERSE: Small-cap stocks known for momentum/gap behavior
# Mix of sectors, float sizes, price ranges
# ============================================================
SMALL_CAP_UNIVERSE = [
    # Biotech / Pharma (frequent catalysts: FDA, trials)
    "SAVA", "BNGO", "CLOV", "WISH", "WKHS", "GOEV", "NKLA", "QS",
    "MVIS", "CLVS", "BBIG", "ATER", "PROG", "FAMI", "CTRM", "ZYNE",
    "VERU", "IMPP", "MULN", "BIOR", "APRE", "PRAX", "RCKT", "EDIT",
    "BEAM", "CRSP", "NTLA", "VERV", "ABCL", "IOVA",
    # Tech / EV / Growth (volatile, news-driven)
    "SOFI", "PLTR", "NIO", "XPEV", "LI", "LCID", "RIVN", "FSR",
    "JOBY", "LILM", "BKKT", "RBLX", "DKNG", "SKLZ", "CLOV",
    "FUBO", "BARK", "OPAD", "IRNT", "TMC", "LIDR", "GENI",
    # Meme / High-volatility / Short-squeeze targets
    "AMC", "BB", "BBBY", "SPCE", "CLOV", "EXPR", "KOSS", "NAKD",
    "SNDL", "TLRY", "CGC", "APHA", "ACB",
    # Recent IPOs / SPACs (volatile, less institutional coverage)
    "DNA", "IONQ", "RKLB", "ASTR", "MNTS", "ASTS", "RDW", "VORB",
    "MAPS", "HIMS", "BARK", "PAYO", "SKIN", "DM", "OUST",
    # Energy / Mining (commodity catalyst driven)
    "TELL", "NEXT", "RIG", "BORR", "SWN", "AR", "CTRA",
    # Other small-cap movers
    "UPST", "AFRM", "HOOD", "COIN", "MARA", "RIOT", "HUT", "BITF",
    "SI", "OPEN", "OFPD", "REAL", "RDFN", "OPCH", "DOCS",
    # Additional small/micro caps
    "GSAT", "ORGN", "MVST", "REE", "PTRA", "GBS", "CANO",
    "BGFV", "PRTS", "PERI", "VNET", "IQ", "TIGR", "FUTU",
    "BABA", "JD", "PDD", "BIDU",  # Chinese ADRs (gap-heavy)
    # Penny-ish / micro-cap (high gap frequency)
    "CENN", "AGTC", "AEMD", "PHUN", "DWAC", "CFVI",
    # SPY / QQQ for market context
    "SPY", "QQQ"
]

# Remove duplicates
SMALL_CAP_UNIVERSE = list(set(SMALL_CAP_UNIVERSE))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def download_data(symbols, period="5y"):
    """Download historical daily data for all symbols."""
    print(f"Downloading data for {len(symbols)} symbols (period={period})...")
    all_data = {}
    failed = []
    
    for i, sym in enumerate(symbols):
        if sym in ("SPY", "QQQ"):
            continue  # download separately
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=period, auto_adjust=True)
            if len(df) > 50:  # need enough data
                all_data[sym] = df
                if (i + 1) % 20 == 0:
                    print(f"  ... downloaded {i+1}/{len(symbols)} symbols")
            else:
                failed.append(sym)
        except Exception as e:
            failed.append(sym)
    
    # Download market context
    for market_sym in ["SPY", "QQQ"]:
        try:
            ticker = yf.Ticker(market_sym)
            df = ticker.history(period=period, auto_adjust=True)
            all_data[market_sym] = df
        except:
            pass
    
    print(f"Successfully downloaded: {len(all_data)} symbols")
    print(f"Failed/insufficient data: {len(failed)} symbols")
    return all_data


def identify_gaps(df, symbol):
    """
    Identify all gap events in a stock's history.
    A gap = (today's open - yesterday's close) / yesterday's close
    """
    df = df.copy()
    df['prev_close'] = df['Close'].shift(1)
    df['gap_pct'] = (df['Open'] - df['prev_close']) / df['prev_close'] * 100
    
    # Intraday return (open to close)
    df['intraday_return'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    # High/Low relative to open
    df['high_from_open'] = (df['High'] - df['Open']) / df['Open'] * 100
    df['low_from_open'] = (df['Low'] - df['Open']) / df['Open'] * 100
    
    # Did the gap continue or fade?
    # For gap UP: continuation = close > open; fade = close < open
    # For gap DOWN: continuation = close < open; fade = close > open
    df['gap_continued'] = np.where(
        df['gap_pct'] > 0,
        df['Close'] > df['Open'],  # Gap up: continued if closed above open
        df['Close'] < df['Open']   # Gap down: continued if closed below open
    )
    
    # Full gap fill? (price returned to previous close)
    df['gap_filled'] = np.where(
        df['gap_pct'] > 0,
        df['Low'] <= df['prev_close'],   # Gap up filled if low touched prev close
        df['High'] >= df['prev_close']   # Gap down filled if high touched prev close
    )
    
    # Volume relative to 20-day average
    df['avg_volume_20d'] = df['Volume'].rolling(20).mean()
    df['relative_volume'] = df['Volume'] / df['avg_volume_20d']
    
    # Volatility context (20-day ATR as % of price)
    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['prev_close']),
            abs(df['Low'] - df['prev_close'])
        )
    )
    df['atr_20'] = df['tr'].rolling(20).mean()
    df['atr_pct'] = df['atr_20'] / df['Close'] * 100
    
    # Next day return (for multi-day analysis)
    df['next_day_return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['next_day_gap'] = df['gap_pct'].shift(-1)
    
    # Day of week
    df['day_of_week'] = df.index.dayofweek  # 0=Mon, 4=Fri
    df['day_name'] = df.index.day_name()
    
    # Price bucket
    df['price_bucket'] = pd.cut(df['Open'], bins=[0, 2, 5, 10, 20, 50, 1000],
                                 labels=['$0-2', '$2-5', '$5-10', '$10-20', '$20-50', '$50+'])
    
    df['symbol'] = symbol
    df.dropna(subset=['gap_pct', 'prev_close'], inplace=True)
    
    return df


def analyze_gaps(all_data):
    """Run comprehensive gap analysis across all stocks."""
    
    # Combine all gap data
    all_gaps = []
    for sym, df in all_data.items():
        if sym in ("SPY", "QQQ"):
            continue
        gap_df = identify_gaps(df, sym)
        all_gaps.append(gap_df)
    
    gaps = pd.concat(all_gaps, ignore_index=False)
    
    # Get SPY data for market context
    spy_data = None
    if "SPY" in all_data:
        spy = all_data["SPY"].copy()
        spy['spy_sma20'] = spy['Close'].rolling(20).mean()
        spy['spy_trend'] = np.where(spy['Close'] > spy['spy_sma20'], 'bullish', 'bearish')
        spy['spy_daily_return'] = spy['Close'].pct_change() * 100
        spy_data = spy[['spy_trend', 'spy_daily_return']].copy()
    
    # Merge market context
    if spy_data is not None:
        gaps = gaps.join(spy_data, how='left')
    else:
        gaps['spy_trend'] = 'unknown'
        gaps['spy_daily_return'] = 0
    
    results = {}
    
    # ================================================================
    # ANALYSIS 1: Gap Size vs. Continuation Rate
    # ================================================================
    print("\n" + "="*70)
    print("ANALYSIS 1: Gap Size vs. Continuation Rate")
    print("="*70)
    
    gap_buckets = [
        ('3-5%', 3, 5), ('5-8%', 5, 8), ('8-12%', 8, 12),
        ('12-20%', 12, 20), ('20-30%', 20, 30), ('30-50%', 30, 50),
        ('50%+', 50, 200)
    ]
    
    gap_up_analysis = []
    gap_down_analysis = []
    
    for label, low, high in gap_buckets:
        # Gap UP analysis
        mask_up = (gaps['gap_pct'] >= low) & (gaps['gap_pct'] < high)
        subset_up = gaps[mask_up]
        if len(subset_up) > 10:
            row = {
                'gap_range': label,
                'count': len(subset_up),
                'continuation_rate': subset_up['gap_continued'].mean() * 100,
                'avg_intraday_return': subset_up['intraday_return'].mean(),
                'median_intraday_return': subset_up['intraday_return'].median(),
                'gap_fill_rate': subset_up['gap_filled'].mean() * 100,
                'avg_high_from_open': subset_up['high_from_open'].mean(),
                'avg_low_from_open': subset_up['low_from_open'].mean(),
                'avg_relative_volume': subset_up['relative_volume'].mean(),
                'std_intraday_return': subset_up['intraday_return'].std(),
            }
            gap_up_analysis.append(row)
            print(f"\n  Gap UP {label}: n={row['count']}")
            print(f"    Continuation rate: {row['continuation_rate']:.1f}%")
            print(f"    Avg intraday return (open→close): {row['avg_intraday_return']:.2f}%")
            print(f"    Median intraday return: {row['median_intraday_return']:.2f}%")
            print(f"    Gap fill rate: {row['gap_fill_rate']:.1f}%")
            print(f"    Avg high from open: +{row['avg_high_from_open']:.2f}%")
            print(f"    Avg low from open: {row['avg_low_from_open']:.2f}%")
        
        # Gap DOWN analysis
        mask_down = (gaps['gap_pct'] <= -low) & (gaps['gap_pct'] > -high)
        subset_down = gaps[mask_down]
        if len(subset_down) > 10:
            row = {
                'gap_range': f'-{label}',
                'count': len(subset_down),
                'continuation_rate': subset_down['gap_continued'].mean() * 100,
                'avg_intraday_return': subset_down['intraday_return'].mean(),
                'median_intraday_return': subset_down['intraday_return'].median(),
                'gap_fill_rate': subset_down['gap_filled'].mean() * 100,
                'avg_high_from_open': subset_down['high_from_open'].mean(),
                'avg_low_from_open': subset_down['low_from_open'].mean(),
            }
            gap_down_analysis.append(row)
    
    results['gap_up_by_size'] = gap_up_analysis
    results['gap_down_by_size'] = gap_down_analysis
    
    # ================================================================
    # ANALYSIS 2: Day-of-Week Effects
    # ================================================================
    print("\n" + "="*70)
    print("ANALYSIS 2: Day-of-Week Effects on Gaps (≥5%)")
    print("="*70)
    
    sig_gaps = gaps[(gaps['gap_pct'].abs() >= 5)]
    dow_analysis = []
    
    for day_num, day_name in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']):
        day_gaps = sig_gaps[sig_gaps['day_of_week'] == day_num]
        if len(day_gaps) > 20:
            # Separate up and down
            up = day_gaps[day_gaps['gap_pct'] > 0]
            down = day_gaps[day_gaps['gap_pct'] < 0]
            
            row = {
                'day': day_name,
                'total_gaps': len(day_gaps),
                'gap_up_count': len(up),
                'gap_down_count': len(down),
                'up_continuation_rate': up['gap_continued'].mean() * 100 if len(up) > 5 else None,
                'up_avg_return': up['intraday_return'].mean() if len(up) > 5 else None,
                'down_continuation_rate': down['gap_continued'].mean() * 100 if len(down) > 5 else None,
                'up_gap_fill_rate': up['gap_filled'].mean() * 100 if len(up) > 5 else None,
            }
            dow_analysis.append(row)
            print(f"\n  {day_name}: {row['total_gaps']} gaps")
            if row['up_continuation_rate'] is not None:
                print(f"    Gap UP continuation: {row['up_continuation_rate']:.1f}% (n={len(up)})")
                print(f"    Gap UP avg return: {row['up_avg_return']:.2f}%")
                print(f"    Gap UP fill rate: {row['up_gap_fill_rate']:.1f}%")
    
    results['day_of_week'] = dow_analysis
    
    # ================================================================
    # ANALYSIS 3: Market Context Impact
    # ================================================================
    print("\n" + "="*70)
    print("ANALYSIS 3: Market Context Impact (SPY trend vs gap outcomes)")
    print("="*70)
    
    sig_up_gaps = gaps[(gaps['gap_pct'] >= 5)]
    market_analysis = []
    
    for trend in ['bullish', 'bearish']:
        trend_gaps = sig_up_gaps[sig_up_gaps['spy_trend'] == trend]
        if len(trend_gaps) > 20:
            row = {
                'market_trend': trend,
                'count': len(trend_gaps),
                'continuation_rate': trend_gaps['gap_continued'].mean() * 100,
                'avg_intraday_return': trend_gaps['intraday_return'].mean(),
                'gap_fill_rate': trend_gaps['gap_filled'].mean() * 100,
                'avg_high_from_open': trend_gaps['high_from_open'].mean(),
            }
            market_analysis.append(row)
            print(f"\n  SPY {trend.upper()}: n={row['count']}")
            print(f"    Continuation rate: {row['continuation_rate']:.1f}%")
            print(f"    Avg intraday return: {row['avg_intraday_return']:.2f}%")
            print(f"    Gap fill rate: {row['gap_fill_rate']:.1f}%")
    
    results['market_context'] = market_analysis
    
    # ================================================================
    # ANALYSIS 4: Volume Confirmation
    # ================================================================
    print("\n" + "="*70)
    print("ANALYSIS 4: Relative Volume as Edge Filter")
    print("="*70)
    
    sig_up = gaps[(gaps['gap_pct'] >= 5)]
    vol_buckets = [
        ('< 1x (low vol)', 0, 1),
        ('1-2x (normal)', 1, 2),
        ('2-3x (elevated)', 2, 3),
        ('3-5x (high)', 3, 5),
        ('5-10x (very high)', 5, 10),
        ('10x+ (extreme)', 10, 100),
    ]
    
    vol_analysis = []
    for label, low, high in vol_buckets:
        mask = (sig_up['relative_volume'] >= low) & (sig_up['relative_volume'] < high)
        subset = sig_up[mask]
        if len(subset) > 10:
            row = {
                'volume_range': label,
                'count': len(subset),
                'continuation_rate': subset['gap_continued'].mean() * 100,
                'avg_intraday_return': subset['intraday_return'].mean(),
                'median_intraday_return': subset['intraday_return'].median(),
                'gap_fill_rate': subset['gap_filled'].mean() * 100,
                'avg_high_from_open': subset['high_from_open'].mean(),
            }
            vol_analysis.append(row)
            print(f"\n  {label}: n={row['count']}")
            print(f"    Continuation rate: {row['continuation_rate']:.1f}%")
            print(f"    Avg return: {row['avg_intraday_return']:.2f}%")
            print(f"    Gap fill rate: {row['gap_fill_rate']:.1f}%")
    
    results['volume_filter'] = vol_analysis
    
    # ================================================================
    # ANALYSIS 5: Price Level Impact
    # ================================================================
    print("\n" + "="*70)
    print("ANALYSIS 5: Price Level vs Gap Outcome")
    print("="*70)
    
    price_analysis = []
    for bucket in ['$0-2', '$2-5', '$5-10', '$10-20', '$20-50', '$50+']:
        mask = (sig_up['price_bucket'] == bucket)
        subset = sig_up[mask]
        if len(subset) > 10:
            row = {
                'price_range': bucket,
                'count': len(subset),
                'continuation_rate': subset['gap_continued'].mean() * 100,
                'avg_intraday_return': subset['intraday_return'].mean(),
                'gap_fill_rate': subset['gap_filled'].mean() * 100,
                'avg_high_from_open': subset['high_from_open'].mean(),
                'avg_low_from_open': subset['low_from_open'].mean(),
            }
            price_analysis.append(row)
            print(f"\n  {bucket}: n={row['count']}")
            print(f"    Continuation rate: {row['continuation_rate']:.1f}%")
            print(f"    Avg return: {row['avg_intraday_return']:.2f}%")
            print(f"    Avg high from open: +{row['avg_high_from_open']:.2f}%")
    
    results['price_level'] = price_analysis
    
    # ================================================================
    # ANALYSIS 6: Multi-Day Momentum (Day 2 after a gap)
    # ================================================================
    print("\n" + "="*70)
    print("ANALYSIS 6: Next-Day Follow-Through After Gap Up")
    print("="*70)
    
    # After a gap up that CONTINUED, what happens the next day?
    continued_gaps = sig_up[sig_up['gap_continued'] == True]
    faded_gaps = sig_up[sig_up['gap_continued'] == False]
    
    multi_day = []
    if len(continued_gaps) > 20:
        next_day_after_cont = continued_gaps['next_day_return'].dropna()
        row = {
            'scenario': 'Gap up + continuation → Next day',
            'count': len(next_day_after_cont),
            'avg_next_day_return': next_day_after_cont.mean() * 100,
            'median_next_day_return': next_day_after_cont.median() * 100,
            'positive_rate': (next_day_after_cont > 0).mean() * 100,
            'avg_next_day_gap': continued_gaps['next_day_gap'].dropna().mean(),
        }
        multi_day.append(row)
        print(f"\n  After gap up + continuation: n={row['count']}")
        print(f"    Avg next day return: {row['avg_next_day_return']:.2f}%")
        print(f"    Positive next day rate: {row['positive_rate']:.1f}%")
    
    if len(faded_gaps) > 20:
        next_day_after_fade = faded_gaps['next_day_return'].dropna()
        row = {
            'scenario': 'Gap up + fade → Next day',
            'count': len(next_day_after_fade),
            'avg_next_day_return': next_day_after_fade.mean() * 100,
            'median_next_day_return': next_day_after_fade.median() * 100,
            'positive_rate': (next_day_after_fade > 0).mean() * 100,
        }
        multi_day.append(row)
        print(f"\n  After gap up + FADE: n={row['count']}")
        print(f"    Avg next day return: {row['avg_next_day_return']:.2f}%")
        print(f"    Positive next day rate: {row['positive_rate']:.1f}%")
    
    results['multi_day'] = multi_day
    
    # ================================================================
    # ANALYSIS 7: Volatility Context
    # ================================================================
    print("\n" + "="*70)
    print("ANALYSIS 7: Prior Volatility vs Gap Outcome")
    print("="*70)
    
    vol_context = []
    atr_buckets = [
        ('Low vol (<3% ATR)', 0, 3),
        ('Medium vol (3-6%)', 3, 6),
        ('High vol (6-10%)', 6, 10),
        ('Extreme vol (>10%)', 10, 100),
    ]
    
    for label, low, high in atr_buckets:
        mask = (sig_up['atr_pct'] >= low) & (sig_up['atr_pct'] < high)
        subset = sig_up[mask]
        if len(subset) > 10:
            row = {
                'volatility': label,
                'count': len(subset),
                'continuation_rate': subset['gap_continued'].mean() * 100,
                'avg_intraday_return': subset['intraday_return'].mean(),
                'gap_fill_rate': subset['gap_filled'].mean() * 100,
            }
            vol_context.append(row)
            print(f"\n  {label}: n={row['count']}")
            print(f"    Continuation rate: {row['continuation_rate']:.1f}%")
            print(f"    Avg return: {row['avg_intraday_return']:.2f}%")
    
    results['volatility_context'] = vol_context
    
    # ================================================================
    # ANALYSIS 8: The "Goldilocks Zone" - Combined Filters
    # ================================================================
    print("\n" + "="*70)
    print("ANALYSIS 8: COMBINED FILTER OPTIMIZATION")
    print("Finding the sweet spot of gap size + volume + price + market")
    print("="*70)
    
    # Test various filter combinations
    combos = []
    for gap_min in [3, 5, 8, 10, 15, 20]:
        for gap_max in [10, 15, 20, 30, 50]:
            if gap_max <= gap_min:
                continue
            for rvol_min in [1.5, 2, 3, 5]:
                for price_min in [1, 2, 5]:
                    for price_max in [10, 20, 50]:
                        if price_max <= price_min:
                            continue
                        mask = (
                            (gaps['gap_pct'] >= gap_min) & 
                            (gaps['gap_pct'] < gap_max) &
                            (gaps['relative_volume'] >= rvol_min) &
                            (gaps['Open'] >= price_min) &
                            (gaps['Open'] <= price_max)
                        )
                        subset = gaps[mask]
                        if len(subset) >= 30:  # Need statistical significance
                            cont_rate = subset['gap_continued'].mean()
                            avg_ret = subset['intraday_return'].mean()
                            median_ret = subset['intraday_return'].median()
                            fill_rate = subset['gap_filled'].mean()
                            
                            # Calculate expected value
                            avg_win = subset[subset['intraday_return'] > 0]['intraday_return'].mean() if (subset['intraday_return'] > 0).any() else 0
                            avg_loss = subset[subset['intraday_return'] <= 0]['intraday_return'].mean() if (subset['intraday_return'] <= 0).any() else 0
                            win_rate = (subset['intraday_return'] > 0).mean()
                            expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss
                            
                            # Sharpe-like ratio
                            if subset['intraday_return'].std() > 0:
                                sharpe = avg_ret / subset['intraday_return'].std()
                            else:
                                sharpe = 0
                            
                            combos.append({
                                'gap_range': f'{gap_min}-{gap_max}%',
                                'rvol_min': rvol_min,
                                'price_range': f'${price_min}-${price_max}',
                                'count': len(subset),
                                'continuation_rate': cont_rate * 100,
                                'avg_return': avg_ret,
                                'median_return': median_ret,
                                'win_rate': win_rate * 100,
                                'avg_win': avg_win,
                                'avg_loss': avg_loss,
                                'expected_value': expected_value,
                                'sharpe': sharpe,
                                'gap_fill_rate': fill_rate * 100,
                            })
    
    # Sort by expected value * sqrt(count) to balance edge strength with sample size
    combos_df = pd.DataFrame(combos)
    if len(combos_df) > 0:
        combos_df['score'] = combos_df['expected_value'] * np.sqrt(combos_df['count'])
        combos_df = combos_df.sort_values('score', ascending=False)
        
        print("\n  TOP 20 FILTER COMBINATIONS (sorted by edge score):")
        print("  " + "-"*120)
        top = combos_df.head(20)
        for _, row in top.iterrows():
            print(f"  Gap {row['gap_range']:>10} | RVOL≥{row['rvol_min']:>3.1f}x | Price {row['price_range']:>10} | "
                  f"n={row['count']:>4} | WR={row['win_rate']:>5.1f}% | "
                  f"AvgRet={row['avg_return']:>+6.2f}% | EV={row['expected_value']:>+6.2f}% | "
                  f"Sharpe={row['sharpe']:>5.3f}")
        
        results['best_combinations_long'] = top.to_dict('records')
        
        # Also find best SHORT opportunities (gap down continuation)
        print("\n\n  TOP SHORT OPPORTUNITIES (Gap DOWN filter combinations):")
        short_combos = []
        for gap_min in [3, 5, 8, 10, 15]:
            for gap_max in [10, 15, 20, 30]:
                if gap_max <= gap_min:
                    continue
                for rvol_min in [1.5, 2, 3, 5]:
                    for price_min in [2, 5]:
                        for price_max in [20, 50]:
                            if price_max <= price_min:
                                continue
                            mask = (
                                (gaps['gap_pct'] <= -gap_min) & 
                                (gaps['gap_pct'] > -gap_max) &
                                (gaps['relative_volume'] >= rvol_min) &
                                (gaps['Open'] >= price_min) &
                                (gaps['Open'] <= price_max)
                            )
                            subset = gaps[mask]
                            if len(subset) >= 20:
                                # For shorts: profit when price drops (negative return)
                                short_return = -subset['intraday_return']  # flip sign
                                win_rate = (short_return > 0).mean()
                                avg_win = short_return[short_return > 0].mean() if (short_return > 0).any() else 0
                                avg_loss = short_return[short_return <= 0].mean() if (short_return <= 0).any() else 0
                                expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss
                                
                                short_combos.append({
                                    'gap_range': f'-{gap_min} to -{gap_max}%',
                                    'rvol_min': rvol_min,
                                    'price_range': f'${price_min}-${price_max}',
                                    'count': len(subset),
                                    'win_rate': win_rate * 100,
                                    'avg_return': expected_value,
                                    'expected_value': expected_value,
                                })
        
        if short_combos:
            short_df = pd.DataFrame(short_combos)
            short_df['score'] = short_df['expected_value'] * np.sqrt(short_df['count'])
            short_df = short_df.sort_values('score', ascending=False)
            print("  " + "-"*100)
            for _, row in short_df.head(10).iterrows():
                print(f"  Gap {row['gap_range']:>15} | RVOL≥{row['rvol_min']:>3.1f}x | Price {row['price_range']:>10} | "
                      f"n={row['count']:>4} | WR={row['win_rate']:>5.1f}% | EV={row['expected_value']:>+6.2f}%")
            
            results['best_combinations_short'] = short_df.head(10).to_dict('records')
    
    # ================================================================
    # ANALYSIS 9: Gap Up Fade Strategy (Shorting gap-up failures)
    # ================================================================
    print("\n" + "="*70)
    print("ANALYSIS 9: Gap-Up Fade (Short) Strategy")
    print("  When a gap up FAILS, how profitable is shorting it?")
    print("="*70)
    
    large_gaps_up = gaps[(gaps['gap_pct'] >= 10) & (gaps['Open'] >= 2)]
    if len(large_gaps_up) > 20:
        # Strategy: short when close < open (gap faded)
        faders = large_gaps_up[large_gaps_up['intraday_return'] < 0]
        runners = large_gaps_up[large_gaps_up['intraday_return'] >= 0]
        
        print(f"\n  Total 10%+ gap ups: {len(large_gaps_up)}")
        print(f"  Faded (close < open): {len(faders)} ({len(faders)/len(large_gaps_up)*100:.1f}%)")
        print(f"  Ran (close >= open): {len(runners)} ({len(runners)/len(large_gaps_up)*100:.1f}%)")
        print(f"\n  When they FADE:")
        print(f"    Avg fade magnitude: {faders['intraday_return'].mean():.2f}%")
        print(f"    Avg low from open: {faders['low_from_open'].mean():.2f}%")
        print(f"\n  When they RUN:")
        print(f"    Avg continuation magnitude: {runners['intraday_return'].mean():.2f}%")
        print(f"    Avg high from open: +{runners['high_from_open'].mean():.2f}%")
        
        results['gap_fade_short'] = {
            'total': len(large_gaps_up),
            'fade_rate': len(faders)/len(large_gaps_up)*100,
            'avg_fade_magnitude': faders['intraday_return'].mean(),
            'avg_run_magnitude': runners['intraday_return'].mean(),
        }
    
    # ================================================================
    # SUMMARY STATISTICS
    # ================================================================
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    total_gaps = len(gaps[gaps['gap_pct'].abs() >= 3])
    print(f"\n  Total trading days analyzed: {len(gaps)}")
    print(f"  Total significant gaps (≥3%): {total_gaps}")
    print(f"  Unique symbols: {gaps['symbol'].nunique()}")
    print(f"  Date range: {gaps.index.min()} to {gaps.index.max()}")
    
    results['summary'] = {
        'total_trading_days': len(gaps),
        'total_significant_gaps': total_gaps,
        'unique_symbols': int(gaps['symbol'].nunique()),
    }
    
    return results, gaps


def run_experiment():
    """Main entry point."""
    print("="*70)
    print("EXPERIMENT 1: DEEP GAP ANALYSIS ENGINE")
    print("Searching for exploitable patterns in small-cap gaps...")
    print("="*70)
    
    # Download data
    all_data = download_data(SMALL_CAP_UNIVERSE, period="5y")
    
    # Run analysis
    results, gaps_df = analyze_gaps(all_data)
    
    # Save results
    results_file = os.path.join(RESULTS_DIR, 'experiment_1_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")
    
    return results, gaps_df


if __name__ == "__main__":
    results, gaps_df = run_experiment()
