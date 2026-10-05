"""
EXPERIMENT 3: DEEP DIVE ON THE STRONGEST SIGNALS
=================================================
Experiment 1 revealed MASSIVE findings:

KEY DISCOVERY: RVOL (Relative Volume) ≥ 5x is THE edge.
  - Gap 3-10%, RVOL≥5x, Price $2-$20: 83.3% win rate, +13.89% avg return (n=66)
  - Gap 3-10%, RVOL≥5x, Price $2-$50: 82.3% win rate, +13.69% avg return (n=79)
  - RVOL 5-10x overall: 64.8% continuation, +6.54% avg return
  
ALSO DISCOVERED:
  - SPY bullish context: 43.5% continuation vs 34.1% when bearish
  - Mondays have HIGHEST continuation rate for gap ups (41.2%)  
  - Tuesdays are WORST (-1.53% avg return)
  - Low volatility stocks (ATR<3%) that gap: 77.8% continuation (but small sample)
  - Gap DOWN 30-50%: bounce is MASSIVE (+35.7% avg intraday) — mean reversion play?
  - Gap DOWN 50%+: insane bounce (+621% avg) — dominated by extreme events

This experiment will:
1. Stress-test the RVOL≥5x finding with stricter statistical methods
2. Look for intraday timing patterns (when to enter/exit)
3. Build a full strategy simulation with realistic assumptions
4. Test the gap-down mean reversion bounce play
5. Find the OPTIMAL combination of all filters
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

UNIVERSE = [
    "SOFI", "PLTR", "NIO", "XPEV", "LCID", "RIVN", "AMC", "MARA", "RIOT",
    "COIN", "HOOD", "UPST", "AFRM", "DKNG", "FUBO", "IONQ", "RKLB",
    "ASTS", "HIMS", "DNA", "QS", "NKLA", "GOEV", "MULN", "SNDL",
    "TLRY", "CGC", "SPCE", "GSAT", "MVST", "MVIS", "WKHS",
    "CLOV", "WISH", "SKLZ", "OPEN", "RDFN", "DOCS", "SKIN",
    "BKKT", "JOBY", "LILM", "BARK", "REAL", "PAYO",
    "SAVA", "BNGO", "EDIT", "BEAM", "CRSP", "NTLA", "IOVA",
    "BITF", "HUT", "BB", "ACB", "GENI", "RBLX",
    "LI", "BABA", "JD", "PDD", "BIDU", "IQ", "TIGR", "FUTU",
    "BGFV", "VNET", "CENN", "PHUN", "RIG", "AR", "CTRA", "BORR",
    "VERV", "ABCL", "PRAX",
    "SPY", "QQQ", "IWM",
]
UNIVERSE = list(set(UNIVERSE))


def download(symbols, period="5y"):
    data = {}
    for sym in symbols:
        try:
            df = yf.Ticker(sym).history(period=period, auto_adjust=True)
            if len(df) > 60:
                data[sym] = df
        except:
            pass
    print(f"Downloaded {len(data)} symbols")
    return data


def build_dataset(all_data):
    """Build enriched dataset."""
    frames = []
    spy_df = all_data.get("SPY")
    
    for sym, df in all_data.items():
        if sym in ("SPY", "QQQ", "IWM"):
            continue
        d = df.copy()
        d['prev_close'] = d['Close'].shift(1)
        d['gap_pct'] = (d['Open'] - d['prev_close']) / d['prev_close'] * 100
        d['intraday_return'] = (d['Close'] - d['Open']) / d['Open'] * 100
        d['high_from_open'] = (d['High'] - d['Open']) / d['Open'] * 100
        d['low_from_open'] = (d['Low'] - d['Open']) / d['Open'] * 100
        d['close_in_range'] = np.where(d['High'] != d['Low'], (d['Close'] - d['Low'])/(d['High'] - d['Low']), 0.5)
        d['open_in_range'] = np.where(d['High'] != d['Low'], (d['Open'] - d['Low'])/(d['High'] - d['Low']), 0.5)
        
        # Volume
        d['vol_sma20'] = d['Volume'].rolling(20).mean()
        d['rvol'] = d['Volume'] / d['vol_sma20']
        
        # ATR
        d['tr'] = np.maximum(d['High']-d['Low'], np.maximum(abs(d['High']-d['prev_close']), abs(d['Low']-d['prev_close'])))
        d['atr14'] = d['tr'].rolling(14).mean()
        d['atr_pct'] = d['atr14'] / d['Close'] * 100
        
        # Prior trend
        d['prior_5d_ret'] = d['Close'].shift(1) / d['Close'].shift(6) - 1
        d['prior_10d_ret'] = d['Close'].shift(1) / d['Close'].shift(11) - 1
        d['sma20'] = d['Close'].rolling(20).mean()
        d['above_sma20'] = d['prev_close'] > d['sma20'].shift(1)
        
        # RSI
        delta = d['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        d['rsi'] = 100 - (100 / (1 + rs))
        d['prev_rsi'] = d['rsi'].shift(1)
        
        # Multi-day
        d['next_day_return'] = (d['Close'].shift(-1) - d['Close']) / d['Close'] * 100
        d['next_day_gap'] = (d['Open'].shift(-1) - d['Close']) / d['Close'] * 100
        d['prev_gap'] = d['gap_pct'].shift(1)
        d['prev_intraday'] = d['intraday_return'].shift(1)
        d['prev_rvol'] = d['rvol'].shift(1)
        
        d['day_of_week'] = d.index.dayofweek
        d['symbol'] = sym
        
        # SPY context
        if spy_df is not None:
            spy_c = spy_df[['Close']].copy()
            spy_c.columns = ['spy_close']
            spy_c['spy_sma20'] = spy_c['spy_close'].rolling(20).mean()
            spy_c['spy_bullish'] = spy_c['spy_close'] > spy_c['spy_sma20']
            spy_c['spy_daily_ret'] = spy_c['spy_close'].pct_change() * 100
            d = d.join(spy_c[['spy_bullish', 'spy_daily_ret']], how='left')
        
        frames.append(d.dropna(subset=['gap_pct', 'rvol', 'atr14']))
    
    return pd.concat(frames, ignore_index=False)


def stress_test_rvol_edge(df):
    """Rigorous statistical validation of the RVOL edge."""
    print("\n" + "="*70)
    print("STRESS TEST 1: IS THE RVOL≥5x EDGE REAL?")
    print("="*70)
    
    results = {}
    
    # Define strategy
    strat = df[(df['gap_pct'] >= 3) & (df['gap_pct'] < 20) & 
               (df['rvol'] >= 5) & (df['Open'] >= 2) & (df['Open'] <= 20)]
    baseline = df[(df['gap_pct'] >= 3) & (df['gap_pct'] < 20) & 
                  (df['Open'] >= 2) & (df['Open'] <= 20)]
    
    print(f"\n  Strategy (gap 3-20%, RVOL≥5x, $2-20): n={len(strat)}")
    print(f"  Baseline (gap 3-20%, $2-20, ANY vol): n={len(baseline)}")
    
    if len(strat) < 10:
        print("  NOT ENOUGH DATA")
        return results
    
    strat_ret = strat['intraday_return']
    base_ret = baseline['intraday_return']
    
    print(f"\n  Strategy performance:")
    print(f"    Win rate: {(strat_ret > 0).mean()*100:.1f}%")
    print(f"    Avg return: {strat_ret.mean():.2f}%")
    print(f"    Median return: {strat_ret.median():.2f}%")
    print(f"    Std dev: {strat_ret.std():.2f}%")
    print(f"    Best trade: +{strat_ret.max():.1f}%")
    print(f"    Worst trade: {strat_ret.min():.1f}%")
    print(f"    Sharpe (annualized, ~252 trading days): {strat_ret.mean()/strat_ret.std()*np.sqrt(252):.2f}")
    
    print(f"\n  Baseline performance:")
    print(f"    Win rate: {(base_ret > 0).mean()*100:.1f}%")
    print(f"    Avg return: {base_ret.mean():.2f}%")
    print(f"    Median return: {base_ret.median():.2f}%")
    
    # T-test: strategy vs baseline
    t_stat, p_val = stats.ttest_ind(strat_ret, base_ret)
    print(f"\n  T-test (strategy vs baseline): t={t_stat:.3f}, p={p_val:.4f}")
    
    # One-sample t-test: is strategy mean > 0?
    t_stat2, p_val2 = stats.ttest_1samp(strat_ret, 0)
    print(f"  T-test (strategy vs zero): t={t_stat2:.3f}, p={p_val2/2:.4f} (one-tailed)")
    
    # Bootstrap confidence interval
    n_boot = 10000
    boot_means = []
    boot_wrs = []
    for _ in range(n_boot):
        sample = np.random.choice(strat_ret.values, size=len(strat_ret), replace=True)
        boot_means.append(sample.mean())
        boot_wrs.append((sample > 0).mean())
    
    boot_means = np.array(boot_means)
    boot_wrs = np.array(boot_wrs)
    
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    wr_ci_lower = np.percentile(boot_wrs, 2.5)
    wr_ci_upper = np.percentile(boot_wrs, 97.5)
    
    print(f"\n  Bootstrap 95% CI for avg return: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    print(f"  Bootstrap 95% CI for win rate: [{wr_ci_lower*100:.1f}%, {wr_ci_upper*100:.1f}%]")
    
    # Monte Carlo: random trading
    mc_means = []
    all_returns = df['intraday_return'].dropna()
    for _ in range(n_boot):
        sample = np.random.choice(all_returns.values, size=len(strat_ret), replace=True)
        mc_means.append(sample.mean())
    mc_means = np.array(mc_means)
    mc_percentile = (mc_means < strat_ret.mean()).mean() * 100
    
    print(f"\n  Monte Carlo: Strategy is better than {mc_percentile:.1f}% of random selections")
    print(f"  P-value (vs random): {1-mc_percentile/100:.4f}")
    
    # Year-by-year consistency
    print(f"\n  YEAR-BY-YEAR BREAKDOWN:")
    strat_copy = strat.copy()
    strat_copy['year'] = strat_copy.index.year
    for year in sorted(strat_copy['year'].unique()):
        yr_data = strat_copy[strat_copy['year'] == year]
        if len(yr_data) >= 5:
            wr = (yr_data['intraday_return'] > 0).mean() * 100
            ar = yr_data['intraday_return'].mean()
            print(f"    {year}: n={len(yr_data):>3}, WR={wr:>5.1f}%, Avg Return={ar:>+6.2f}%")
    
    results['strategy_stats'] = {
        'n': len(strat),
        'win_rate': (strat_ret > 0).mean() * 100,
        'avg_return': strat_ret.mean(),
        'median_return': strat_ret.median(),
        'sharpe': strat_ret.mean()/strat_ret.std()*np.sqrt(252) if strat_ret.std() > 0 else 0,
        'p_value_vs_zero': p_val2/2,
        'p_value_vs_baseline': p_val,
        'bootstrap_ci': [ci_lower, ci_upper],
        'monte_carlo_percentile': mc_percentile,
    }
    
    return results


def simulate_strategy(df):
    """Full trading simulation with realistic assumptions."""
    print("\n" + "="*70)
    print("STRATEGY SIMULATION (REALISTIC)")
    print("  $30,000 starting capital, 1% risk per trade, slippage modeled")
    print("="*70)
    
    results = {}
    
    # Strategy: Long on gap ups with RVOL
    strat_trades = df[
        (df['gap_pct'] >= 3) & (df['gap_pct'] < 20) &
        (df['rvol'] >= 3) & 
        (df['Open'] >= 2) & (df['Open'] <= 20)
    ].copy()
    
    strat_trades = strat_trades.sort_index()
    
    if len(strat_trades) < 10:
        print("  Not enough trades")
        return results
    
    # Simulation parameters
    starting_capital = 30000
    risk_per_trade = 0.01  # 1% of account per trade
    slippage_pct = 0.3     # 0.3% slippage per trade (entry + exit)
    max_trades_per_day = 3
    
    capital = starting_capital
    trade_log = []
    daily_capital = []
    
    # Group by date and take max 3 per day
    dates = strat_trades.index.unique()
    
    for date in dates:
        day_trades = strat_trades.loc[strat_trades.index == date]
        
        # Limit to max_trades_per_day (take highest RVOL)
        if len(day_trades) > max_trades_per_day:
            day_trades = day_trades.nlargest(max_trades_per_day, 'rvol')
        
        for _, trade in day_trades.iterrows():
            # Position sizing: risk 1% of capital
            # Stop loss at low_from_open (use the actual low as worst case)
            stop_distance = abs(trade['low_from_open']) / 100
            if stop_distance < 0.01:
                stop_distance = 0.03  # minimum 3% stop
            
            risk_amount = capital * risk_per_trade
            position_size = risk_amount / stop_distance
            position_size = min(position_size, capital * 0.15)  # max 15% of capital per trade
            
            # Actual return after slippage
            raw_return = trade['intraday_return'] / 100
            actual_return = raw_return - (slippage_pct / 100)
            
            pnl = position_size * actual_return
            capital += pnl
            
            trade_log.append({
                'date': str(date),
                'symbol': trade['symbol'],
                'gap_pct': trade['gap_pct'],
                'rvol': trade['rvol'],
                'raw_return': raw_return * 100,
                'actual_return': actual_return * 100,
                'position_size': position_size,
                'pnl': pnl,
                'capital': capital,
            })
        
        daily_capital.append({'date': str(date), 'capital': capital})
    
    trades_df = pd.DataFrame(trade_log)
    
    if len(trades_df) == 0:
        print("  No trades executed")
        return results
    
    # Results
    total_return = (capital - starting_capital) / starting_capital * 100
    n_trades = len(trades_df)
    win_rate = (trades_df['pnl'] > 0).mean() * 100
    avg_pnl = trades_df['pnl'].mean()
    
    # Drawdown calculation
    capital_series = trades_df['capital']
    running_max = capital_series.expanding().max()
    drawdown = (capital_series - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # Monthly breakdown
    trades_df['month'] = pd.to_datetime(trades_df['date'], utc=True).dt.to_period('M')
    monthly = trades_df.groupby('month').agg(
        trades=('pnl', 'count'),
        total_pnl=('pnl', 'sum'),
        win_rate=('pnl', lambda x: (x > 0).mean() * 100),
    )
    
    winning_months = (monthly['total_pnl'] > 0).sum()
    total_months = len(monthly)
    
    print(f"\n  SIMULATION RESULTS")
    print(f"  {'='*50}")
    print(f"  Starting Capital:  ${starting_capital:>12,.2f}")
    print(f"  Ending Capital:    ${capital:>12,.2f}")
    print(f"  Total Return:      {total_return:>12.1f}%")
    print(f"  Total Trades:      {n_trades:>12}")
    print(f"  Win Rate:          {win_rate:>12.1f}%")
    print(f"  Avg P&L per trade: ${avg_pnl:>12.2f}")
    print(f"  Max Drawdown:      {max_drawdown:>12.1f}%")
    print(f"  Winning Months:    {winning_months}/{total_months}")
    print(f"  Avg Trades/Month:  {n_trades/max(total_months,1):>12.1f}")
    
    # Profit factor
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    print(f"  Profit Factor:     {profit_factor:>12.2f}")
    print(f"  Gross Profit:      ${gross_profit:>12.2f}")
    print(f"  Gross Loss:        ${gross_loss:>12.2f}")
    
    # Best/Worst
    print(f"\n  Best Trade:  ${trades_df['pnl'].max():>10.2f} ({trades_df.loc[trades_df['pnl'].idxmax(), 'symbol']})")
    print(f"  Worst Trade: ${trades_df['pnl'].min():>10.2f} ({trades_df.loc[trades_df['pnl'].idxmin(), 'symbol']})")
    
    # Monthly details
    print(f"\n  MONTHLY BREAKDOWN:")
    print(f"  {'Month':<12} {'Trades':>6} {'P&L':>12} {'Win Rate':>10}")
    print(f"  {'-'*44}")
    for month, row in monthly.iterrows():
        print(f"  {str(month):<12} {row['trades']:>6} ${row['total_pnl']:>11,.2f} {row['win_rate']:>9.1f}%")
    
    results['simulation'] = {
        'starting_capital': starting_capital,
        'ending_capital': capital,
        'total_return_pct': total_return,
        'total_trades': n_trades,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'avg_pnl_per_trade': avg_pnl,
        'winning_months': winning_months,
        'total_months': total_months,
    }
    
    return results


def test_short_strategy(df):
    """Test shorting overextended gap ups (the fade play)."""
    print("\n" + "="*70)
    print("SHORT STRATEGY TEST: FADING OVEREXTENDED GAP UPS")
    print("  Short when gap ≥ 15%, bet on the fade back to VWAP/open")
    print("="*70)
    
    results = {}
    
    # Various short entry criteria
    scenarios = [
        ("Gap 15-30%, any vol", 15, 30, 0),
        ("Gap 15-30%, RVOL≥2x", 15, 30, 2),
        ("Gap 15-30%, RVOL≥3x", 15, 30, 3),
        ("Gap 20%+, any vol", 20, 200, 0),
        ("Gap 20%+, RVOL≥2x", 20, 200, 2),
        ("Gap 10-15%, RVOL<2 (weak vol)", 10, 15, -2),  # negative means LESS than
        ("Gap 10-20%, already overbought RSI>70", 10, 20, 0),
    ]
    
    for label, gap_min, gap_max, rvol_req in scenarios:
        if "RSI" in label:
            mask = (
                (df['gap_pct'] >= gap_min) & (df['gap_pct'] < gap_max) &
                (df['Open'] >= 2) & (df['prev_rsi'] > 70)
            )
        elif rvol_req < 0:
            mask = (
                (df['gap_pct'] >= gap_min) & (df['gap_pct'] < gap_max) &
                (df['Open'] >= 2) & (df['rvol'] < abs(rvol_req))
            )
        elif rvol_req > 0:
            mask = (
                (df['gap_pct'] >= gap_min) & (df['gap_pct'] < gap_max) &
                (df['Open'] >= 2) & (df['rvol'] >= rvol_req)
            )
        else:
            mask = (
                (df['gap_pct'] >= gap_min) & (df['gap_pct'] < gap_max) &
                (df['Open'] >= 2)
            )
        
        subset = df[mask]
        if len(subset) < 10:
            continue
        
        # Short return = negative of intraday return
        short_ret = -subset['intraday_return']
        win_rate = (short_ret > 0).mean() * 100
        avg_ret = short_ret.mean()
        
        # Risk: what's the worst that can happen? (stock runs against us)
        worst_case = short_ret.min()  # biggest loss
        max_adverse = subset['high_from_open'].max()
        
        print(f"\n  {label}: n={len(subset)}")
        print(f"    Short win rate: {win_rate:.1f}%")
        print(f"    Avg short return: {avg_ret:+.2f}%")
        print(f"    Median short return: {short_ret.median():+.2f}%")
        print(f"    Max adverse excursion: +{max_adverse:.1f}% (worst squeeze against us)")
        print(f"    Worst single trade: {worst_case:.1f}%")
        
        results[label] = {
            'count': len(subset),
            'win_rate': win_rate,
            'avg_return': avg_ret,
            'max_adverse': max_adverse,
        }
    
    return results


def test_gap_down_bounce(df):
    """Test the gap-down bounce (mean reversion long play)."""
    print("\n" + "="*70)
    print("GAP-DOWN BOUNCE STRATEGY (Mean Reversion Long)")
    print("  Buy stocks that gap DOWN sharply → bet on bounce")
    print("="*70)
    
    results = {}
    
    scenarios = [
        ("Gap down 5-10%", -10, -5),
        ("Gap down 10-15%", -15, -10),
        ("Gap down 15-25%", -25, -15),
        ("Gap down 25%+", -100, -25),
    ]
    
    for label, gap_min, gap_max in scenarios:
        mask = (df['gap_pct'] >= gap_min) & (df['gap_pct'] < gap_max) & (df['Open'] >= 2)
        subset = df[mask]
        if len(subset) < 10:
            continue
        
        # Long at open, sell at close
        ret = subset['intraday_return']
        win_rate = (ret > 0).mean() * 100
        avg_ret = ret.mean()
        
        print(f"\n  {label}: n={len(subset)}")
        print(f"    Win rate (bounce): {win_rate:.1f}%")
        print(f"    Avg intraday return: {avg_ret:+.2f}%")
        print(f"    Avg high from open: +{subset['high_from_open'].mean():.2f}%")
        print(f"    Avg low from open: {subset['low_from_open'].mean():.2f}%")
        
        # With RVOL filter
        high_vol = subset[subset['rvol'] >= 3]
        if len(high_vol) > 5:
            hv_ret = high_vol['intraday_return']
            print(f"    With RVOL≥3x: n={len(high_vol)}, WR={((hv_ret>0).mean()*100):.1f}%, Avg={hv_ret.mean():+.2f}%")
        
        results[label] = {
            'count': len(subset),
            'win_rate': win_rate,
            'avg_return': avg_ret,
        }
    
    return results


def optimal_strategy_search(df):
    """Fine-grained search for the absolute best filter combination."""
    print("\n" + "="*70)
    print("FINE-GRAINED OPTIMAL STRATEGY SEARCH")
    print("  Testing thousands of filter combinations for the BEST edge")
    print("="*70)
    
    results = []
    
    # Direction: Long
    for gap_min in [3, 4, 5, 6, 7, 8, 10]:
        for gap_max in [8, 10, 12, 15, 20, 25]:
            if gap_max <= gap_min:
                continue
            for rvol_min in [2, 3, 4, 5, 7, 10]:
                for price_min in [1, 2, 3, 5]:
                    for price_max in [10, 15, 20, 30]:
                        if price_max <= price_min:
                            continue
                        for spy_filter in [None, True, False]:
                            mask = (
                                (df['gap_pct'] >= gap_min) & (df['gap_pct'] < gap_max) &
                                (df['rvol'] >= rvol_min) &
                                (df['Open'] >= price_min) & (df['Open'] <= price_max)
                            )
                            if spy_filter is not None:
                                mask = mask & (df['spy_bullish'] == spy_filter)
                            
                            subset = df[mask]
                            n = len(subset)
                            if n < 25:  # statistical minimum
                                continue
                            
                            ret = subset['intraday_return']
                            win_rate = (ret > 0).mean()
                            avg_ret = ret.mean()
                            med_ret = ret.median()
                            
                            # Expected value with slippage
                            slippage = 0.3
                            adj_ret = avg_ret - slippage
                            
                            if ret.std() > 0:
                                sharpe = adj_ret / ret.std()
                            else:
                                sharpe = 0
                            
                            # Edge score: balance profitability with sample size and consistency
                            # Higher sample = more reliable, higher sharpe = more consistent
                            edge_score = adj_ret * np.sqrt(n) * (1 + sharpe)
                            
                            spy_label = "any" if spy_filter is None else ("bull" if spy_filter else "bear")
                            
                            results.append({
                                'direction': 'LONG',
                                'gap': f'{gap_min}-{gap_max}%',
                                'rvol': f'≥{rvol_min}x',
                                'price': f'${price_min}-${price_max}',
                                'spy': spy_label,
                                'n': n,
                                'win_rate': win_rate * 100,
                                'avg_return': avg_ret,
                                'adj_return': adj_ret,
                                'median_return': med_ret,
                                'sharpe': sharpe,
                                'edge_score': edge_score,
                            })
    
    # Direction: Short (fade the gap)
    for gap_min in [8, 10, 12, 15, 20]:
        for gap_max in [15, 20, 25, 30, 50]:
            if gap_max <= gap_min:
                continue
            for rvol_req in [0, 1, 2, 3]:  # 0 = any volume
                for price_min in [2, 5]:
                    for price_max in [20, 50]:
                        if price_max <= price_min:
                            continue
                        
                        mask = (
                            (df['gap_pct'] >= gap_min) & (df['gap_pct'] < gap_max) &
                            (df['Open'] >= price_min) & (df['Open'] <= price_max)
                        )
                        if rvol_req > 0:
                            # For shorts: LOW volume gaps are more likely to fade
                            mask = mask & (df['rvol'] < rvol_req)
                        
                        subset = df[mask]
                        if len(subset) < 20:
                            continue
                        
                        short_ret = -subset['intraday_return']
                        win_rate = (short_ret > 0).mean()
                        avg_ret = short_ret.mean()
                        adj_ret = avg_ret - 0.5  # higher slippage for shorts
                        
                        if short_ret.std() > 0:
                            sharpe = adj_ret / short_ret.std()
                        else:
                            sharpe = 0
                        
                        edge_score = adj_ret * np.sqrt(len(subset)) * (1 + sharpe)
                        
                        results.append({
                            'direction': 'SHORT',
                            'gap': f'{gap_min}-{gap_max}%',
                            'rvol': f'<{rvol_req}x' if rvol_req > 0 else 'any',
                            'price': f'${price_min}-${price_max}',
                            'spy': 'any',
                            'n': len(subset),
                            'win_rate': win_rate * 100,
                            'avg_return': avg_ret,
                            'adj_return': adj_ret,
                            'median_return': short_ret.median(),
                            'sharpe': sharpe,
                            'edge_score': edge_score,
                        })
    
    if not results:
        return {}
    
    results_df = pd.DataFrame(results)
    
    # Top LONG strategies
    long_strats = results_df[results_df['direction'] == 'LONG'].sort_values('edge_score', ascending=False)
    print(f"\n  TOP 15 LONG STRATEGIES:")
    print(f"  {'='*130}")
    for _, r in long_strats.head(15).iterrows():
        print(f"  Gap {r['gap']:>8} | RVOL {r['rvol']:>5} | Price {r['price']:>10} | SPY={r['spy']:<4} | "
              f"n={r['n']:>3} | WR={r['win_rate']:>5.1f}% | Adj Ret={r['adj_return']:>+6.2f}% | Sharpe={r['sharpe']:.3f}")
    
    # Top SHORT strategies
    short_strats = results_df[results_df['direction'] == 'SHORT'].sort_values('edge_score', ascending=False)
    if len(short_strats) > 0:
        print(f"\n  TOP 10 SHORT STRATEGIES:")
        print(f"  {'='*130}")
        for _, r in short_strats.head(10).iterrows():
            print(f"  Gap {r['gap']:>8} | RVOL {r['rvol']:>5} | Price {r['price']:>10} | "
                  f"n={r['n']:>3} | WR={r['win_rate']:>5.1f}% | Adj Ret={r['adj_return']:>+6.2f}% | Sharpe={r['sharpe']:.3f}")
    
    return {
        'top_long': long_strats.head(15).to_dict('records'),
        'top_short': short_strats.head(10).to_dict('records') if len(short_strats) > 0 else [],
    }


def run_experiment_3():
    print("="*70)
    print("EXPERIMENT 3: DEEP DIVE & STRATEGY SIMULATION")
    print("="*70)
    
    data = download(UNIVERSE, period="5y")
    df = build_dataset(data)
    
    print(f"\nDataset: {len(df)} trading days, {df['symbol'].nunique()} symbols")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    all_results = {}
    all_results['stress_test'] = stress_test_rvol_edge(df)
    all_results['simulation'] = simulate_strategy(df)
    all_results['short_strategies'] = test_short_strategy(df)
    all_results['gap_down_bounce'] = test_gap_down_bounce(df)
    all_results['optimal_search'] = optimal_strategy_search(df)
    
    results_file = os.path.join(RESULTS_DIR, 'experiment_3_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")
    
    return all_results


if __name__ == "__main__":
    run_experiment_3()
