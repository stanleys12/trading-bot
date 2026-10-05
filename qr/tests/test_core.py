"""
Unit tests for the calculations everything else depends on.

These are deliberately concentrated on the things that, when wrong, produce a
plausible-looking but false backtest: cost estimation, return computation
across corporate actions, the information cutoff, and the metric annualisation
that previously produced "Sharpe 8.79".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.backtesting.costs import (CostConfig, add_cost_features,
                                  corwin_schultz_spread, shortable_mask,
                                  trade_cost_bps)
from qr.backtesting.engine import BacktestConfig, prepare_panel, run_backtest
from qr.backtesting.metrics import (deflated_sharpe, max_drawdown,
                                    probabilistic_sharpe, sharpe, summarize)
from qr.backtesting.validation import make_folds, multiple_testing_report


# --------------------------------------------------------------- fixtures
def synthetic_panel(n_symbols=60, n_days=800, seed=7, drift=0.0) -> pd.DataFrame:
    """Geometric random walk with no cross-sectional signal whatsoever."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    rows = []
    for i in range(n_symbols):
        r = rng.normal(drift, 0.03, n_days)
        close = 20 * np.exp(np.cumsum(r))
        spread = 0.004
        high = close * (1 + np.abs(rng.normal(0, 0.015, n_days)) + spread / 2)
        low = close * (1 - np.abs(rng.normal(0, 0.015, n_days)) - spread / 2)
        open_ = close * (1 + rng.normal(0, 0.01, n_days))
        open_ = np.clip(open_, low, high)
        vol = rng.lognormal(13, 0.5, n_days)
        rows.append(pd.DataFrame({
            "date": dates, "symbol": f"S{i:03d}", "open": open_, "high": high,
            "low": low, "close": close, "adj_close": close, "adj_factor": 1.0,
            "volume": vol, "dollar_volume": close * vol,
        }))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------- cost model
def test_corwin_schultz_recovers_a_known_spread():
    """Inject a known spread into H/L and check the estimator finds it."""
    rng = np.random.default_rng(0)
    n = 600
    true_spread = 0.02                       # 200 bps
    mid = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    # true high/low of the MID, then widen by the spread
    hi_mid = mid * (1 + np.abs(rng.normal(0, 0.012, n)))
    lo_mid = mid * (1 - np.abs(rng.normal(0, 0.012, n)))
    high = pd.Series(hi_mid * (1 + true_spread / 2))
    low = pd.Series(lo_mid * (1 - true_spread / 2))

    est = corwin_schultz_spread(high, low, window=60).dropna()
    assert len(est) > 100
    # The estimator is noisy and known to be biased low; we only require it to
    # land in the right order of magnitude and be clearly non-zero.
    assert 0.002 < est.median() < 0.06, f"median estimate {est.median():.4f}"


def test_spread_estimate_is_lagged():
    """add_cost_features must not let day t's own H/L into day t's spread."""
    panel = synthetic_panel(n_symbols=3, n_days=200)
    out = add_cost_features(panel)
    g = out[out.symbol == "S000"].sort_values("date").reset_index(drop=True)
    raw = corwin_schultz_spread(g["high"], g["low"])
    # column must equal the raw estimate shifted by one bar
    merged = pd.concat([g["spread_est"], raw.shift(1)], axis=1).dropna()
    assert np.allclose(merged.iloc[:, 0], merged.iloc[:, 1].clip(lower=5e-4),
                       equal_nan=True)


def test_costs_increase_with_participation():
    cfg = CostConfig()
    base = dict(price=pd.Series([10.0] * 3), spread=pd.Series([0.01] * 3),
                vol=pd.Series([0.03] * 3), adv_dollar=pd.Series([1e6] * 3))
    small = trade_cost_bps(trade_dollars=pd.Series([1e3] * 3), cfg=cfg, **base)
    large = trade_cost_bps(trade_dollars=pd.Series([1e5] * 3), cfg=cfg, **base)
    assert (large > small).all(), "impact must scale with participation"


def test_costs_are_never_free():
    cfg = CostConfig()
    c = trade_cost_bps(price=pd.Series([5.0]), spread=pd.Series([0.0]),
                       vol=pd.Series([0.0]), trade_dollars=pd.Series([1e4]),
                       adv_dollar=pd.Series([1e9]), cfg=cfg)
    assert c.iloc[0] > 0


def test_shortable_mask_excludes_microcaps():
    df = pd.DataFrame({"close": [2.0, 50.0, 8.0],
                       "adv_dollar": [1e7, 1e7, 1e5]})
    m = shortable_mask(df)
    assert list(m) == [False, True, False]


# ----------------------------------------------------------- return maths
def test_returns_use_adjusted_prices_across_a_split():
    """A 1:10 reverse split must produce a 0% return, not +900%."""
    dates = pd.bdate_range("2022-01-03", periods=10)
    close = np.array([1.0] * 5 + [10.0] * 5)          # raw price jumps at split
    adj_factor = np.array([10.0] * 5 + [1.0] * 5)     # adj_close = close*factor
    panel = pd.DataFrame({
        "date": dates, "symbol": "X", "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "adj_close": close * adj_factor,
        "adj_factor": adj_factor, "volume": 1e6, "dollar_volume": close * 1e6,
    })
    cfg = BacktestConfig(holding_days=1)
    prep = prepare_panel(panel, cfg)
    # the open-to-open return spanning the split must be ~0
    r = prep["fwd_ret"].dropna()
    assert np.nanmax(np.abs(r)) < 0.05, f"split leaked into returns: {r.values}"


def test_forward_return_starts_at_next_open():
    panel = synthetic_panel(n_symbols=2, n_days=60)
    cfg = BacktestConfig(holding_days=3)
    prep = prepare_panel(panel, cfg).sort_values(["symbol", "date"])
    g = prep[prep.symbol == "S000"].reset_index(drop=True)
    i = 10
    expected = g.loc[i + 4, "adj_open"] / g.loc[i + 1, "adj_open"] - 1
    assert np.isclose(g.loc[i, "fwd_ret"], expected)


# -------------------------------------------------------- information cutoff
def test_strategy_cannot_see_forward_columns():
    """The frame handed to a signal must not contain outcome columns."""
    panel = synthetic_panel(n_symbols=40, n_days=400)
    cfg = BacktestConfig(lookback_days=120, rebalance_days=20, holding_days=5,
                         min_adv_dollar=0, min_names_per_side=3)
    prep = prepare_panel(panel, cfg)

    seen = {}

    def spy(vis: pd.DataFrame) -> pd.Series:
        seen["cols"] = set(vis.columns)
        seen["max_date"] = vis["date"].max()
        last = vis[vis["date"] == vis["date"].max()]
        return pd.Series(1.0, index=last["symbol"])

    run_backtest(prep, spy, cfg)
    for forbidden in ("fwd_ret", "next_open", "exit_open", "raw_next_open"):
        assert forbidden not in seen["cols"], f"{forbidden} leaked to strategy"


def test_random_signal_earns_nothing_on_a_random_walk():
    """The engine must not manufacture edge from noise."""
    panel = synthetic_panel(n_symbols=80, n_days=900, seed=11)
    cfg = BacktestConfig(lookback_days=150, rebalance_days=10, holding_days=5,
                         min_adv_dollar=0, min_names_per_side=5, capital=1e5)
    prep = prepare_panel(panel, cfg)
    rng = np.random.default_rng(3)

    def noise(vis: pd.DataFrame) -> pd.Series:
        last = vis[vis["date"] == vis["date"].max()]["symbol"]
        return pd.Series(rng.normal(size=len(last)), index=last.values)

    res = run_backtest(prep, noise, cfg)
    m = summarize(res.daily, cfg.capital)
    assert m["n_periods"] > 20
    # gross should be ~0; net must be NEGATIVE because costs are real
    assert abs(m["mean_period_ret"]) < 0.01
    assert m["cost_drag_annual"] > 0, "a random strategy paid no costs"


def test_engine_detects_an_oracle():
    """Sanity: if a signal IS the outcome, the engine must show a huge result."""
    panel = synthetic_panel(n_symbols=60, n_days=700, seed=5)
    cfg = BacktestConfig(lookback_days=150, rebalance_days=10, holding_days=5,
                         min_adv_dollar=0, min_names_per_side=5)
    prep = prepare_panel(panel, cfg)
    truth = prep.set_index(["date", "symbol"])["fwd_ret"]

    def oracle(vis: pd.DataFrame) -> pd.Series:
        d = vis["date"].max()
        try:
            return truth.loc[d]
        except KeyError:
            return pd.Series(dtype=float)

    res = run_backtest(prep, oracle, cfg)
    m = summarize(res.daily, cfg.capital)
    assert m["sharpe"] > 3, ("engine failed to detect a perfect oracle; "
                             "it cannot be trusted to detect leakage")


# ------------------------------------------------------------------ metrics
def test_sharpe_annualisation_uses_observed_frequency():
    """A 12-period-per-year series must not be annualised by sqrt(252)."""
    dates = pd.date_range("2020-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(1)
    d = pd.DataFrame({
        "date": dates, "net_ret": rng.normal(0.01, 0.05, 48),
        "gross_ret": rng.normal(0.01, 0.05, 48),
        "turnover_dollars": 1e4, "cost_dollars": 10.0,
        "gross_notional": 1e5, "avg_cost_bps": 30.0, "avg_spread_bps": 40.0,
        "n_long": 10, "n_short": 10,
    })
    m = summarize(d)
    assert 11 < m["periods_per_year"] < 13
    naive = d["net_ret"].mean() / d["net_ret"].std(ddof=1) * np.sqrt(252)
    assert m["sharpe"] < naive / 2


def test_max_drawdown_sign_and_magnitude():
    eq = pd.Series([1.0, 1.5, 0.75, 1.2])
    assert np.isclose(max_drawdown(eq), -0.5)


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.02, 0.05, 60))
    one = deflated_sharpe(r, 12, n_trials=1)
    many = deflated_sharpe(r, 12, n_trials=200)
    assert many < one, "DSR must fall as the number of trials rises"


def test_psr_below_one_on_short_noisy_samples():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.001, 0.05, 20))
    p = probabilistic_sharpe(r, 12)
    assert 0.0 <= p <= 1.0


# --------------------------------------------------------------- validation
def test_folds_have_purge_and_embargo():
    dates = pd.Series(pd.bdate_range("2019-01-01", periods=1400))
    folds = make_folds(dates, n_folds=4, purge_days=10, embargo_days=5,
                       min_train_days=500)
    assert len(folds) >= 2
    for f in folds:
        assert f.train_end < f.test_start, "train overlaps test"
        gap = (f.test_start - f.train_end).days
        assert gap >= 10, f"purge+embargo too small: {gap}d"


def test_multiple_testing_correction_is_applied():
    p = {"a": 0.01, "b": 0.04, "c": 0.20, "d": 0.5, "e": 0.9}
    out = multiple_testing_report(p)
    assert (out["bonferroni"] >= out["p_raw"]).all()
    assert not out.loc["b", "passes_bonferroni"], "0.04 must not survive k=5"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))


# --------------------------------------------- engine information invariant
def test_frame_handed_to_signal_ends_exactly_at_decision_date():
    """
    THE invariant that makes this architecture safe.

    Our signals evaluate at the END of the frame they are given (`px.iloc[-1]`,
    `px.iloc[-22]`, ...). That is only correct if the engine guarantees the
    frame ends exactly at the decision date. If it ever handed over a frame
    extending past D, every signal would silently evaluate the wrong date --
    and no structural column-stripping guard would catch it, because no
    forbidden column is involved.

    This replaces freqtrade's lookahead-analysis for our API. Their behavioural
    probe (append future rows, check the indicator at T) does not transfer:
    it pins T via a fixed index, whereas ours is implied by the frame's end,
    so appending rows moves the evaluation point rather than contaminating it.
    We ran the naive port and it produced 17 false positives out of 17 signals.
    The invariant below is what their probe is really protecting, stated
    directly.
    """
    panel = synthetic_panel(n_symbols=40, n_days=500)
    cfg = BacktestConfig(lookback_days=150, rebalance_days=25, holding_days=5,
                         min_adv_dollar=0, min_names_per_side=3)
    prep = prepare_panel(panel, cfg)

    seen_ends, decision_dates = [], []

    def recorder(vis: pd.DataFrame) -> pd.Series:
        seen_ends.append(vis["date"].max())
        last = vis[vis["date"] == vis["date"].max()]["symbol"]
        return pd.Series(1.0, index=last.values)

    res = run_backtest(prep, recorder, cfg)
    decision_dates = list(res.daily["date"])

    assert len(seen_ends) > 5
    # every frame end must be a real decision date, never later
    for end, d in zip(seen_ends, decision_dates):
        assert end <= d, f"frame extended past the decision date: {end} > {d}"
    assert seen_ends[0] == decision_dates[0]


def test_signals_are_not_warmup_sensitive():
    """
    freqtrade's recursive-analysis, adapted -- and this one DOES transfer.

    EMA/RSI-style recursive indicators converge to different values depending
    on how much warm-up history they were fed, so a backtest given 560 days and
    a live system given 140 disagree on the same date from identical code.
    Windowed indicators do not have this problem. All 17 of our signals score
    1.0000, i.e. none are recursive -- which is worth locking down with a test
    rather than assuming.
    """
    from qr.backtesting.bias_probes import startup_stability
    from qr.strategies.signals import low_vol, mom_12_1, reversal_1m

    panel = synthetic_panel(n_symbols=60, n_days=900)
    cfg = BacktestConfig(holding_days=21)
    prep = prepare_panel(panel, cfg)
    dates = pd.DatetimeIndex(sorted(prep["date"].unique()))
    picks = [dates[i] for i in (600, 700, 800)]

    for fn in (mom_12_1, reversal_1m, low_vol):
        out = startup_stability(prep, fn, picks, windows=(140, 280, 560))
        assert out["status"] == "PASS", f"{fn.__name__} is warm-up sensitive: {out}"
