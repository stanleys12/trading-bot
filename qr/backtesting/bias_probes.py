"""
Behavioural bias probes.
========================

Adapted from the two checks Freqtrade ships as `lookahead-analysis` and
`recursive-analysis`. Credit where due: those are the best-designed bias tools
in the open-source retail trading ecosystem, and they test something our
structural guard cannot.

Why we need both kinds of check
-------------------------------
`engine.run_backtest` prevents leakage STRUCTURALLY: it drops forward-looking
columns before handing a frame to a signal, so the signal cannot read a column
it should not have. That is airtight for the failure mode it targets.

It is blind to a second failure mode: a signal that only ever touches
legitimate columns but uses them in a way that embeds future information.
Examples that would sail past the structural guard:

    (px - px.mean()) / px.std()        # full-sample normalisation
    px.rolling(20, center=True).mean() # centred window peeks forward
    px.rank(pct=True)                  # over the whole visible history

None of those read a forbidden column. All of them use information from after
the point they are evaluated at.

CROSS-SECTIONAL CONTAMINATION (freqtrade's lookahead-analysis, adapted)
-----------------------------------------------------------------------
The naive port of freqtrade's test does not work against our signal API, and
we found that out by running it: it flagged all 17 signals, and all 17 were
false positives. See the docstring of `cross_sectional_contamination` for why,
and for the version that is actually meaningful here.

STARTUP STABILITY (freqtrade's recursive-analysis, adapted)
------------------------------------------------------------
Recursive indicators -- EMA, RSI, anything with a decay -- converge to
different values depending on how much warm-up history they were given. A
backtest fed 560 days and a live system fed 200 will disagree on the same
date, from identical code. We evaluate each signal with several history
windows and measure how much the ranking moves.

Neither probe proves a signal is correct. Both can prove it is broken.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

FORBIDDEN = ("fwd_ret", "next_open", "exit_open", "raw_next_open")


def _visible(panel: pd.DataFrame, upto: pd.Timestamp,
             lookback_days: int) -> pd.DataFrame:
    lo = upto - pd.Timedelta(days=lookback_days * 2)
    v = panel[(panel["date"] <= upto) & (panel["date"] >= lo)]
    return v.drop(columns=[c for c in FORBIDDEN if c in v.columns])


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank agreement on the symbols both series score."""
    idx = a.dropna().index.intersection(b.dropna().index)
    if len(idx) < 10:
        return np.nan
    x, y = a.loc[idx], b.loc[idx]
    if x.nunique() < 3 or y.nunique() < 3:
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def cross_sectional_contamination(panel: pd.DataFrame, signal_fn,
                                  test_dates: list[pd.Timestamp],
                                  lookback_days: int = 280,
                                  future_days: int = 60,
                                  seed: int = 0) -> dict:
    """
    Does one symbol's FUTURE data change ANOTHER symbol's signal today?

    A NOTE ON THE NAIVE VERSION OF THIS TEST, WHICH IS WRONG HERE.
    Freqtrade's lookahead-analysis appends future rows and checks whether an
    indicator's value at timestamp T moved. That works because a freqtrade
    strategy populates indicator COLUMNS across a fixed timestamp index, so T
    is pinned.

    Our signal API is different: a signal receives a truncated frame and
    returns one value per symbol, evaluated at the END of that frame
    (`px.iloc[-1]`, `px.iloc[-22]`, ...). Appending future rows therefore moves
    the evaluation date rather than contaminating it. Running the naive probe
    flagged all 17 signals as leaking; every one of those was the probe
    measuring its own mutation. Recording that here because it is exactly the
    kind of false positive this module exists to prevent.

    The version that IS meaningful for a cross-sectional API: extend the frame
    into the future for HALF the symbols only, and check the signal values of
    the OTHER half -- whose own data is untouched and still ends at D. If those
    move, the signal is routing information across symbols from the future,
    which is what a cross-sectional normaliser (`- r.median()`, `.rank(pct=True)`,
    a beta against a universe mean) can silently do.
    """
    rng = np.random.default_rng(seed)
    corrs, n_checked = [], 0

    for d in test_dates:
        clean = _visible(panel, d, lookback_days)
        if clean.empty:
            continue
        syms = np.sort(clean["symbol"].unique())
        if len(syms) < 40:
            continue
        contaminated_syms = set(rng.choice(syms, size=len(syms) // 2,
                                           replace=False))
        held_out = [s for s in syms if s not in contaminated_syms]

        lo = d - pd.Timedelta(days=lookback_days * 2)
        hi = d + pd.Timedelta(days=future_days)
        window = panel[(panel["date"] >= lo) & (panel["date"] <= hi)]
        # half the symbols get future rows; the held-out half stays cut at D
        ragged = window[
            (window["symbol"].isin(contaminated_syms))
            | (window["date"] <= d)
        ].drop(columns=[c for c in FORBIDDEN if c in window.columns])

        base = signal_fn(clean)
        test = signal_fn(ragged)
        if base is None or test is None or len(base) == 0 or len(test) == 0:
            continue

        # compare ONLY the symbols whose own data was never extended
        idx = [s for s in held_out if s in base.index and s in test.index]
        if len(idx) < 10:
            continue
        c = _spearman(base.loc[idx], test.loc[idx])
        if np.isfinite(c):
            corrs.append(c)
            n_checked += 1

    if not corrs:
        return {"status": "no_data", "n_checked": 0}
    corrs = np.array(corrs)
    worst = float(corrs.min())
    return {
        "status": "PASS" if worst > 0.999 else "CROSS_CONTAMINATED",
        "n_checked": n_checked,
        "mean_rank_corr": float(corrs.mean()),
        "worst_rank_corr": worst,
    }


def startup_stability(panel: pd.DataFrame, signal_fn,
                      test_dates: list[pd.Timestamp],
                      windows: tuple[int, ...] = (140, 280, 560)) -> dict:
    """
    Does the signal's value on D depend on how much warm-up history it got?

    Compares each shorter window against the longest. A windowed (non-recursive)
    indicator scores 1.0. A recursive one drifts, and the drift is the gap
    between what a backtest computes and what a live system would.
    """
    ref_w = max(windows)
    out = {}
    for w in windows:
        if w == ref_w:
            continue
        corrs = []
        for d in test_dates:
            a = signal_fn(_visible(panel, d, ref_w))
            b = signal_fn(_visible(panel, d, w))
            if a is None or b is None or len(a) == 0 or len(b) == 0:
                continue
            c = _spearman(a, b)
            if np.isfinite(c):
                corrs.append(c)
        out[f"corr_{w}d_vs_{ref_w}d"] = float(np.mean(corrs)) if corrs else np.nan

    vals = [v for v in out.values() if np.isfinite(v)]
    worst = min(vals) if vals else np.nan
    out["worst"] = float(worst) if vals else np.nan
    # A signal whose ranking moves by more than a hair on warm-up length will
    # not reproduce live, even though nothing about it is "leaky".
    out["status"] = ("PASS" if np.isfinite(worst) and worst > 0.99
                     else "UNSTABLE" if np.isfinite(worst) else "no_data")
    return out


def probe_all(panel: pd.DataFrame, registry: dict, n_dates: int = 6,
              seed: int = 0) -> pd.DataFrame:
    """Run both probes over every signal in a registry."""
    dates = np.sort(panel["date"].unique())
    # Sample decision dates from the middle of the panel so every signal has
    # both warm-up history behind it and future rows available ahead of it.
    lo, hi = int(len(dates) * 0.45), int(len(dates) * 0.85)
    rng = np.random.default_rng(seed)
    picks = [pd.Timestamp(dates[i])
             for i in rng.choice(np.arange(lo, hi), size=n_dates, replace=False)]

    rows = []
    for name, (category, fn) in registry.items():
        ti = cross_sectional_contamination(panel, fn, picks)
        ss = startup_stability(panel, fn, picks)
        rows.append({
            "signal": name, "category": category,
            "cross_contam": ti.get("status"),
            "worst_rank_corr": ti.get("worst_rank_corr"),
            "startup": ss.get("status"),
            "worst_startup_corr": ss.get("worst"),
        })
        logger.info("  %-20s cross_contam=%-18s (%.4f)  startup=%-9s (%.4f)",
                    name, ti.get("status"),
                    ti.get("worst_rank_corr", np.nan),
                    ss.get("status"), ss.get("worst", np.nan))
    return pd.DataFrame(rows)
