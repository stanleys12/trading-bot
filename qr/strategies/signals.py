"""
Signal library.
===============

Every signal is a pure function of a frame that contains ONLY rows at or before
the information cutoff. The engine strips forward-looking columns before
calling these, so a signal physically cannot read the future; the discipline
here is about not reading the *present* bar in ways that are not observable.

Convention: each function returns a Series indexed by symbol, higher = more
attractive for a LONG. The engine ranks cross-sectionally and takes the top
quantile long, bottom quantile short.

Categories map to the research brief:
    momentum / reversal     : mom_12_1, reversal_1m, reversal_1w
    volume & liquidity      : volume_shock, illiquidity_amihud, turnover_trend
    volatility              : low_vol, vol_of_vol, range_compression
    relative strength       : rs_vs_market, rs_vs_peers
    gap behaviour           : gap_reversal, overnight_intraday_split
    mean reversion          : dist_from_ma, bollinger_z
    market/sector-relative  : beta_adjusted_mom, idio_reversal
    combinations            : combo_* built in research/screen.py

Each has a stated economic hypothesis. A signal without a reason to work is
just a number that happened to fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ helpers
def _pivot(vis: pd.DataFrame, col: str) -> pd.DataFrame:
    """symbols x dates matrix of `col`, sorted ascending by date."""
    return (vis.pivot_table(index="date", columns="symbol", values=col,
                            aggfunc="last").sort_index())


def _last_valid(df: pd.DataFrame, min_obs: int = 1) -> pd.Series:
    ok = df.notna().sum() >= min_obs
    return df.ffill().iloc[-1].where(ok)


def _ret(px: pd.DataFrame, n: int) -> pd.Series:
    if len(px) <= n:
        return pd.Series(dtype=float)
    return px.iloc[-1] / px.iloc[-1 - n] - 1.0


# --------------------------------------------------- momentum / reversal
def mom_12_1(vis: pd.DataFrame) -> pd.Series:
    """
    Cross-sectional momentum, 12 months skipping the most recent month.
    Hypothesis: under-reaction to gradually-diffusing information produces
    continuation at 3-12m horizons; the skipped month avoids the well-known
    short-term reversal that contaminates raw 12m momentum.
    """
    px = _pivot(vis, "adj_close")
    if len(px) < 253:
        return pd.Series(dtype=float)
    return px.iloc[-22] / px.iloc[-252] - 1.0


def reversal_1m(vis: pd.DataFrame) -> pd.Series:
    """
    Short-term reversal: buy last month's losers.
    Hypothesis: liquidity provision. Investors demanding immediacy push prices
    away from fair value; the reversal is compensation for absorbing that.
    Sign is NEGATIVE past return -> higher score.
    """
    px = _pivot(vis, "adj_close")
    r = _ret(px, 21)
    return -r if len(r) else r


def reversal_1w(vis: pd.DataFrame) -> pd.Series:
    """One-week reversal. Same hypothesis, shorter horizon, higher turnover."""
    px = _pivot(vis, "adj_close")
    r = _ret(px, 5)
    return -r if len(r) else r


# ------------------------------------------------- volume & liquidity
def volume_shock(vis: pd.DataFrame) -> pd.Series:
    """
    Lagged volume surge. NOTE: this is the honest version of the signal that
    broke the previous research. It uses volume through the PRIOR bar only.
    Hypothesis: abnormal volume marks information arrival; if the market
    under-reacts, elevated volume precedes continuation.
    """
    vol = _pivot(vis, "volume")
    if len(vol) < 25:
        return pd.Series(dtype=float)
    recent = vol.iloc[-5:].mean()
    base = vol.iloc[-25:-5].mean()
    return np.log((recent / base.replace(0, np.nan)).clip(lower=1e-6))


def illiquidity_amihud(vis: pd.DataFrame) -> pd.Series:
    """
    Amihud illiquidity: mean(|return| / dollar volume).
    Hypothesis: illiquidity is a priced risk factor; illiquid names earn a
    premium. This is also the signal MOST likely to be untradeable in
    practice, which is exactly why the cost model matters.
    """
    px, dv = _pivot(vis, "adj_close"), _pivot(vis, "dollar_volume")
    if len(px) < 25:
        return pd.Series(dtype=float)
    r = px.pct_change(fill_method=None).abs().iloc[-21:]
    d = dv.iloc[-21:].replace(0, np.nan)
    return (r / d).mean() * 1e9


def turnover_trend(vis: pd.DataFrame) -> pd.Series:
    """Rising dollar volume over 3 months. Hypothesis: attention/flow proxy."""
    dv = _pivot(vis, "dollar_volume")
    if len(dv) < 65:
        return pd.Series(dtype=float)
    return np.log((dv.iloc[-21:].mean() / dv.iloc[-63:-21].mean()
                   .replace(0, np.nan)).clip(lower=1e-6))


# -------------------------------------------------------------- volatility
def low_vol(vis: pd.DataFrame) -> pd.Series:
    """
    Low-volatility anomaly: low-vol names have historically outperformed on a
    risk-adjusted basis. Hypothesis: leverage constraints push return-seeking
    investors into high-beta names, overpricing them.
    """
    px = _pivot(vis, "adj_close")
    if len(px) < 65:
        return pd.Series(dtype=float)
    return -px.pct_change(fill_method=None).iloc[-63:].std()


def vol_of_vol(vis: pd.DataFrame) -> pd.Series:
    """Instability of volatility. Hypothesis: unstable-vol names are avoided."""
    px = _pivot(vis, "adj_close")
    if len(px) < 90:
        return pd.Series(dtype=float)
    rv = px.pct_change(fill_method=None).rolling(10).std()
    return -rv.iloc[-63:].std()


def range_compression(vis: pd.DataFrame) -> pd.Series:
    """
    Narrowing high-low range vs its own history.
    Hypothesis: volatility clusters; compression precedes expansion. Direction
    is ambiguous a priori, which is a reason to be suspicious of it.
    """
    hi, lo = _pivot(vis, "high"), _pivot(vis, "low")
    if len(hi) < 65:
        return pd.Series(dtype=float)
    rng = ((hi - lo) / lo.replace(0, np.nan))
    return -(rng.iloc[-10:].mean() / rng.iloc[-63:].mean().replace(0, np.nan))


# --------------------------------------------------------- relative strength
def rs_vs_market(vis: pd.DataFrame) -> pd.Series:
    """
    3-month return minus the equal-weighted universe return.
    Hypothesis: relative strength persists where absolute momentum is noisy.
    """
    px = _pivot(vis, "adj_close")
    if len(px) < 65:
        return pd.Series(dtype=float)
    r = _ret(px, 63)
    return r - r.median()


def rs_vs_peers(vis: pd.DataFrame) -> pd.Series:
    """Relative strength scaled by own vol -- a cross-sectional information ratio."""
    px = _pivot(vis, "adj_close")
    if len(px) < 65:
        return pd.Series(dtype=float)
    r = _ret(px, 63)
    v = px.pct_change(fill_method=None).iloc[-63:].std()
    return (r - r.median()) / v.replace(0, np.nan)


# ------------------------------------------------------------ gap behaviour
def gap_reversal(vis: pd.DataFrame) -> pd.Series:
    """
    Fade the most recent overnight gap.
    Hypothesis: overnight moves in small caps overshoot because the overnight
    book is thin; the day session reverts part of it. Uses the gap into the
    LAST COMPLETED bar, not the entry bar.
    """
    op, cl = _pivot(vis, "open"), _pivot(vis, "close")
    if len(cl) < 3:
        return pd.Series(dtype=float)
    gap = op.iloc[-1] / cl.iloc[-2] - 1.0
    return -gap


def overnight_intraday_split(vis: pd.DataFrame) -> pd.Series:
    """
    Cumulative overnight return minus cumulative intraday return over a month.
    Hypothesis: the two sessions are driven by different participants; a name
    whose gains all come overnight behaves differently from one that grinds up
    during the day.
    """
    op, cl = _pivot(vis, "open"), _pivot(vis, "close")
    if len(cl) < 22:
        return pd.Series(dtype=float)
    overnight = (op / cl.shift(1) - 1).iloc[-21:].sum()
    intraday = (cl / op - 1).iloc[-21:].sum()
    return overnight - intraday


# ---------------------------------------------------------- mean reversion
def dist_from_ma(vis: pd.DataFrame) -> pd.Series:
    """Distance below the 50-day MA. Hypothesis: classic mean reversion."""
    px = _pivot(vis, "adj_close")
    if len(px) < 51:
        return pd.Series(dtype=float)
    ma = px.rolling(50).mean().iloc[-1]
    return -(px.iloc[-1] / ma - 1.0)


def bollinger_z(vis: pd.DataFrame) -> pd.Series:
    """Z-score vs 20-day mean/sd, sign-flipped. Hypothesis: mean reversion."""
    px = _pivot(vis, "adj_close")
    if len(px) < 25:
        return pd.Series(dtype=float)
    w = px.iloc[-20:]
    return -((px.iloc[-1] - w.mean()) / w.std().replace(0, np.nan))


# ------------------------------------------------- market/sector relative
def beta_adjusted_mom(vis: pd.DataFrame) -> pd.Series:
    """
    6-month return with the universe-beta component removed.
    Hypothesis: raw momentum in small caps is largely a beta bet; the residual
    is the part that is actually stock-specific.
    """
    px = _pivot(vis, "adj_close")
    if len(px) < 130:
        return pd.Series(dtype=float)
    rets = px.pct_change(fill_method=None).iloc[-126:]
    mkt = rets.mean(axis=1)
    var_m = mkt.var()
    if not np.isfinite(var_m) or var_m == 0:
        return pd.Series(dtype=float)
    beta = rets.apply(lambda c: c.cov(mkt) / var_m)
    return rets.sum() - beta * mkt.sum()


def idio_reversal(vis: pd.DataFrame) -> pd.Series:
    """
    One-month reversal on the residual after removing the market move.
    Hypothesis: the tradeable part of short-term reversal is idiosyncratic;
    the market component is not something a small account should be fading.
    """
    px = _pivot(vis, "adj_close")
    if len(px) < 130:
        return pd.Series(dtype=float)
    rets = px.pct_change(fill_method=None)
    mkt = rets.mean(axis=1)
    var_m = rets.iloc[-126:].apply(lambda c: mkt.iloc[-126:].var()).iloc[0] \
        if len(rets) > 126 else np.nan
    if not np.isfinite(var_m) or var_m == 0:
        return pd.Series(dtype=float)
    beta = rets.iloc[-126:].apply(lambda c: c.cov(mkt.iloc[-126:]) / var_m)
    resid = rets.iloc[-21:].sum() - beta * mkt.iloc[-21:].sum()
    return -resid


REGISTRY = {
    # category, function, one-line hypothesis
    "mom_12_1":            ("momentum",       mom_12_1),
    "reversal_1m":         ("reversal",       reversal_1m),
    "reversal_1w":         ("reversal",       reversal_1w),
    "volume_shock":        ("volume",         volume_shock),
    "illiquidity_amihud":  ("liquidity",      illiquidity_amihud),
    "turnover_trend":      ("volume",         turnover_trend),
    "low_vol":             ("volatility",     low_vol),
    "vol_of_vol":          ("volatility",     vol_of_vol),
    "range_compression":   ("volatility",     range_compression),
    "rs_vs_market":        ("rel_strength",   rs_vs_market),
    "rs_vs_peers":         ("rel_strength",   rs_vs_peers),
    "gap_reversal":        ("gap",            gap_reversal),
    "overnight_intraday":  ("gap",            overnight_intraday_split),
    "dist_from_ma":        ("mean_reversion", dist_from_ma),
    "bollinger_z":         ("mean_reversion", bollinger_z),
    "beta_adjusted_mom":   ("market_rel",     beta_adjusted_mom),
    "idio_reversal":       ("market_rel",     idio_reversal),
}
