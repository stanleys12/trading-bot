"""
exp12: VIX term-structure regime filter on SPY.
===============================================

HYPOTHESIS
    The VIX futures/index term structure prices the volatility risk premium.
    In normal times VIX3M > VIX (contango); inversion (backwardation) marks
    stress with elevated crash risk and poor short-horizon equity returns.
    Unlike trailing realised vol (exp06's failure: it de-risks AFTER the
    crash), the term structure is forward-looking — it inverts DURING the
    event and un-inverts when the vol market prices recovery, not when
    trailing windows roll off.

DATA
    ^VIX and ^VIX3M daily closes (CBOE indices via Yahoo), 2007->. Both are
    index levels, no survivorship or adjustment issues. SPY with dividends
    from the frozen ETF panel.

INFORMATION CUTOFF
    Signal at close D uses VIX/VIX3M closes of day D. Position taken at the
    OPEN of D+1 via MOO auction order. Strategy earns open(D+1)->open(D+2).
    Enforced by shifting the position series by one full day against
    open-to-open returns.

STRATEGY
    Long SPY when VIX3M/VIX >= 1 at prior close, else cash (0%).
    No leverage, no shorting.

COSTS
    1bp per auction side + SEC fee on sells, charged per regime flip.

Pre-registered promotion criteria:
    - conditioning is real: mean next-day open-to-open SPY return on
      backwardation days < contango days, sign-consistent in BOTH windows
    - filtered Sharpe >= buy-and-hold Sharpe in BOTH windows
    - filtered MaxDD shallower than buy-and-hold in BOTH windows
    - validation filtered-vs-BH return difference not negative by more than
      1%/yr (the filter must not cost material return to buy its safety)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
PROCESSED = DATA / "processed"
INTERIM = DATA / "interim"
SEC_FEE = 0.0000278
SLIP = 1e-4

RESEARCH = ("2010-01-01", "2017-12-31")
VALIDATION = ("2018-01-01", "2022-12-31")


def load_vix() -> pd.DataFrame:
    """VIX and VIX3M closes, cached locally so the experiment is reproducible."""
    cache = INTERIM / "vix_term_structure.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frames = {}
    for sym in ["^VIX", "^VIX3M"]:
        h = yf.Ticker(sym).history(start="2006-01-01", auto_adjust=False)
        h.index = pd.to_datetime(h.index).tz_localize(None)
        frames[sym.strip("^")] = h["Close"]
    df = pd.DataFrame(frames).dropna()
    df.to_parquet(cache)
    return df


def load_spy() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "etf_panel.parquet")
    df["date"] = pd.to_datetime(df["date"])
    s = (df[df["symbol"] == "SPY"].sort_values("date").set_index("date"))
    # dividend-adjusted open so open-to-open returns are total returns
    s["adj_open"] = s["open"] * s["adj_close"] / s["close"]
    return s[["adj_open"]].dropna()


def perf(r: pd.Series) -> dict:
    n, mu, sd = len(r), r.mean(), r.std()
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {"n": n, "ann_ret": mu * 252, "ann_vol": sd * np.sqrt(252),
            "sharpe": (mu * 252) / (sd * np.sqrt(252)) if sd > 0 else np.nan,
            "maxdd": dd,
            "t": mu / (sd / np.sqrt(n)) if sd > 0 else np.nan}


def main() -> None:
    vix = load_vix()
    spy = load_spy()

    # open-to-open total return: r_oo(t) = adj_open(t+1)/adj_open(t) - 1
    # is earned by holding from open(t) to open(t+1).
    r_oo = (spy["adj_open"].shift(-1) / spy["adj_open"] - 1).dropna()

    # signal at close D: contango if VIX3M >= VIX. Tradeable at open D+1.
    contango = (vix["VIX3M"] / vix["VIX"] >= 1.0)
    # position held from open(t) to open(t+1) uses signal from close(t-1)
    pos = contango.shift(1).reindex(r_oo.index).ffill()
    n_missing = pos.isna().sum()
    pos = pos.fillna(True)          # before VIX3M history: default long
    flips = pos.astype(int).diff().abs().fillna(0)
    cost = flips * (SLIP + SEC_FEE / 2) * 2       # both auction sides per flip
    strat = pos * r_oo - cost

    print(f"days: {len(r_oo)}, missing-signal days defaulted long: {n_missing}")
    print(f"backwardation frequency: {(~contango).mean():.1%}")

    results = {}
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        idx = (r_oo.index >= lo) & (r_oo.index <= hi)
        w_r, w_s, w_p = r_oo[idx], strat[idx], pos[idx]
        cond_on = w_r[w_p]           # returns earned while long (contango)
        cond_off = w_r[~w_p]         # returns avoided (backwardation)
        results[name] = {"bh": perf(w_r), "strat": perf(w_s),
                         "on": perf(cond_on), "off": perf(cond_off),
                         "flips_per_yr": flips[idx].sum() / (idx.sum() / 252)}
        print(f"\n=== {name} ===  (regime flips/yr: {results[name]['flips_per_yr']:.1f})")
        for k in ["bh", "strat", "on", "off"]:
            p = results[name][k]
            print(f"{k:>6}: n={p['n']:5d} ann={p['ann_ret']:+.3f} vol={p['ann_vol']:.3f} "
                  f"sharpe={p['sharpe']:+.2f} maxdd={p['maxdd']:+.3f} t={p['t']:+.2f}")

    r, v = results["research"], results["validation"]
    crit = {
        "conditioning_sign_research": bool(r["off"]["ann_ret"] < r["on"]["ann_ret"]),
        "conditioning_sign_validation": bool(v["off"]["ann_ret"] < v["on"]["ann_ret"]),
        "sharpe_beats_bh_research": bool(r["strat"]["sharpe"] >= r["bh"]["sharpe"]),
        "sharpe_beats_bh_validation": bool(v["strat"]["sharpe"] >= v["bh"]["sharpe"]),
        "dd_shallower_research": bool(r["strat"]["maxdd"] > r["bh"]["maxdd"]),
        "dd_shallower_validation": bool(v["strat"]["maxdd"] > v["bh"]["maxdd"]),
        "val_return_give_up_le_1pct": bool(
            v["strat"]["ann_ret"] - v["bh"]["ann_ret"] >= -0.01),
    }
    passed = all(crit.values())
    print(f"\ncriteria: {crit}\n-> {'PASS' if passed else 'FAIL'}")

    register(Experiment(
        name="exp12_vix_regime_SPY",
        family="vol_regime",
        hypothesis="VIX term-structure inversion is a forward-looking stress "
                   "signal; long SPY only in contango preserves return with "
                   "shallower drawdowns (vol risk premium literature)",
        features=["vix3m_over_vix"],
        split="research+validation",
        universe="SPY",
        rebalance_days=1, holding_days=1,
        n_observations=v["strat"]["n"],
        metrics={f"{w}_{k}": round(results[w]["strat"][k], 3)
                 for w in ["research", "validation"]
                 for k in ["ann_ret", "sharpe", "maxdd"]}
        | {f"{w}_bh_{k}": round(results[w]["bh"][k], 3)
           for w in ["research", "validation"]
           for k in ["ann_ret", "sharpe", "maxdd"]}
        | {k: bool(x) for k, x in crit.items()},
        outcome="PASS" if passed else "FAIL",
        failure_reason="" if passed else
        "failed pre-registered criterion(s): "
        + ", ".join(k for k, x in crit.items() if not x),
    ))


if __name__ == "__main__":
    main()
