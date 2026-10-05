"""
exp11: Turn-of-month effect in SPY.
===================================

HYPOTHESIS
    Equity returns concentrate in the window from the last trading day of the
    month through the first three trading days of the next (Lakonishok-Smidt
    1988; Ogden 1990: pay-cycle / pension contribution flows; McConnell-Xu
    2008 confirm out of sample). Flows are mechanical and calendar-known, so
    if it persists it is cheap to time: 2 auction trades per month.

INFORMATION CUTOFF
    The signal is the calendar. Known infinitely far in advance.

DEFINITION (fixed before scoring)
    TOM window = last trading day of month M through 3rd trading day of M+1
    (4 trading days per month, ~19% market exposure).

COSTS
    24 auction round-trips/yr in SPY. Charged at 1bp per side + SEC fee.

Pre-registered promotion criteria:
    - TOM-minus-rest daily mean difference t >= 2.0 in validation
    - sign agreement of the difference across research and validation
    - TOM-only net Sharpe >= buy-and-hold Sharpe in BOTH windows
      (else it is a curiosity, not a usable position)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"
SEC_FEE = 0.0000278
SLIP = 1e-4                     # 1bp per auction side

RESEARCH = ("2010-01-01", "2017-12-31")
VALIDATION = ("2018-01-01", "2022-12-31")


def tom_mask(dates: pd.DatetimeIndex) -> pd.Series:
    """True on the last trading day of a month and first 3 of the next."""
    d = pd.Series(index=dates, dtype=bool)
    month = dates.to_period("M")
    is_last = month != np.roll(month, -1)          # next day is a new month
    is_last[-1] = False                            # unknown for final day
    # rank of day within its month (1-based)
    rank = pd.Series(1, index=dates).groupby(month).cumsum()
    d[:] = is_last | (rank.values <= 3)
    return d


def stats(r: pd.Series) -> dict:
    n, mu, sd = len(r), r.mean(), r.std()
    return {"n": n, "ann_ret": mu * 252, "ann_vol": sd * np.sqrt(252),
            "sharpe": (mu * 252) / (sd * np.sqrt(252)) if sd > 0 else np.nan,
            "t": mu / (sd / np.sqrt(n)) if sd > 0 else np.nan}


def welch_t(a: pd.Series, b: pd.Series) -> float:
    va, vb = a.var() / len(a), b.var() / len(b)
    return (a.mean() - b.mean()) / np.sqrt(va + vb)


def main() -> None:
    df = pd.read_parquet(DATA / "etf_panel.parquet")
    df["date"] = pd.to_datetime(df["date"])
    spy = (df[df["symbol"] == "SPY"].sort_values("date")
           .set_index("date")["adj_close"].dropna())
    ret = spy.pct_change().dropna()
    mask = tom_mask(ret.index)

    results = {}
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        w = ret[(ret.index >= lo) & (ret.index <= hi)]
        m = mask.loc[w.index]
        tom, rest = w[m], w[~m]
        # TOM-only strategy: in market only during TOM days, else cash (0%)
        # 2 auction sides per month at SLIP each + SEC fee monthly
        n_months = w.index.to_period("M").nunique()
        drag_daily = (2 * SLIP + SEC_FEE) * n_months / len(w)
        strat = w.where(m, 0.0) - drag_daily
        results[name] = {
            "tom": stats(tom), "rest": stats(rest), "bh": stats(w),
            "strat": stats(strat), "diff_t": welch_t(tom, rest),
        }
        print(f"\n=== {name} ===")
        print(f"TOM  days: {stats(tom)}")
        print(f"rest days: {stats(rest)}")
        print(f"diff t (Welch): {welch_t(tom, rest):.2f}")
        print(f"TOM-only strategy net: {stats(strat)}")
        print(f"buy-and-hold:          {stats(w)}")

    r, v = results["research"], results["validation"]
    crit = {
        "val_diff_t_ge_2": bool(v["diff_t"] >= 2.0),
        "sign_agreement": bool(
            np.sign(r["tom"]["ann_ret"] - r["rest"]["ann_ret"])
            == np.sign(v["tom"]["ann_ret"] - v["rest"]["ann_ret"])),
        "net_sharpe_beats_bh_research": bool(r["strat"]["sharpe"] >= r["bh"]["sharpe"]),
        "net_sharpe_beats_bh_validation": bool(v["strat"]["sharpe"] >= v["bh"]["sharpe"]),
    }
    passed = all(crit.values())
    print(f"\ncriteria: {crit}  ->  {'PASS' if passed else 'FAIL'}")

    register(Experiment(
        name="exp11_turn_of_month_SPY",
        family="calendar",
        hypothesis="pay-cycle and pension flows concentrate returns at the "
                   "turn of the month (Lakonishok-Smidt 1988, Ogden 1990)",
        features=["tom_window"],
        split="research+validation",
        universe="SPY",
        rebalance_days=21, holding_days=4,
        n_observations=v["strat"]["n"],
        metrics={"research_tom_ann": round(r["tom"]["ann_ret"], 4),
                 "research_rest_ann": round(r["rest"]["ann_ret"], 4),
                 "validation_tom_ann": round(v["tom"]["ann_ret"], 4),
                 "validation_rest_ann": round(v["rest"]["ann_ret"], 4),
                 "validation_diff_t": round(v["diff_t"], 2),
                 "research_net_sharpe": round(r["strat"]["sharpe"], 3),
                 "validation_net_sharpe": round(v["strat"]["sharpe"], 3),
                 **{k: bool(x) for k, x in crit.items()}},
        outcome="PASS" if passed else "FAIL",
        failure_reason="" if passed else
        "failed pre-registered criterion(s): "
        + ", ".join(k for k, x in crit.items() if not x),
    ))


if __name__ == "__main__":
    main()
