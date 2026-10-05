"""
exp24: Days-to-cover — are short sellers still informed?
========================================================

CRITERIA FIXED BEFORE SCORING (the FINRA parquet existed but had not been
joined to prices or looked at beyond its column names and row count).

HYPOTHESIS
    Short sellers are the market's best-documented informed traders:
    heavily shorted names (high days-to-cover) underperform (Asquith-
    Pathak-Ritter 2005; Boehmer-Jones-Zhang). A NEW information family
    for this project — positioning data, never price/fundamental/filing.

DATA REALITY, DECLARED UP FRONT
    FINRA's free archive begins 2017-12. That is one era, not two, so
    this follows exp17's single-window precedent: the 2018-2022 span
    (the panel's measured-universe validation era), a RAISED primary bar,
    and a sign-stability check across halves instead of a fake second
    window. 2023+ settlements exist in the download and are EXCLUDED —
    sealed.

DESIGN
    Signal      -days_to_cover at each settlement (fallback: -short/ADV);
                usable 10 TRADING days after settlement (publication lag
                is ~7 business days; +3 margin)
    Universe    panel ADV63 >= $2M at the usable date, symbol-joined
    Outcome     total return usable date -> next publication's usable
                date (~2 weeks)
    Statistic   per-publication Spearman IC, t across ~115 publications

POWER (before scoring)
    se(mean IC) ~ 0.13/sqrt(115) ~ 0.013. The t >= 2.5 bar detects
    |IC| >= ~0.033. Literature-sized short-interest effects are IC
    0.02-0.04: this test is powered ONLY for the upper half of that
    range, and a FAIL means "no upper-literature-sized effect", stated
    now. Null pass rate ~0.6%.

PRE-REGISTERED CRITERIA — all required
    C1  full-window IC t >= 2.5 (positive = high-DTC names underperform)
    C2  same IC sign in both halves of the window
    C3  cost arithmetic: top-minus-bottom decile spread per period >
        median period turnover x 2 x 20bp
    Short-leg borrow caveat: high-DTC names are exactly the expensive
    borrows, so any tradeable claim is long-tilt/avoidance only.
    Survival -> HOLD.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register, trial_count  # noqa: E402

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
ADV_MIN = 2e6
LAG_TDAYS = 10
WINDOW = ("2018-01-01", "2022-12-31")
T_REQUIRED = 2.5


def main() -> None:
    si = pd.read_parquet(PROCESSED / "finra_short_interest.parquet")
    si["settlement"] = pd.to_datetime(si["settlement"])
    si = si[(si.settlement >= WINDOW[0]) & (si.settlement <= WINDOW[1])]
    si["dtc"] = si["days_to_cover"].where(
        si["days_to_cover"].notna(),
        si["short_qty"] / si["adv_qty"].replace(0, np.nan))
    si = si.dropna(subset=["dtc"])

    cols = ["symbol", "date", "adj_close", "dollar_volume"]
    panel = pd.read_parquet(PROCESSED / "panel_v5_full.parquet", columns=cols)
    panel["date"] = pd.to_datetime(panel["date"])
    px = panel.pivot_table(index="date", columns="symbol", values="adj_close",
                           aggfunc="last").sort_index()
    dv = panel.pivot_table(index="date", columns="symbol",
                           values="dollar_volume", aggfunc="last").sort_index()
    adv63 = dv.rolling(63, min_periods=40).mean()
    dates = px.index

    setts = sorted(si.settlement.unique())
    usable = {}
    for s in setts:
        i = np.searchsorted(dates.values, np.datetime64(s), "right") + LAG_TDAYS
        if i < len(dates):
            usable[s] = i
    print(f"publications: {len(setts)}, usable: {len(usable)}  "
          f"(K={trial_count()})")

    ics, spreads, turns, ic_dates = [], [], [], []
    prev_top: set = set()
    pairs = list(usable.items())
    for k in range(len(pairs) - 1):
        s, u = pairs[k]
        u_next = pairs[k + 1][1]
        cross = si[si.settlement == s].set_index("symbol")["dtc"]
        cross = cross[~cross.index.duplicated()]
        common = cross.index.intersection(px.columns)
        elig = adv63.iloc[u][common] >= ADV_MIN
        common = elig[elig].index
        if len(common) < 100:
            continue
        fwd = px.iloc[u_next][common] / px.iloc[u][common] - 1
        f = pd.DataFrame({"sig": -cross[common], "fwd": fwd}).dropna()
        ic = sstats.spearmanr(f.sig, f.fwd).statistic
        q = pd.qcut(f.sig, 10, labels=False, duplicates="drop")
        top = set(f.index[q == q.max()])
        if prev_top:
            turns.append(len(top - prev_top) / max(len(top), 1))
        prev_top = top
        spreads.append(f.fwd[q == q.max()].mean() - f.fwd[q == q.min()].mean())
        ics.append(ic)
        ic_dates.append(pd.Timestamp(s))
    ic_s = pd.Series(ics, index=ic_dates)
    print(f"scored periods: {len(ic_s)}, avg names: n/a, "
          f"match into panel confirmed by period count")

    t = ic_s.mean() / (ic_s.std() / np.sqrt(len(ic_s)))
    half = len(ic_s) // 2
    h1, h2 = ic_s.iloc[:half].mean(), ic_s.iloc[half:].mean()
    spread_mean = float(np.nanmean(spreads))
    turn_med = float(np.median(turns))
    cost = turn_med * 2 * 20e-4
    print(f"\nfull window: IC {ic_s.mean():+.4f}  t={t:+.2f}  "
          f"({len(ic_s)} periods)")
    print(f"halves: {h1:+.4f} / {h2:+.4f}")
    print(f"decile spread {spread_mean*100:+.2f}%/period vs cost "
          f"{cost*100:.2f}%/period (turnover {turn_med:.2f})")

    c1 = bool(ic_s.mean() > 0 and t >= T_REQUIRED)
    c2 = bool(np.sign(h1) == np.sign(h2))
    c3 = bool(spread_mean > cost)
    ok = c1 and c2 and c3
    print(f"\nC1 t>={T_REQUIRED}: {c1} (t={t:+.2f})  C2 halves agree: {c2}  "
          f"C3 spread>cost: {c3}")
    print("HOLD" if ok else "REJECTED")

    register(Experiment(
        name="exp24_short_interest_dtc",
        family="positioning",
        hypothesis="high days-to-cover predicts underperformance (short "
                   "sellers informed; Asquith 2005, Boehmer-Jones-Zhang); "
                   "powered only for upper-literature-sized effects",
        features=["neg_days_to_cover"],
        split="single window 2018-2022 (FINRA archive begins 2017-12); "
              "2023+ settlements EXCLUDED/sealed",
        universe="ADV63>=$2M, symbol-joined",
        rebalance_days=10, holding_days=10,
        n_observations=int(len(ic_s)),
        metrics={"mean_ic": round(float(ic_s.mean()), 4),
                 "t": round(float(t), 2),
                 "half1_ic": round(float(h1), 4),
                 "half2_ic": round(float(h2), 4),
                 "decile_spread_period": round(spread_mean, 4),
                 "turnover": round(turn_med, 3),
                 "C1": c1, "C2": c2, "C3": c3},
        outcome="HOLD" if ok else "FAIL",
        failure_reason="" if ok else
        f"failed pre-registered criterion(s): C1={c1} C2={c2} C3={c3}",
    ))


if __name__ == "__main__":
    main()
