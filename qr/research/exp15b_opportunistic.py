"""
exp15b: Opportunistic vs routine insider purchases (Cohen-Malloy-Pomorski).
===========================================================================

CRITERIA FIXED BEFORE SCORING.

HYPOTHESIS
    Most insider buying is calendar-routine (scheduled, diversification,
    comp-cycle) and carries no information. CMP (2012, JF): a trade is
    ROUTINE if the same insider traded in the same calendar month in each
    of the three prior years; the OPPORTUNISTIC remainder predicted ~8%/yr
    abnormal returns in 1989-2007 data. exp15's cluster null does not test
    this cut. This is the last untested variant of the family here.

CLASSIFICATION (CMP, purchases only)
    A purchase by insider O in year Y, month M is ROUTINE iff O also
    purchased in month M of Y-1, Y-2 AND Y-3. Insiders whose first
    purchase is later than Y-3 are UNCLASSIFIED and excluded — the
    classification must be decidable point-in-time.

DESIGN
    Universe ADV63 >= $2M; entry close E+1 after the FILING date; hold 63
    sessions; two calendar-time equal-weight cohorts (opportunistic-bought
    names, routine-bought names). SCORE: daily return spread
    (opportunistic minus routine) — differencing removes market and most
    factor noise, which is where the power comes from.

POWER (before scoring)
    Spread of two overlapping small-cap portfolios: residual daily vol
    ~0.35-0.5% -> annual se over a 5y window ~2.5-3.5%/yr. t >= 2 detects
    ~+6%/yr, against CMP's ~8%/yr. Powered only for a literature-sized
    effect; a decayed 2-3%/yr effect is undetectable here and a FAIL will
    say "no literature-sized effect", not "no effect". Stated now.

PRE-REGISTERED CRITERIA — all required
    C1  research spread > 0, t >= 2.0
    C2  validation spread > 0, t >= 2.0
    TEST 2023+ SEALED. Survival -> HOLD.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register, trial_count  # noqa: E402

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

ADV_MIN = 2e6
HOLD = 63
RESEARCH = ("2010-01-01", "2017-12-31")
VALIDATION = ("2018-01-01", "2022-12-31")


def classify(ins: pd.DataFrame) -> pd.DataFrame:
    ins = ins.dropna(subset=["FILING_DATE", "ISSUERCIK", "RPTOWNERCIK",
                             "dollar_value"])
    ins = ins[ins.dollar_value > 0].copy()
    ins["y"] = ins.FILING_DATE.dt.year
    ins["m"] = ins.FILING_DATE.dt.month
    ym = set(map(tuple, ins[["RPTOWNERCIK", "y", "m"]].drop_duplicates().values))
    first_year = ins.groupby("RPTOWNERCIK")["y"].min()
    ins["has_history"] = ins.RPTOWNERCIK.map(first_year) <= ins.y - 3
    ins["routine"] = [
        (o, y - 1, m) in ym and (o, y - 2, m) in ym and (o, y - 3, m) in ym
        for o, y, m in ins[["RPTOWNERCIK", "y", "m"]].values]
    return ins[ins.has_history]


def main() -> None:
    ins = pd.read_parquet(PROCESSED / "insider_purchases.parquet")
    ins = classify(ins)
    print(f"classified purchases: {len(ins)} "
          f"(routine {ins.routine.mean():.1%})  (K={trial_count()})")

    cols = ["symbol", "date", "cik", "adj_close", "dollar_volume"]
    panel = pd.read_parquet(PROCESSED / "panel_v5_full.parquet", columns=cols)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["cik_s"] = panel["cik"].astype("Int64").astype(str).str.lstrip("0")
    cik2sym = (panel.dropna(subset=["cik"])
               .groupby("cik_s")["symbol"].agg(lambda s: s.mode().iat[0]))
    ins["ISSUERCIK"] = ins.ISSUERCIK.astype(str).str.lstrip("0")
    ins["symbol"] = ins.ISSUERCIK.map(cik2sym)
    ins = ins.dropna(subset=["symbol"])

    px = panel.pivot_table(index="date", columns="symbol", values="adj_close",
                           aggfunc="last").sort_index()
    dv = panel.pivot_table(index="date", columns="symbol",
                           values="dollar_volume", aggfunc="last").sort_index()
    adv63 = dv.rolling(63, min_periods=40).mean()
    daily = px.pct_change()
    dates = px.index
    scol = pd.Series(np.arange(px.shape[1]), index=px.columns)

    pos = np.searchsorted(dates.values, ins["FILING_DATE"].values, "right")
    ok = (pos + 1 + HOLD < len(dates)) & (pos >= 1)
    ins, pos = ins[ok], pos[ok]
    sidx = ins.symbol.map(scol).astype(int).values
    adv_at = adv63.values[np.minimum(pos + 1, len(dates) - 1), sidx]
    keep = adv_at >= ADV_MIN
    ins, pos, sidx = ins[keep], pos[keep], sidx[keep]
    print(f"events in universe: {len(ins)} "
          f"(routine {ins.routine.mean():.1%})")

    def cohort_returns(mask: np.ndarray) -> pd.Series:
        holdings: dict[int, set] = {}
        for p_, s_ in zip(pos[mask], sidx[mask]):
            for t in range(p_ + 2, p_ + 2 + HOLD):
                holdings.setdefault(t, set()).add(s_)
        dr = daily.values
        idx, vals = [], []
        for t, ss in sorted(holdings.items()):
            if t < len(dates):
                idx.append(dates[t])
                vals.append(np.nanmean(dr[t, list(ss)]))
        return pd.Series(vals, index=idx)

    opp = cohort_returns(~ins.routine.values)
    rou = cohort_returns(ins.routine.values)
    spread = (opp - rou).dropna()

    out = {}
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        w = spread[(spread.index >= lo) & (spread.index <= hi)]
        t = w.mean() / (w.std() / np.sqrt(len(w)))
        out[name] = {"days": len(w), "ann_spread": w.mean() * 252, "t": t}
        print(f"{name}: spread {w.mean()*252*100:+.2f}%/yr  t={t:+.2f}  "
              f"({len(w)} days)")

    r, v = out["research"], out["validation"]
    c1 = bool(r["ann_spread"] > 0 and r["t"] >= 2.0)
    c2 = bool(v["ann_spread"] > 0 and v["t"] >= 2.0)
    ok2 = c1 and c2
    print(f"\nC1 research: {c1} (t={r['t']:+.2f})  "
          f"C2 validation: {c2} (t={v['t']:+.2f})")
    print("HOLD" if ok2 else "REJECTED")

    register(Experiment(
        name="exp15b_opportunistic_insiders",
        family="insider",
        hypothesis="only NON-routine (opportunistic) insider purchases carry "
                   "information (Cohen-Malloy-Pomorski 2012); powered only "
                   "for a literature-sized ~8%/yr effect",
        features=["cmp_routine_classifier"],
        split="research+validation; 2023+ SEALED",
        universe="ADV63>=$2M",
        rebalance_days=1, holding_days=HOLD,
        n_observations=int(len(ins)),
        metrics={f"{s}_{k}": round(float(x), 4)
                 for s, d in out.items() for k, x in d.items()}
        | {"C1": c1, "C2": c2},
        outcome="HOLD" if ok2 else "FAIL",
        failure_reason="" if ok2 else
        f"failed pre-registered criterion(s): C1={c1} C2={c2}",
    ))


if __name__ == "__main__":
    main()
