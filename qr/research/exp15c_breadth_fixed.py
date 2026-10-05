"""
exp15c: Opportunistic insiders, breadth-fixed measurement.
==========================================================

ORIGIN, STATED HONESTLY
    DATA-INSPIRED REDESIGN of exp15b, formed after seeing that its
    validation spread carried a +37%/yr point estimate with t=1.30 —
    noise of ~4.5%/day that is a property of the INSTRUMENT, not the
    effect: on many days the routine cohort holds a handful of names and
    one blowup dominates. This is a thermometer repair, not a new
    hypothesis; it still pays its own K and this label.

THE ONE CHANGE (nothing else moves)
    The daily spread is scored ONLY on days when BOTH cohorts hold at
    least 20 names. Everything else — CMP classifier, universe, entry
    close E+1, 63-session hold — is identical to exp15b.

PRE-REGISTERED CRITERIA (before running; identical bars to exp15b)
    C1  research spread > 0, t >= 2.0
    C2  validation spread > 0, t >= 2.0
    Also reported: fraction of days surviving the breadth floor. If fewer
    than half of validation days survive, the design is declared
    UNMEASURABLE regardless of t — a spread that mostly cannot be formed
    is not a strategy.
    TEST 2023+ SEALED. Survival -> HOLD.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register, trial_count  # noqa: E402
from qr.research.exp15b_opportunistic import classify  # noqa: E402

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
ADV_MIN, HOLD, MIN_NAMES = 2e6, 63, 20
RESEARCH = ("2010-01-01", "2017-12-31")
VALIDATION = ("2018-01-01", "2022-12-31")


def main() -> None:
    ins = classify(pd.read_parquet(PROCESSED / "insider_purchases.parquet"))
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

    def cohort(mask: np.ndarray) -> tuple[pd.Series, pd.Series]:
        holdings: dict[int, set] = {}
        for p_, s_ in zip(pos[mask], sidx[mask]):
            for t in range(p_ + 2, p_ + 2 + HOLD):
                holdings.setdefault(t, set()).add(s_)
        dr = daily.values
        idx, vals, ns = [], [], []
        for t, ss in sorted(holdings.items()):
            if t < len(dates):
                idx.append(dates[t])
                vals.append(np.nanmean(dr[t, list(ss)]))
                ns.append(len(ss))
        return (pd.Series(vals, index=idx), pd.Series(ns, index=idx))

    opp, n_opp = cohort(~ins.routine.values)
    rou, n_rou = cohort(ins.routine.values)
    joint = pd.DataFrame({"opp": opp, "rou": rou,
                          "n_opp": n_opp, "n_rou": n_rou}).dropna()
    joint["spread"] = joint.opp - joint.rou
    ok_days = (joint.n_opp >= MIN_NAMES) & (joint.n_rou >= MIN_NAMES)
    print(f"events: {len(ins)} | days scored: {ok_days.sum()}/{len(joint)} "
          f"({ok_days.mean():.0%} survive breadth floor)  (K={trial_count()})")

    out, frac = {}, {}
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        w_all = joint[(joint.index >= lo) & (joint.index <= hi)]
        w = w_all[ok_days.reindex(w_all.index).fillna(False)]
        frac[name] = len(w) / max(len(w_all), 1)
        t = w.spread.mean() / (w.spread.std() / np.sqrt(len(w))) if len(w) > 50 else np.nan
        out[name] = {"days": len(w), "frac_scored": frac[name],
                     "ann_spread": w.spread.mean() * 252, "t": t,
                     "median_n_rou": float(w.n_rou.median()) if len(w) else np.nan}
        print(f"{name}: {len(w)} days ({frac[name]:.0%}), spread "
              f"{w.spread.mean()*252*100:+.2f}%/yr, t={t:+.2f}, "
              f"median routine breadth {out[name]['median_n_rou']:.0f}")

    r, v = out["research"], out["validation"]
    measurable = bool(frac["validation"] >= 0.5)
    c1 = bool(measurable and r["ann_spread"] > 0 and (r["t"] or 0) >= 2.0)
    c2 = bool(measurable and v["ann_spread"] > 0 and (v["t"] or 0) >= 2.0)
    ok2 = c1 and c2
    print(f"\nmeasurable (>=50% val days): {measurable}")
    print(f"C1 research: {c1} (t={r['t']:+.2f})  C2 validation: {c2} (t={v['t']:+.2f})")
    print("HOLD" if ok2 else "REJECTED")

    register(Experiment(
        name="exp15c_opportunistic_breadth_fixed",
        family="insider",
        hypothesis="DATA-INSPIRED instrument repair of exp15b: same CMP "
                   "hypothesis, spread scored only on days both cohorts "
                   "hold >=20 names",
        features=["cmp_routine_classifier", "breadth_floor_20"],
        split="research+validation; 2023+ SEALED",
        universe="ADV63>=$2M",
        rebalance_days=1, holding_days=HOLD,
        n_observations=int(v["days"]),
        metrics={f"{s}_{k}": (round(float(x), 4) if x == x else None)
                 for s, d in out.items() for k, x in d.items()}
        | {"C1": c1, "C2": c2, "measurable": measurable, "data_inspired": True},
        outcome="HOLD" if ok2 else "FAIL",
        failure_reason="" if ok2 else
        f"failed pre-registered criterion(s): measurable={measurable} "
        f"C1={c1} C2={c2}",
    ))


if __name__ == "__main__":
    main()
