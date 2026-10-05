"""
exp13: Post-filing announcement-reaction drift (PEAD variant), liquid universe.
===============================================================================

HYPOTHESIS
    Prices underreact to earnings news: the direction of the market's initial
    reaction to a filing predicts continued drift for weeks (Ball-Brown 1968;
    Bernard-Thomas 1989; still the most replicated anomaly in the literature).
    We proxy the surprise with the market's own reaction — the market-adjusted
    return around the filing — which needs no analyst data and is point-in-time
    by construction.

WHY THIS IS A NEW FAMILY HERE
    All six families previously tested were LEVEL characteristics (vol, value,
    issuance, momentum...). This is an EVENT family: the information is a
    dated disclosure, the panel's `filed` column is joined as-of SEC filing
    date, and filings survive delisting — partially escaping the censoring
    problem that poisoned the microcap work.

KNOWN LIMITATION (stated before scoring)
    `filed` is the 10-K/10-Q SEC filing date, not the earnings press-release
    date. Many firms announce via 8-K days earlier, which BLURS the event and
    biases us toward finding nothing. A null here does not kill PEAD; a
    positive here is conservative.

DESIGN
    Universe   trailing-63d mean dollar volume >= $50M at event time
               (measured half-spread 4.6-7.5bps)
    Event      first panel date E where `filed` changes and (E - filed) <= 5d
    Reaction   adj_close(E+1)/adj_close(E-1) - 1, minus SPY same window
    Signal     reaction >= +5%  -> positive surprise cohort
               reaction <= -5%  -> negative surprise cohort  (fixed
               thresholds; no cross-sectional ranking, no peeking)
    Cutoff     reaction known at close E+1; enter at close E+2 (one full day
               later than necessary — conservative)
    Hold       42 trading days
    Scoring    calendar-time portfolio (Fama): each day equal-weight all
               names within their holding window; daily excess vs SPY;
               t-stat on that daily series. Costs 7.5bps/side amortised.

Pre-registered promotion criteria:
    - positive-cohort excess-vs-SPY t >= 2.0 in validation
    - positive-cohort net excess return > 0 in BOTH windows
    - monotonicity: positive cohort beats negative cohort in BOTH windows
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"

ADV_MIN = 50e6
REACT_THRESH = 0.05
HOLD = 42
COST_PER_SIDE = 7.5e-4 / 10 * 10   # 7.5 bps
RESEARCH = ("2010-01-01", "2017-12-31")
VALIDATION = ("2018-01-01", "2022-12-31")


def load() -> tuple[pd.DataFrame, pd.Series]:
    cols = ["symbol", "date", "filed", "adj_close", "dollar_volume"]
    p = pd.read_parquet(DATA / "panel_v5_full.parquet", columns=cols)
    p["date"] = pd.to_datetime(p["date"])
    p["filed"] = pd.to_datetime(p["filed"])
    p = p.sort_values(["symbol", "date"])
    spy = (pd.read_parquet(DATA / "etf_panel.parquet")
           .query("symbol == 'SPY'").set_index("date")["adj_close"])
    spy.index = pd.to_datetime(spy.index)
    return p, spy


def find_events(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby("symbol", sort=False)
    p = p.assign(
        adv63=g["dollar_volume"].transform(lambda s: s.rolling(63, min_periods=40).mean()),
        prev_filed=g["filed"].shift(1),
        row_in_sym=g.cumcount(),
    )
    ev = p[(p["filed"].notna())
           & (p["filed"] != p["prev_filed"])
           & (p["prev_filed"].notna())            # not the first-ever join row
           & ((p["date"] - p["filed"]).dt.days.between(0, 5))
           & (p["adv63"] >= ADV_MIN)].copy()
    return ev[["symbol", "date"]].rename(columns={"date": "event_date"})


def main() -> None:
    p, spy = load()
    spy_ret = spy.pct_change()

    # per-symbol adj_close pivot for fast windowed returns
    px = p.pivot_table(index="date", columns="symbol", values="adj_close",
                       aggfunc="last").sort_index()
    dates = px.index
    date_pos = pd.Series(np.arange(len(dates)), index=dates)

    events = find_events(p)
    events = events[events["event_date"].isin(dates)]
    print(f"filing events in liquid universe: {len(events)}")

    # reaction = market-adjusted return close(E-1) -> close(E+1)
    pos_of = events["event_date"].map(date_pos).astype(int).values
    ok = (pos_of >= 1) & (pos_of + 2 + HOLD < len(dates))
    events, pos_of = events[ok], pos_of[ok]

    sym_idx = events["symbol"].map(
        pd.Series(np.arange(px.shape[1]), index=px.columns)).astype(int).values
    a = px.values
    c_m1 = a[pos_of - 1, sym_idx]
    c_p1 = a[pos_of + 1, sym_idx]
    spy_a = spy.reindex(dates).ffill().values
    react = (c_p1 / c_m1 - 1) - (spy_a[pos_of + 1] / spy_a[pos_of - 1] - 1)
    events = events.assign(reaction=react, entry_pos=pos_of + 2)
    events = events.dropna(subset=["reaction"])
    print(f"scored events: {len(events)}; "
          f"pos cohort: {(events['reaction'] >= REACT_THRESH).sum()}, "
          f"neg cohort: {(events['reaction'] <= -REACT_THRESH).sum()}")

    daily_ret = px.pct_change()

    def calendar_time(cohort: pd.DataFrame) -> pd.Series:
        """Daily equal-weight return of all names inside their holding window."""
        holdings: dict[int, list[int]] = {}
        for e_pos, s_idx in zip(cohort["entry_pos"], cohort["symbol"].map(
                pd.Series(np.arange(px.shape[1]), index=px.columns)).astype(int)):
            for t in range(e_pos + 1, e_pos + 1 + HOLD):   # earn from day after entry close
                holdings.setdefault(t, []).append(s_idx)
        rows = []
        dr = daily_ret.values
        for t, idxs in sorted(holdings.items()):
            if t < len(dates):
                r = np.nanmean(dr[t, idxs])
                rows.append((dates[t], r, len(idxs)))
        out = pd.DataFrame(rows, columns=["date", "ret", "n"]).set_index("date")
        # amortised round-trip cost: each event pays 2 sides over HOLD days
        out["ret"] -= 2 * COST_PER_SIDE / HOLD
        return out

    def perf(port: pd.DataFrame, lo: str, hi: str) -> dict:
        w = port[(port.index >= lo) & (port.index <= hi)]
        if len(w) < 60:
            return {"n": len(w), "ann_exc": np.nan, "t": np.nan, "avg_names": np.nan}
        exc = w["ret"] - spy_ret.reindex(w.index).fillna(0)
        mu, sd, n = exc.mean(), exc.std(), len(exc)
        return {"n": n, "ann_exc": mu * 252,
                "t": mu / (sd / np.sqrt(n)) if sd > 0 else np.nan,
                "avg_names": w["n"].mean()}

    pos_port = calendar_time(events[events["reaction"] >= REACT_THRESH])
    neg_port = calendar_time(events[events["reaction"] <= -REACT_THRESH])

    results = {}
    for wname, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        rp, rn = perf(pos_port, lo, hi), perf(neg_port, lo, hi)
        results[wname] = {"pos": rp, "neg": rn}
        print(f"\n=== {wname} ===")
        print(f"positive cohort: {rp}")
        print(f"negative cohort: {rn}")

    r, v = results["research"], results["validation"]
    crit = {
        "val_pos_t_ge_2": bool((v["pos"]["t"] or 0) >= 2.0),
        "net_excess_pos_research": bool((r["pos"]["ann_exc"] or 0) > 0),
        "net_excess_pos_validation": bool((v["pos"]["ann_exc"] or 0) > 0),
        "monotonic_research": bool((r["pos"]["ann_exc"] or 0) > (r["neg"]["ann_exc"] or 0)),
        "monotonic_validation": bool((v["pos"]["ann_exc"] or 0) > (v["neg"]["ann_exc"] or 0)),
    }
    passed = all(crit.values())
    print(f"\ncriteria: {crit}\n-> {'PASS' if passed else 'FAIL'}")

    register(Experiment(
        name="exp13_filing_drift_liquid",
        family="events",
        hypothesis="market reaction to SEC filing predicts continued drift "
                   "(PEAD via announcement-reaction proxy, Ball-Brown 1968)",
        features=["filing_reaction_2d"],
        split="research+validation",
        universe=f"ADV63>=${ADV_MIN/1e6:.0f}M",
        rebalance_days=1, holding_days=HOLD,
        n_observations=int(len(events)),
        metrics={f"{w}_{c}_{k}": (round(float(results[w][c][k]), 4)
                                  if results[w][c][k] == results[w][c][k] else None)
                 for w in ["research", "validation"] for c in ["pos", "neg"]
                 for k in ["ann_exc", "t"]}
        | {k: bool(x) for k, x in crit.items()},
        outcome="PASS" if passed else "FAIL",
        failure_reason="" if passed else
        "failed pre-registered criterion(s): "
        + ", ".join(k for k, x in crit.items() if not x),
    ))


if __name__ == "__main__":
    main()
