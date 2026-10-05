"""
exp15: Insider cluster purchases (Form 4, open-market code P).
==============================================================

HYPOTHESIS
    Officers and directors buying their own stock with their own money is a
    costly, legally exposed signal. Single purchases are noisy; CLUSTERS —
    several distinct insiders buying within a short window — are the variant
    the literature still reports as economically significant (Cohen-Malloy-
    Pomorski 2012; Alldredge-Cicero 2015). Sales are ignored: they are
    dominated by diversification and compensation mechanics.

WHY THIS FAMILY CAN ESCAPE THE PROJECT'S DEEPEST TRAP
    The panel's informative-censoring problem was that delisting deleted the
    evidence. Form 4 filings are preserved by the SEC regardless of what
    later happened to the issuer, and the DERA datasets are point-in-time by
    FILING_DATE. The signal side is censoring-free; only the OUTCOME side
    still depends on price coverage, which is measured (95% post-2018 in the
    validation window — where every decision is made).

DESIGN (fixed before any result is seen)
    Data       qr/data/processed/insider_purchases.parquet (DERA, 2006->)
    Join       issuer CIK -> panel `cik`; symbol fallback
    Cluster    >=2 distinct insider CIKs filing code-P buys for the same
               issuer within a trailing 21-calendar-day window, combined
               value >= $50,000. Event day E = filing date on which the
               cluster condition first becomes true. Re-triggers allowed
               only after 63 trading days out of the window.
    Universe   trailing-63d mean dollar volume >= $2M at event
    Cutoff     filing public on day E (possibly after hours); enter at the
               CLOSE of E+1 — a full session later, conservative
    Hold       63 trading days
    Costs      per-event round trip = 2 x fitted half-spread at event ADV
               (log10(spread_bps) = 4.288 - 0.416*log10(ADV), halved),
               amortised over the holding window
    Scoring    calendar-time equal-weight portfolio of active events,
               daily excess vs SPY

Pre-registered promotion criteria:
    - validation excess-vs-SPY t >= 2.0
    - net excess return > 0 in BOTH research and validation
    - sign agreement of excess between research and validation
Report-only diagnostics (no promotion weight): size-tercile breakdown,
single-insider comparison, event counts by era.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"

ADV_MIN = 2e6
CLUSTER_WINDOW_DAYS = 21
CLUSTER_MIN_INSIDERS = 2
CLUSTER_MIN_DOLLARS = 50_000
HOLD = 63
RESEARCH = ("2010-01-01", "2017-12-31")
VALIDATION = ("2018-01-01", "2022-12-31")


def half_spread_bps(adv: float) -> float:
    return 0.5 * 10 ** (4.288 - 0.416 * np.log10(max(adv, 1e5)))


def load_panel() -> tuple[pd.DataFrame, pd.Series]:
    cols = ["symbol", "date", "cik", "adj_close", "dollar_volume"]
    p = pd.read_parquet(DATA / "panel_v5_full.parquet", columns=cols)
    p["date"] = pd.to_datetime(p["date"])
    spy = (pd.read_parquet(DATA / "etf_panel.parquet")
           .query("symbol == 'SPY'").set_index("date")["adj_close"])
    spy.index = pd.to_datetime(spy.index)
    return p, spy


def find_clusters(ins: pd.DataFrame) -> pd.DataFrame:
    """Event rows: (issuer_cik, event_date) where a cluster first forms."""
    ins = ins.dropna(subset=["FILING_DATE", "ISSUERCIK", "dollar_value"])
    ins = ins[ins["dollar_value"] > 0]
    ins["ISSUERCIK"] = ins["ISSUERCIK"].astype(str).str.lstrip("0")
    ins["RPTOWNERCIK"] = ins["RPTOWNERCIK"].astype(str)
    events = []
    for cik, g in ins.groupby("ISSUERCIK", sort=False):
        g = g.sort_values("FILING_DATE")
        dates = g["FILING_DATE"].values
        owners = g["RPTOWNERCIK"].values
        dollars = g["dollar_value"].values
        last_event = None
        for i in range(len(g)):
            lo = dates[i] - np.timedelta64(CLUSTER_WINDOW_DAYS, "D")
            w = (dates >= lo) & (dates <= dates[i])
            if (len(set(owners[w])) >= CLUSTER_MIN_INSIDERS
                    and dollars[w].sum() >= CLUSTER_MIN_DOLLARS):
                ts = pd.Timestamp(dates[i])
                if last_event is None or (ts - last_event).days > 91:
                    events.append({"cik": cik, "event_date": ts,
                                   "n_insiders": len(set(owners[w])),
                                   "cluster_dollars": float(dollars[w].sum())})
                    last_event = ts
    return pd.DataFrame(events)


def main() -> None:
    ins = pd.read_parquet(DATA / "insider_purchases.parquet")
    panel, spy = load_panel()
    spy_ret = spy.pct_change()

    ev = find_clusters(ins)
    print(f"cluster events (all issuers): {len(ev)}")

    # map issuer CIK -> panel symbol (panel cik column, zero-stripped)
    panel["cik_s"] = panel["cik"].astype("Int64").astype(str).str.lstrip("0")
    cik2sym = (panel.dropna(subset=["cik"])
               .groupby("cik_s")["symbol"].agg(lambda s: s.mode().iat[0]))
    ev["symbol"] = ev["cik"].map(cik2sym)
    ev = ev.dropna(subset=["symbol"])
    print(f"events matched to panel symbols: {len(ev)}")

    px = panel.pivot_table(index="date", columns="symbol", values="adj_close",
                           aggfunc="last").sort_index()
    dv = panel.pivot_table(index="date", columns="symbol",
                           values="dollar_volume", aggfunc="last").sort_index()
    adv63 = dv.rolling(63, min_periods=40).mean()
    dates = px.index
    date_pos = pd.Series(np.arange(len(dates)), index=dates)

    # entry at close of E+1: first panel date strictly after event date, +1
    ev["e_pos"] = np.searchsorted(dates.values, ev["event_date"].values, "right")
    ev = ev[(ev["e_pos"] + 1 + HOLD) < len(dates)]
    sym_col = pd.Series(np.arange(px.shape[1]), index=px.columns)
    ev["s_idx"] = ev["symbol"].map(sym_col).astype(int)
    ev["entry_pos"] = ev["e_pos"] + 1

    # liquidity filter + event-time cost, both measured AT the event
    adv_at = adv63.values[ev["entry_pos"], ev["s_idx"]]
    ev["adv"] = adv_at
    ev = ev[ev["adv"] >= ADV_MIN]
    ev["rt_cost"] = [2 * half_spread_bps(a) / 1e4 for a in ev["adv"]]
    print(f"events after ADV>=${ADV_MIN/1e6:.0f}M filter: {len(ev)}")

    daily_ret = px.pct_change()

    def calendar_time(cohort: pd.DataFrame) -> pd.DataFrame:
        holdings: dict[int, list[tuple[int, float]]] = {}
        for e, s, c in zip(cohort["entry_pos"], cohort["s_idx"], cohort["rt_cost"]):
            for t in range(e + 1, e + 1 + HOLD):
                holdings.setdefault(t, []).append((s, c / HOLD))
        rows = []
        dr = daily_ret.values
        for t, entries in sorted(holdings.items()):
            if t < len(dates):
                idxs = [s for s, _ in entries]
                cost = float(np.mean([c for _, c in entries]))
                r = np.nanmean(dr[t, idxs]) - cost
                rows.append((dates[t], r, len(idxs)))
        return pd.DataFrame(rows, columns=["date", "ret", "n"]).set_index("date")

    port = calendar_time(ev)

    def perf(lo: str, hi: str) -> dict:
        w = port[(port.index >= lo) & (port.index <= hi)]
        if len(w) < 120:
            return {"n": len(w), "ann_exc": np.nan, "t": np.nan,
                    "avg_names": np.nan, "coverage_days": len(w)}
        exc = (w["ret"] - spy_ret.reindex(w.index).fillna(0)).dropna()
        mu, sd, n = exc.mean(), exc.std(), len(exc)
        return {"n": n, "ann_exc": mu * 252,
                "t": mu / (sd / np.sqrt(n)) if sd > 0 else np.nan,
                "avg_names": round(float(w["n"].mean()), 1)}

    r, v = perf(*RESEARCH), perf(*VALIDATION)
    print(f"\nresearch:   {r}")
    print(f"validation: {v}")

    # report-only: size terciles in validation
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        w = ev[(ev["event_date"] >= lo) & (ev["event_date"] <= hi)]
        print(f"{name}: {len(w)} events, median cluster ${w['cluster_dollars'].median():,.0f}, "
              f"median ADV ${w['adv'].median()/1e6:.1f}M")

    crit = {
        "val_t_ge_2": bool((v["t"] or 0) >= 2.0),
        "net_excess_research": bool((r["ann_exc"] or 0) > 0),
        "net_excess_validation": bool((v["ann_exc"] or 0) > 0),
        "sign_agreement": bool(np.sign(r["ann_exc"] or 0) == np.sign(v["ann_exc"] or 0)),
    }
    passed = all(crit.values())
    print(f"\ncriteria: {crit}\n-> {'PASS' if passed else 'FAIL'}")

    register(Experiment(
        name="exp15_insider_clusters",
        family="insider",
        hypothesis="clustered open-market insider buying is a costly signal "
                   "predicting abnormal returns (Cohen-Malloy-Pomorski 2012); "
                   "signal side is delisting-censoring-free",
        features=["cluster_21d_2insiders_50k"],
        split="research+validation",
        universe=f"ADV63>=${ADV_MIN/1e6:.0f}M",
        rebalance_days=1, holding_days=HOLD,
        n_observations=int(len(ev)),
        metrics={"research_ann_exc": round(float(r["ann_exc"]), 4) if r["ann_exc"] == r["ann_exc"] else None,
                 "research_t": round(float(r["t"]), 2) if r["t"] == r["t"] else None,
                 "validation_ann_exc": round(float(v["ann_exc"]), 4) if v["ann_exc"] == v["ann_exc"] else None,
                 "validation_t": round(float(v["t"]), 2) if v["t"] == v["t"] else None,
                 **{k: bool(x) for k, x in crit.items()}},
        outcome="PASS" if passed else "FAIL",
        failure_reason="" if passed else
        "failed pre-registered criterion(s): "
        + ", ".join(k for k, x in crit.items() if not x),
    ))


if __name__ == "__main__":
    main()
