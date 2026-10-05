"""
exp25: The weird batch — three mechanism-backed oddities, one sitting.
======================================================================

CRITERIA FIXED BEFORE SCORING, ALL THREE REGISTERED REGARDLESS OF RESULT.
Batch honesty: these are trials 240-242; nothing in this file was run,
peeked at, or tuned before the criteria below were frozen.

A. TUG-OF-WAR (exp25a) — the strong-prior one
    Decompose each stock's return into overnight and intraday legs.
    Stocks whose gains come overnight while daytime bleeds carry the
    retail-attention clientele signature; Lou-Polk-Skouras (2019) and
    follow-ups report they UNDERPERFORM. Mechanism: clientele
    segmentation — the open is retail's market, the day is institutions'.
    Signal   -(trail-63d mean overnight - mean intraday), legs winsorized
             at +/-50%; positive IC confirms
    Universe ADV63 >= $10M (opens must be real prints)
    Test     weekly pooled Spearman IC vs forward 5d, research 2010-2017
             / validation 2018-2022
    POWER    ~1,000+ names x 400/260 weeks -> se(IC) ~ 0.007-0.012;
             detects |IC| >= 0.02 easily. Fully powered.
    F1 research t >= 3.0; F2 validation t >= 2.0 same sign;
    F3 validation Q5-Q1 weekly spread > turnover x 2 x 16bp.

B. THE $5 LINE (exp25b) — the constraint one
    Margin rules and institutional mandates forbid sub-$5 stocks. A stock
    crossing below $5 faces weeks of price-insensitive forced selling —
    the same leverage/mandate-constraint mechanism behind low-vol, which
    is the only anomaly class this project found credible. TRUE closes
    (not Yahoo-adjusted; the exp17 lesson).
    Event    first true_close < $5 after >= 60 consecutive sessions
             >= $5; ADV63 >= $2M; re-arm after 126 sessions back above
    Outcome  market-adjusted return close(E+1) -> close(E+22)
    POWER    expected hundreds of events per window; weekly-clustered t
             detects ~1% drifts. Direction pre-registered NEGATIVE.
    C1 research mean < 0, t <= -3.0; C2 validation mean < 0, t <= -2.0.
    (Up-crosses and early/late decomposition reported, never scored.)

C. FORWARD SPLITS (exp25c) — the attention one
    A voluntary 2-for-1 split buys retail attention and signals
    management confidence; older literature reports positive post-split
    drift. Ratio >= 1.5 only (Yahoo's table mixes spinoff factors into
    small ratios). Weakest prior of the three; stated.
    Event    ex-date, ADV63 >= $2M
    Outcome  market-adjusted close(E+1) -> close(E+22)
    POWER    a few hundred events; weekly-clustered t detects ~1.5-2%
             drifts only — powered for a large effect, stated honestly.
    C1 research mean > 0, t >= 2.5; C2 validation mean > 0, t >= 2.0.
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
INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim"
RESEARCH = ("2010-01-01", "2017-12-31")
VALIDATION = ("2018-01-01", "2022-12-31")


def wk_t(series_by_week: pd.Series) -> float:
    return series_by_week.mean() / (series_by_week.std()
                                    / np.sqrt(len(series_by_week)))


def load_panel():
    cols = ["symbol", "date", "open", "close", "true_close", "adj_close",
            "dollar_volume"]
    p = pd.read_parquet(PROCESSED / "panel_v5_full.parquet", columns=cols)
    p["date"] = pd.to_datetime(p["date"])
    piv = {c: p.pivot_table(index="date", columns="symbol", values=c,
                            aggfunc="last").sort_index()
           for c in ["open", "close", "true_close", "adj_close",
                     "dollar_volume"]}
    piv["adv63"] = piv["dollar_volume"].rolling(63, min_periods=40).mean()
    return piv


def spy_series():
    spy = (pd.read_parquet(PROCESSED / "etf_panel.parquet")
           .query("symbol=='SPY'").set_index("date")["adj_close"])
    spy.index = pd.to_datetime(spy.index)
    return spy


def reg(name, family, hypo, features, uni, hold, n, metrics, ok, why):
    register(Experiment(
        name=name, family=family, hypothesis=hypo, features=features,
        split="research+validation; 2023+ SEALED", universe=uni,
        rebalance_days=hold, holding_days=hold, n_observations=n,
        metrics=metrics, outcome="HOLD" if ok else "FAIL",
        failure_reason="" if ok else why))


# ------------------------------------------------------------ A tug-of-war
def exp25a(piv, spy):
    o, c, adj = piv["open"], piv["close"], piv["adj_close"]
    on = (o / c.shift(1) - 1).clip(-0.5, 0.5)
    intra = (c / o - 1).clip(-0.5, 0.5)
    sig_full = -(on.rolling(63, min_periods=40).mean()
                 - intra.rolling(63, min_periods=40).mean())
    dates = adj.index
    out = {}
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        rows, prev_top = [], set()
        for i in range(70, len(dates) - 7, 5):
            d = dates[i]
            if not (lo <= str(d.date()) <= hi):
                continue
            elig = piv["adv63"].iloc[i] >= 1e7
            f = pd.DataFrame({"sig": sig_full.iloc[i],
                              "fwd": adj.iloc[i + 6] / adj.iloc[i + 1] - 1}
                             )[elig].dropna()
            if len(f) < 200:
                continue
            ic = sstats.spearmanr(f.sig, f.fwd).statistic
            q = pd.qcut(f.sig, 5, labels=False, duplicates="drop")
            top = set(f.index[q == q.max()])
            turn = (len(top - prev_top) / max(len(top), 1)) if prev_top else np.nan
            prev_top = top
            rows.append({"ic": ic, "spread": f.fwd[q == q.max()].mean()
                         - f.fwd[q == 0].mean(), "turn": turn, "n": len(f)})
        w = pd.DataFrame(rows)
        out[name] = {"weeks": len(w), "mean_ic": w.ic.mean(),
                     "ic_t": wk_t(w.ic), "spread_wk": w.spread.mean(),
                     "turnover": w.turn.median(), "avg_n": w.n.mean()}
    r, v = out["research"], out["validation"]
    cost = v["turnover"] * 2 * 16e-4
    f1 = bool(r["ic_t"] >= 3.0)
    f2 = bool(v["ic_t"] >= 2.0 and np.sign(v["mean_ic"]) == np.sign(r["mean_ic"]))
    f3 = bool(v["spread_wk"] > cost)
    ok = f1 and f2 and f3
    print(f"\nA TUG-OF-WAR: research IC {r['mean_ic']:+.4f} t={r['ic_t']:+.2f}"
          f" | validation IC {v['mean_ic']:+.4f} t={v['ic_t']:+.2f}"
          f" | spread {v['spread_wk']*100:+.2f}%/wk vs cost {cost*100:.2f}%"
          f"\n  F1={f1} F2={f2} F3={f3} -> {'HOLD' if ok else 'FAIL'}")
    reg("exp25a_tug_of_war", "intraday_timing",
        "high overnight-share names carry the retail-clientele signature "
        "and underperform (Lou-Polk-Skouras 2019, cross-sectional)",
        ["overnight_minus_intraday_63d"], "ADV63>=$10M", 5,
        int(v["weeks"]),
        {f"{s}_{k}": round(float(x), 4) for s, d in out.items()
         for k, x in d.items()} | {"F1": f1, "F2": f2, "F3": f3},
        ok, f"failed: F1={f1} F2={f2} F3={f3}")


# ------------------------------------------------------------- B $5 line
def exp25b(piv, spy):
    tc, adj, adv = piv["true_close"], piv["adj_close"], piv["adv63"]
    dates = adj.index
    spy_a = spy.reindex(dates).ffill().values
    above = (tc >= 5.0)
    ge60 = above.rolling(60, min_periods=60).sum().eq(60).shift(1)
    cross = (tc < 5.0) & ge60.fillna(False)
    ev = cross.stack()
    ev = ev[ev].reset_index()
    ev.columns = ["date", "symbol", "x"]
    # re-arm: keep first event per symbol per 126 sessions
    ev = ev.sort_values(["symbol", "date"])
    keep = ev.groupby("symbol")["date"].diff().isna() | (
        ev.groupby("symbol")["date"].diff().dt.days > 190)
    ev = ev[keep]
    pos = np.searchsorted(dates.values, ev["date"].values, "left")
    scol = pd.Series(np.arange(adj.shape[1]), index=adj.columns)
    sidx = ev["symbol"].map(scol).astype(int).values
    okm = (pos + 22 < len(dates)) & (adv.values[pos, sidx] >= 2e6)
    ev, pos, sidx = ev[okm], pos[okm], sidx[okm]
    a = adj.values
    drift = (a[pos + 22, sidx] / a[pos + 1, sidx]
             - spy_a[pos + 22] / spy_a[pos + 1])
    ev = ev.assign(drift=drift).dropna(subset=["drift"])
    ev["week"] = ev["date"].dt.to_period("W")
    out = {}
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        w = ev[(ev.date >= lo) & (ev.date <= hi)]
        wk = w.groupby("week")["drift"].mean()
        out[name] = {"events": len(w), "weeks": len(wk),
                     "mean_drift": w.drift.mean(), "t": wk_t(wk)}
    r, v = out["research"], out["validation"]
    c1 = bool(r["mean_drift"] < 0 and r["t"] <= -3.0)
    c2 = bool(v["mean_drift"] < 0 and v["t"] <= -2.0)
    ok = c1 and c2
    print(f"\nB $5 LINE: research {r['mean_drift']*100:+.2f}% t={r['t']:+.2f} "
          f"({r['events']} ev) | validation {v['mean_drift']*100:+.2f}% "
          f"t={v['t']:+.2f} ({v['events']} ev)"
          f"\n  C1={c1} C2={c2} -> {'HOLD' if ok else 'FAIL'}")
    reg("exp25b_five_dollar_line", "corporate_events",
        "crossing below $5 triggers weeks of mandate/margin-constrained "
        "forced selling (institutional-constraint mechanism)",
        ["five_dollar_downcross"], "ADV63>=$2M", 21,
        int(v["events"]),
        {f"{s}_{k}": round(float(x), 4) for s, d in out.items()
         for k, x in d.items()} | {"C1": c1, "C2": c2},
        ok, f"failed: C1={c1} C2={c2}")


# -------------------------------------------------------- C forward splits
def exp25c(piv, spy):
    s = pd.read_parquet(INTERIM / "splits.parquet")
    s["date"] = pd.to_datetime(s["date"])
    s = s[s.ratio >= 1.5]
    adj, adv = piv["adj_close"], piv["adv63"]
    dates = adj.index
    spy_a = spy.reindex(dates).ffill().values
    s = s[s.symbol.isin(adj.columns)]
    pos = np.searchsorted(dates.values, s["date"].values, "left")
    scol = pd.Series(np.arange(adj.shape[1]), index=adj.columns)
    sidx = s["symbol"].map(scol).astype(int).values
    okm = (pos + 22 < len(dates)) & (pos > 0) & (adv.values[np.maximum(pos - 1, 0), sidx] >= 2e6)
    s, pos, sidx = s[okm], pos[okm], sidx[okm]
    a = adj.values
    drift = (a[pos + 22, sidx] / a[pos + 1, sidx]
             - spy_a[pos + 22] / spy_a[pos + 1])
    s = s.assign(drift=drift).dropna(subset=["drift"])
    s["week"] = s["date"].dt.to_period("W")
    out = {}
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        w = s[(s.date >= lo) & (s.date <= hi)]
        wk = w.groupby("week")["drift"].mean()
        out[name] = {"events": len(w), "weeks": len(wk),
                     "mean_drift": w.drift.mean(),
                     "t": wk_t(wk) if len(wk) > 20 else np.nan}
    r, v = out["research"], out["validation"]
    c1 = bool(r["mean_drift"] > 0 and (r["t"] or 0) >= 2.5)
    c2 = bool(v["mean_drift"] > 0 and (v["t"] or 0) >= 2.0)
    ok = c1 and c2
    print(f"\nC FORWARD SPLITS: research {r['mean_drift']*100:+.2f}% "
          f"t={r['t']:+.2f} ({r['events']} ev) | validation "
          f"{v['mean_drift']*100:+.2f}% t={v['t']:+.2f} ({v['events']} ev)"
          f"\n  C1={c1} C2={c2} -> {'HOLD' if ok else 'FAIL'}")
    reg("exp25c_forward_splits", "corporate_events",
        "voluntary forward splits (ratio>=1.5) buy retail attention and "
        "signal confidence; positive post-ex drift (older literature, "
        "weakest prior of the batch)",
        ["forward_split_ex"], "ADV63>=$2M", 21,
        int(v["events"]),
        {f"{s_}_{k}": (round(float(x), 4) if x == x else None)
         for s_, d in out.items() for k, x in d.items()}
        | {"C1": c1, "C2": c2},
        ok, f"failed: C1={c1} C2={c2}")


def main():
    piv = load_panel()
    spy = spy_series()
    print(f"weird batch, K={trial_count()} before registration")
    exp25a(piv, spy)
    exp25b(piv, spy)
    exp25c(piv, spy)


if __name__ == "__main__":
    main()
