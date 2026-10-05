"""
exp19: Post-offering drift after shelf takedowns (424B5).
=========================================================

CRITERIA FIXED BEFORE SCORING.

HYPOTHESIS
    A 424B5 prospectus supplement is the moment a shelf registration
    becomes actual dilution: new shares priced, usually at a discount,
    often into weakness. Literature on SEOs documents negative announcement
    effects and continued short-horizon underperformance (Loughran-Ritter
    1995 long-run; Corwin 2003 discounting). Small caps digest the overhang
    slowly. Prediction: negative market-adjusted drift in the two weeks
    AFTER the takedown becomes public.

EVENT
    First 424B5 per CIK with no prior 424B5 in the previous 21 days,
    from the on-disk EDGAR form indexes (2010-2026). CIK -> symbol via the
    panel. ADV63 >= $2M at event. Filing may be after hours: drift is
    measured from close(E+1) to close(E+11) — ten sessions, one full
    session of lag.

STATISTIC (the exp09 clustering lesson)
    Events grouped by calendar week; weekly mean drift; t across weeks.
    POWER: validation should hold ~1,000+ events over ~260 weeks; weekly
    mean sd ~ 4-5% (small caps, few events/week) -> se ~ 0.25-0.3% per
    10 sessions. t >= 2 detects a -0.6% drift against literature effects
    of -1 to -3%. Under the null this passes ~2.5% of the time. Powered.

PRE-REGISTERED CRITERIA — both required
    C1  research (2010-2017): mean weekly drift < 0, t <= -3.0
    C2  validation (2018-2022): mean weekly drift < 0, t <= -2.0
    TEST 2023+ SEALED.

IMPLEMENTABILITY, STATED BEFORE RESULTS
    Shorting the event names costs ~2-2.5% per 10 sessions (borrow at
    ~50%/yr for small caps + spread), so a SHORT is viable only if drift
    <= -2.5%: reported, not required. The primary use of a smaller real
    effect is as an AVOIDANCE overlay for any long book in this band
    (including exp17's buy candidates), which costs nothing to apply.
    Announcement-window reaction close(E-1)->close(E+1) is reported
    descriptively, never scored.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register, trial_count  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RAW_IDX = HERE / "data" / "raw" / "edgar_index"
PROCESSED = HERE / "data" / "processed"
INTERIM = HERE / "data" / "interim"

ADV_MIN = 2e6
DRIFT_START, DRIFT_END = 1, 11        # close(E+1) -> close(E+11)
RESEARCH = ("2010-01-01", "2017-12-31")
VALIDATION = ("2018-01-01", "2022-12-31")
T_RESEARCH, T_VALIDATION = -3.0, -2.0


def extract_424b5() -> pd.DataFrame:
    cache = INTERIM / "edgar_424b5.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    for f in sorted(RAW_IDX.glob("form_*.idx")):
        for line in f.read_text(errors="replace").splitlines():
            if line.startswith("424B5 "):
                parts = line.split()
                try:
                    date = next(p for p in parts if len(p) == 10
                                and p[4] == "-" and p[7] == "-")
                    cik = parts[parts.index(date) - 1]
                    rows.append((int(cik), date))
                except (StopIteration, ValueError):
                    continue
    df = pd.DataFrame(rows, columns=["cik", "filing_date"]).drop_duplicates()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df.to_parquet(cache)
    return df


def main() -> None:
    ev = extract_424b5()
    print(f"424B5 filings: {len(ev)} across {ev.cik.nunique()} CIKs  "
          f"(K={trial_count()})")

    cols = ["symbol", "date", "cik", "adj_close", "dollar_volume"]
    panel = pd.read_parquet(PROCESSED / "panel_v5_full.parquet", columns=cols)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["cik_i"] = panel["cik"].astype("Int64")
    cik2sym = (panel.dropna(subset=["cik"])
               .groupby("cik_i")["symbol"].agg(lambda s: s.mode().iat[0]))
    ev["symbol"] = ev["cik"].map(cik2sym)
    ev = ev.dropna(subset=["symbol"]).sort_values(["cik", "filing_date"])
    ev = ev[ev.groupby("cik")["filing_date"].diff().isna()
            | (ev.groupby("cik")["filing_date"].diff().dt.days > 21)]
    print(f"matched + deduped events: {len(ev)}")

    px = panel.pivot_table(index="date", columns="symbol", values="adj_close",
                           aggfunc="last").sort_index()
    dv = panel.pivot_table(index="date", columns="symbol",
                           values="dollar_volume", aggfunc="last").sort_index()
    adv63 = dv.rolling(63, min_periods=40).mean()
    spy = (pd.read_parquet(PROCESSED / "etf_panel.parquet")
           .query("symbol=='SPY'").set_index("date")["adj_close"])
    spy.index = pd.to_datetime(spy.index)
    spy = spy.reindex(px.index).ffill()

    dates = px.index
    pos = np.searchsorted(dates.values, ev["filing_date"].values, "right")
    ok = (pos >= 2) & (pos + DRIFT_END < len(dates))
    ev, pos = ev[ok], pos[ok]
    sidx = ev["symbol"].map(pd.Series(np.arange(px.shape[1]), index=px.columns)).values
    a, s = px.values, spy.values

    adv_at = adv63.values[pos, sidx]
    keep = adv_at >= ADV_MIN
    ev, pos, sidx = ev[keep], pos[keep], sidx[keep]
    print(f"after ADV>=$2M: {len(ev)} events")

    drift = (a[pos + DRIFT_END, sidx] / a[pos + DRIFT_START, sidx]
             - s[pos + DRIFT_END] / s[pos + DRIFT_START])
    react = (a[pos + 1, sidx] / a[pos - 1, sidx] - s[pos + 1] / s[pos - 1])
    ev = ev.assign(drift=drift, react=react).dropna(subset=["drift"])
    ev["week"] = ev["filing_date"].dt.to_period("W")

    out = {}
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        w = ev[(ev.filing_date >= lo) & (ev.filing_date <= hi)]
        wk = w.groupby("week")["drift"].mean()
        t = wk.mean() / (wk.std() / np.sqrt(len(wk)))
        out[name] = {"events": len(w), "weeks": len(wk),
                     "mean_drift": w.drift.mean(), "weekly_t": t,
                     "mean_react": w.react.mean(),
                     "share_negative": (w.drift < 0).mean()}
        print(f"\n=== {name} ===")
        for k, v in out[name].items():
            print(f"  {k:>15}: {v:+.4f}" if isinstance(v, float) else f"  {k:>15}: {v}")

    r, v = out["research"], out["validation"]
    c1 = bool(r["mean_drift"] < 0 and r["weekly_t"] <= T_RESEARCH)
    c2 = bool(v["mean_drift"] < 0 and v["weekly_t"] <= T_VALIDATION)
    ok2 = c1 and c2
    shortable = bool(v["mean_drift"] <= -0.025)
    print(f"\nC1 research t<={T_RESEARCH}: {c1} (t={r['weekly_t']:+.2f})")
    print(f"C2 validation t<={T_VALIDATION}: {c2} (t={v['weekly_t']:+.2f})")
    print(f"short viable (drift <= -2.5%): {shortable} "
          f"({v['mean_drift']*100:+.2f}%)")
    print(f"\n{'HOLD (avoidance overlay; short only if viable)' if ok2 else 'REJECTED'}")

    register(Experiment(
        name="exp19_dilution_424b5",
        family="events",
        hypothesis="shelf takedowns (424B5) are followed by negative "
                   "market-adjusted drift as dilution is digested "
                   "(SEO literature: Loughran-Ritter, Corwin)",
        features=["424b5_event"],
        split="research+validation; 2023+ SEALED",
        universe="ADV63>=$2M, CIK-matched",
        rebalance_days=1, holding_days=10,
        n_observations=int(v["events"]),
        metrics={f"{s_}_{k}": (round(float(x), 4) if isinstance(x, float) else x)
                 for s_, d in out.items() for k, x in d.items()}
        | {"C1": c1, "C2": c2, "short_viable": shortable},
        outcome="HOLD" if ok2 else "FAIL",
        failure_reason="" if ok2 else
        f"failed pre-registered criterion(s): C1={c1} C2={c2}",
    ))


if __name__ == "__main__":
    main()
