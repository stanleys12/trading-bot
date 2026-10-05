"""
exp18a: Do overnight 8-K gaps continue intraday in small caps?
==============================================================

WRITTEN BEFORE qr/data/processed/edgar_8k_timestamps.parquet EXISTED
(the harvest was ~5% complete when these criteria were frozen).

HYPOTHESIS
    When material news (an 8-K) is accepted AFTER the close, the opening
    auction reprices the stock overnight — but small caps are thin, few
    arbitrageurs attend, and the literature on underreaction says the
    initial repricing is incomplete: the open-to-close return of the next
    session should CONTINUE the direction of the news gap. This is exp09's
    question conditioned on an actual information event instead of a raw
    premarket gap — the version the framework called "event-driven".

EVENTS
    8-Ks with acceptance time in (16:00 ET, 09:30 ET next session).
    Universe: follower band ADV63 $2-50M, CIK-matched to the panel.
    One event per symbol-day (multiple same-night 8-Ks collapse to one).

MEASUREMENT (daily panel only — no minute data needed)
    gap      = open(E+1)/close(E) - 1, minus SPY's same overnight gap
    outcome  = close(E+1)/open(E+1) - 1, minus SPY same intraday
    where E is the last session before acceptance, E+1 the first after.

STATISTIC (the clustering lesson)
    Weekly pooled Spearman IC of gap vs outcome; t across weeks, both
    windows. POWER: expected >=10k in-band overnight events per split ->
    weekly IC over ~250-400 weeks, se ~ 0.01-0.02; t >= 2 detects
    |IC| >= ~0.04. Under the null passes ~2.5% of the time.

PRE-REGISTERED CRITERIA — all required
    F1  research IC t >= 3.0 (positive = continuation)
    F2  validation IC t >= 2.0, same sign as research
    F3  cost arithmetic: validation top-minus-bottom gap-decile intraday
        spread must exceed 100bps (an event-day long-short at taker cost,
        2 sides x 2 legs x 25bps). Auction execution could halve this but
        exp10c left OPEN-print provenance unverified, so the taker floor
        stands.
    TEST 2023+ SEALED. Survival -> HOLD.
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
ADV_LO, ADV_HI = 2e6, 5e7
RESEARCH = ("2010-01-01", "2017-12-31")
VALIDATION = ("2018-01-01", "2022-12-31")


def main() -> None:
    ts = pd.read_parquet(PROCESSED / "edgar_8k_timestamps.parquet")
    ts["acc_et"] = (ts["acceptance_dt"].dt.tz_localize("UTC", nonexistent="NaT",
                                                       ambiguous="NaT")
                    if ts["acceptance_dt"].dt.tz is None else ts["acceptance_dt"]
                    ).dt.tz_convert("America/New_York")

    cols = ["symbol", "date", "cik", "open", "close", "adj_close",
            "dollar_volume"]
    panel = pd.read_parquet(PROCESSED / "panel_v5_full.parquet", columns=cols)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["cik_i"] = panel["cik"].astype("Int64")
    cik2sym = (panel.dropna(subset=["cik"])
               .groupby("cik_i")["symbol"].agg(lambda s: s.mode().iat[0]))
    ts["symbol"] = ts["cik"].map(cik2sym)
    ts = ts.dropna(subset=["symbol"])

    px_o = panel.pivot_table(index="date", columns="symbol", values="open",
                             aggfunc="last").sort_index()
    px_c = panel.pivot_table(index="date", columns="symbol", values="close",
                             aggfunc="last").sort_index()
    dv = panel.pivot_table(index="date", columns="symbol",
                           values="dollar_volume", aggfunc="last").sort_index()
    adv63 = dv.rolling(63, min_periods=40).mean()
    dates = px_o.index

    spy = (pd.read_parquet(PROCESSED / "etf_panel.parquet")
           .query("symbol=='SPY'").set_index("date")[["open", "close"]])
    spy.index = pd.to_datetime(spy.index)
    spy = spy.reindex(dates).ffill()
    spy_gap = (spy["open"] / spy["close"].shift(1) - 1).values
    spy_intr = (spy["close"] / spy["open"] - 1).values

    # overnight events: accepted after 16:00 ET, before 09:30 ET next day
    et = ts["acc_et"].dropna()
    ts = ts.loc[et.index]
    after_close = ts["acc_et"].dt.hour >= 16
    before_open = (ts["acc_et"].dt.hour < 9) | (
        (ts["acc_et"].dt.hour == 9) & (ts["acc_et"].dt.minute < 30))
    ts = ts[after_close | before_open].copy()
    # effective news date: filings after close belong to the NEXT session
    eff = ts["acc_et"].dt.tz_localize(None).dt.normalize() \
        + pd.to_timedelta(np.where(after_close.loc[ts.index], 1, 0), unit="D")
    ts["pos"] = np.searchsorted(dates.values, eff.values, "left")
    ts = ts[(ts.pos >= 1) & (ts.pos < len(dates))]
    ts = ts.drop_duplicates(subset=["symbol", "pos"])
    print(f"overnight 8-K events: {len(ts)}  (K={trial_count()})")

    scol = pd.Series(np.arange(px_o.shape[1]), index=px_o.columns)
    sidx = ts["symbol"].map(scol).astype(int).values
    pos = ts["pos"].values
    adv_at = adv63.values[pos - 1, sidx]
    keep = (adv_at >= ADV_LO) & (adv_at <= ADV_HI)
    ts, sidx, pos = ts[keep], sidx[keep], pos[keep]
    print(f"in $2-50M band: {len(ts)}")

    o, c = px_o.values, px_c.values
    gap = o[pos, sidx] / c[pos - 1, sidx] - 1 - spy_gap[pos]
    intr = c[pos, sidx] / o[pos, sidx] - 1 - spy_intr[pos]
    ev = pd.DataFrame({"date": dates[pos], "gap": gap, "intr": intr}).dropna()
    ev["week"] = ev["date"].dt.to_period("W")

    out = {}
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        w = ev[(ev.date >= lo) & (ev.date <= hi)]
        ics, spreads = [], []
        for _, g in w.groupby("week"):
            if len(g) < 8:
                continue
            ics.append(sstats.spearmanr(g.gap, g.intr).statistic)
            q = pd.qcut(g.gap, 5, labels=False, duplicates="drop")
            if q.nunique() >= 5:
                spreads.append(g.intr[q == q.max()].mean()
                               - g.intr[q == q.min()].mean())
        ics = pd.Series(ics).dropna()
        t = ics.mean() / (ics.std() / np.sqrt(len(ics)))
        out[name] = {"events": len(w), "weeks": len(ics),
                     "mean_ic": ics.mean(), "ic_t": t,
                     "decile_spread": float(np.nanmean(spreads))}
        print(f"{name}: {len(w)} events, {len(ics)} wks, IC {ics.mean():+.4f} "
              f"t={t:+.2f}, Q5-Q1 intraday spread "
              f"{np.nanmean(spreads)*100:+.2f}%")

    r, v = out["research"], out["validation"]
    f1 = bool(r["ic_t"] >= 3.0)
    f2 = bool(v["ic_t"] >= 2.0 and np.sign(v["mean_ic"]) == np.sign(r["mean_ic"]))
    f3 = bool(v["decile_spread"] > 0.01)
    ok = f1 and f2 and f3
    print(f"\nF1: {f1} (t={r['ic_t']:+.2f})  F2: {f2} (t={v['ic_t']:+.2f})  "
          f"F3 spread>1%: {f3} ({v['decile_spread']*100:+.2f}%)")
    print("HOLD" if ok else "REJECTED")

    register(Experiment(
        name="exp18a_overnight_8k_continuation",
        family="events",
        hypothesis="overnight 8-K news gaps in small caps continue "
                   "open-to-close (incomplete overnight repricing); exp09's "
                   "question conditioned on an actual information event",
        features=["overnight_8k_gap"],
        split="research+validation; 2023+ SEALED",
        universe="$2-50M ADV, CIK-matched",
        rebalance_days=1, holding_days=1,
        n_observations=int(v["events"]),
        metrics={f"{s}_{k}": round(float(x), 4)
                 for s, d in out.items() for k, x in d.items()}
        | {"F1": f1, "F2": f2, "F3": f3},
        outcome="HOLD" if ok else "FAIL",
        failure_reason="" if ok else
        f"failed pre-registered criterion(s): F1={f1} F2={f2} F3={f3}",
    ))


if __name__ == "__main__":
    main()
