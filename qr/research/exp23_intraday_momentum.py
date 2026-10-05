"""
exp23: Market intraday momentum — does the day's first leg predict its last?
============================================================================

CRITERIA FIXED BEFORE SCORING (the minute panel was being fetched, unseen,
when this was written).

HYPOTHESIS
    The market's return from the open through 15:30 predicts the final
    half hour in the SAME direction (Gao-Han-Li-Zhou 2018 JFE, replicated
    internationally). Mechanisms: late-day rebalancing flows from
    leveraged/target-date funds that must trade the day's move, and
    informed traders deferring execution to the liquid close. This is a
    NEW family here: intraday return structure, using the minute history
    this project only wired up this week.

WHY THE COST WALL IS THIN HERE
    The trade is: at 15:30, go long or short SPY in the direction of the
    day so far; exit at the close. SPY quoted half-spread is ~0.4bp and
    the exit is the verified closing auction (exp10c: 0.000bp median).
    Round trip charged at 1bp/day flat, pre-registered.

DATA
    SPY 1-minute bars, SIP, 2016-2022 (free tier starts 2016). RTH only.
    SPY ONLY — no QQQ/IWM variants, computed or peeked at, so this is
    exactly one trial of K.

SPLITS (minute data begins 2016; the 2023+ equity seal is respected)
    research    2016-01-01 .. 2019-12-31   (~1,000 days)
    validation  2020-01-01 .. 2022-12-31   (~750 days, includes the 2020
                vol regime where the published effect was strongest)

POWER (before scoring)
    Final-half-hour daily vol ~25-50bp. Published gross predictability
    ~3-5bp/day. Research: t = 4bp/(30bp/sqrt(1000)) ~ 4.2 at published
    size, ~2.1 at half size -> F1's 3.0 bar demands the effect near full
    published strength. Validation: powered for ~3.6bp at 2020-22 vol.
    Under the null each criterion passes ~0.1-2.5% of the time. Can pass,
    can fail.

PRE-REGISTERED CRITERIA — all required
    F1  research: mean gross strategy return > 0, t >= 3.0
    F2  validation: same sign, t >= 2.0
    F3  validation net mean (1bp/day cost) > 0
    Survival -> HOLD (a fourth paper book would be the next adjudicator,
    trading at 12:30 PT with MOC exits). 2023+ SEALED.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.execution.live_paper import _headers  # noqa: E402
from qr.research.protocol import Experiment, register, trial_count  # noqa: E402

INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim"
CACHE = INTERIM / "spy_minute_2016_2022.parquet"

COST_PER_DAY = 1e-4
RESEARCH = ("2016-01-01", "2019-12-31")
VALIDATION = ("2020-01-01", "2022-12-31")


def fetch_minutes() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    rows, token = [], None
    while True:
        p = {"symbols": "SPY", "timeframe": "1Min", "start": "2016-01-01",
             "end": "2022-12-31", "feed": "sip", "limit": 10000,
             "adjustment": "raw"}
        if token:
            p["page_token"] = token
        r = requests.get("https://data.alpaca.markets/v2/stocks/bars",
                         headers=_headers(), params=p, timeout=60)
        r.raise_for_status()
        j = r.json()
        for b in (j.get("bars") or {}).get("SPY", []):
            rows.append({"t": b["t"], "o": b["o"], "c": b["c"]})
        token = j.get("next_page_token")
        if not token:
            break
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
    df = df.drop(columns=["t"])
    df.to_parquet(CACHE)
    return df


def daily_legs(m: pd.DataFrame) -> pd.DataFrame:
    m = m[(m.ts.dt.time >= pd.Timestamp("09:30").time())
          & (m.ts.dt.time <= pd.Timestamp("16:00").time())].copy()
    m["date"] = m.ts.dt.date
    rows = []
    for d, g in m.groupby("date"):
        g = g.sort_values("ts")
        open_px = g.iloc[0]["o"]
        upto_1530 = g[g.ts.dt.time <= pd.Timestamp("15:30").time()]
        if len(upto_1530) < 300:          # half days / bad tapes excluded
            continue
        p1530 = upto_1530.iloc[-1]["c"]
        close_px = g.iloc[-1]["c"]
        rows.append({"date": pd.Timestamp(d), "r_first": p1530 / open_px - 1,
                     "r_last": close_px / p1530 - 1})
    return pd.DataFrame(rows).set_index("date")


def main() -> None:
    m = fetch_minutes()
    d = daily_legs(m)
    d["strat"] = np.sign(d["r_first"]) * d["r_last"]
    d["strat_net"] = d["strat"] - COST_PER_DAY
    print(f"days: {len(d)}  (K={trial_count()})")

    out = {}
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        w = d[(d.index >= lo) & (d.index <= hi)]
        t = w.strat.mean() / (w.strat.std() / np.sqrt(len(w)))
        tn = w.strat_net.mean() / (w.strat_net.std() / np.sqrt(len(w)))
        hit = (np.sign(w.r_first) == np.sign(w.r_last)).mean()
        out[name] = {"days": len(w), "gross_bp_day": w.strat.mean() * 1e4,
                     "t": t, "net_bp_day": w.strat_net.mean() * 1e4,
                     "net_t": tn, "hit_rate": hit,
                     "ann_net_pct": w.strat_net.mean() * 252 * 100}
        print(f"{name}: {len(w)} days | gross {w.strat.mean()*1e4:+.2f}bp/day "
              f"t={t:+.2f} | net {w.strat_net.mean()*1e4:+.2f}bp/day "
              f"({w.strat_net.mean()*252*100:+.1f}%/yr) | hit {hit:.1%}")

    r, v = out["research"], out["validation"]
    f1 = bool(r["gross_bp_day"] > 0 and r["t"] >= 3.0)
    f2 = bool(v["t"] >= 2.0 and np.sign(v["gross_bp_day"]) == np.sign(r["gross_bp_day"]))
    f3 = bool(v["net_bp_day"] > 0)
    ok = f1 and f2 and f3
    print(f"\nF1 research t>=3: {f1} (t={r['t']:+.2f})")
    print(f"F2 validation t>=2, same sign: {f2} (t={v['t']:+.2f})")
    print(f"F3 validation net > 0: {f3} ({v['net_bp_day']:+.2f}bp/day)")
    print("HOLD" if ok else "REJECTED")

    register(Experiment(
        name="exp23_intraday_momentum_spy",
        family="intraday_timing",
        hypothesis="open-to-15:30 return predicts the final half hour in the "
                   "same direction (Gao-Han-Li-Zhou 2018): late-day "
                   "rebalancing flows must trade the day's move",
        features=["r_open_1530"],
        split="research 2016-2019 / validation 2020-2022 (SIP minute data "
              "begins 2016); 2023+ SEALED",
        universe="SPY",
        rebalance_days=1, holding_days=1,
        n_observations=int(v["days"]),
        metrics={f"{s}_{k}": round(float(x), 4)
                 for s, dd in out.items() for k, x in dd.items()}
        | {"F1": f1, "F2": f2, "F3": f3},
        outcome="HOLD" if ok else "FAIL",
        failure_reason="" if ok else
        f"failed pre-registered criterion(s): F1={f1} F2={f2} F3={f3}",
    ))


if __name__ == "__main__":
    main()
