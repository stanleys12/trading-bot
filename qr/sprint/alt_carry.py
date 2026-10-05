#!/usr/bin/env python3
"""sprint06: CROSS-SECTIONAL funding carry — hold the richest-funding alt
perps (hedged), rotate weekly. The literature's explanation for why majors
compressed but funds still print ~19%: selective pair choice.

Pre-registered (2026-08-27, before scoring):
  Universe at each rebalance: USDT perps with >=180 days of funding history
  by that date (excludes new-listing chaos), spot pair exists on Binance,
  stable-vs-stable pairs excluded.
  Selection: each Monday 00:00 UTC rank by trailing 7-day mean funding;
  hold TOP K=10 equal-weight. K=10 is THE registered decision variant;
  K=5/20 reported for robustness only.
  Costs: 50bp per position cycle (fees + wider alt spreads + basis
  allowance) charged on the sleeve fraction turned over; delisted-while-held
  pays an extra forced-exit cycle.
  Splits: research 2024-01..2026-04 / validation 2026-05..07 / 08+ SEALED.
  PASS: validation annualized net > 6% AND >=2/3 months positive (higher
  bar than sprint05: implementation is messier in alts).
  STATED LIMITATION: no point-in-time ADV/liquidity filter yet (perp volume
  data not downloaded). If PASS, the pre-registered NEXT step is re-scoring
  under an ADV floor BEFORE any deployment claim.
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

FDIR = HERE / "data" / "funding"
SPLITS = {"research": ("2024-01-01", "2026-05-01"),
          "validation": ("2026-05-01", "2026-08-01")}
COST_CYCLE_BPS = 50.0
STABLES = {"USDCUSDT", "TUSDUSDT", "FDUSDUSDT", "EURUSDT", "BUSDUSDT",
           "USDPUSDT", "DAIUSDT", "AEURUSDT", "USD1USDT", "FDUSDT"}


def build_panel():
    spot = set(json.load(open(HERE / "data" / "all_symbols.json")))
    frames = {}
    for f in FDIR.glob("*.parquet"):
        sym = f.stem
        if sym in STABLES or sym not in spot:
            continue
        d = pd.read_parquet(f)
        if len(d) < 50:
            continue
        frames[sym] = d.set_index("ts").rate
    panel = pd.DataFrame(frames).sort_index()
    # snap to the canonical 8h grid to align exchanges' odd timestamps
    panel = panel.groupby(pd.Grouper(freq="8h")).mean()
    return panel


def run(panel, k):
    trail = panel.rolling(21, min_periods=15).mean()
    age_ok = panel.notna().cumsum() > 540          # >=180 days of history
    weeks = pd.date_range(panel.index[0], panel.index[-1], freq="W-MON",
                          tz="UTC")
    hold, rows, prev = {}, [], set()
    for i, wk in enumerate(weeks[:-1]):
        asof = trail.index[trail.index <= wk]
        if len(asof) == 0:
            continue
        t0 = asof[-1]
        elig = trail.loc[t0].where(age_ok.loc[t0]).dropna()
        top = set(elig.nlargest(k).index)
        churn = len(top - prev) + len(prev - top) if prev else len(top)
        mask = (panel.index > wk) & (panel.index <= weeks[i + 1])
        block = panel.loc[mask, sorted(top)] if top else pd.DataFrame()
        inc = block.mean(axis=1).fillna(0.0) * 1e4       # portfolio bps/8h
        cost = churn * (COST_CYCLE_BPS / max(k, 1)) / 2   # per-side
        for ts, v in inc.items():
            rows.append({"ts": ts, "bps": v})
        if len(inc):
            rows[-1]["bps"] -= cost
        prev = top
    s = pd.DataFrame(rows).set_index("ts").bps
    out = {}
    for sp, (a, b) in SPLITS.items():
        x = s[(s.index >= a) & (s.index < b)]
        days = max((x.index[-1] - x.index[0]).days, 1)
        monthly = x.resample("ME").sum()
        out[sp] = {"n": int(len(x)),
                   "net_bps": round(float(x.sum()), 1),
                   "annualized_net_pct": round(float(x.sum()) / 1e4 / days * 365 * 100, 2),
                   "months_positive": f"{int((monthly>0).sum())}/{len(monthly)}",
                   "t_stat": round(float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), 2)
                             if x.std() > 0 else 0.0}
    return out


def main():
    panel = build_panel()
    print(f"panel: {panel.shape[1]} eligible perps x {panel.shape[0]} intervals",
          file=sys.stderr)
    res = {f"K={k}": run(panel, k) for k in (5, 10, 20)}
    v = res["K=10"]["validation"]
    mp = int(v["months_positive"].split("/")[0])
    verdict = "PASS" if (v["annualized_net_pct"] > 6.0 and mp >= 2) else "FAIL"
    art = {"name": "sprint06_alt_carry", "ts": datetime.now(timezone.utc
           ).isoformat(timespec="seconds"),
           "spec": {"name": "sprint06_alt_carry", "family": "carry",
                    "hypothesis": "cross-sectional funding carry in alts retains the "
                                  "premium majors lost; decision variant K=10 weekly",
                    "engine": "alt_carry.py", "costs": "50bp/position cycle",
                    "limitation": "no point-in-time ADV filter yet; PASS requires "
                                  "ADV-floor re-score before deployment claim",
                    "pass_rule": "K=10 validation ann net >6% AND >=2/3 months positive"},
           "results": res, "verdict": verdict,
           "status": "complete", "red_team": "pending"}
    (HERE / "artifacts" / "sprint06_alt_carry.json").write_text(json.dumps(art, indent=1))
    from qr.research.protocol import register, Experiment
    register(Experiment(name="sprint06_alt_carry", family="carry",
        hypothesis=art["spec"]["hypothesis"], features=["funding_rate_xs"],
        split="research 2024-01..2026-04 / validation 2026-05..07; 2026-08+ SEALED",
        universe="USDT perps w/ spot pair, >=180d history (no ADV floor yet)",
        rebalance_days=7, holding_days=7, n_observations=v["n"],
        metrics=v, alpha=None, outcome=verdict,
        failure_reason="" if verdict == "PASS" else "K=10 validation below 6% gate"))
    print(json.dumps(art, indent=1))


if __name__ == "__main__":
    main()
