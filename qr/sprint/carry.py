#!/usr/bin/env python3
"""Experiment: delta-neutral funding-rate carry (long spot + short perp).

THE MECHANISM IS NOT PREDICTION. Perpetual futures pay funding from longs to
shorts when the rate is positive (usually, because leverage demand is long-
biased). A hedged position (long spot, short perp) collects that flow with
~zero price exposure. Documented: 8-20% APY across cycles, positive every
month of 2025 in one published record; compressed to high single digits 2026.

Pre-registered (2026-08-27, before scoring):
  Splits: research 2024-01..2026-04 / validation 2026-05..07 / 2026-08+ SEALED.
  Variants: STATIC (always on) and FILTERED (on only when trailing 7-day mean
  funding > 0; re-entry costs charged per flip).
  Costs: 20bp round-trip per entry+exit cycle (4 legs, taker), plus a 10bp
  one-time basis haircut per cycle (entry/exit basis mismatch allowance).
  Basis mark-to-market noise between settlements is NOT modelled and is
  stated as a limitation in the artifact (funding mechanism reverts perp to
  spot; held-to-convergence assumption).
  PASS = validation annualized net > 4% AND >=2 of 3 validation months
  positive, for either variant. Power: ~276 8h observations in validation;
  the income stream is directly observed, not inferred.
"""
import io, json, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

SPLITS = {"research": ("2024-01-01", "2026-05-01"),
          "validation": ("2026-05-01", "2026-08-01")}
COST_CYCLE_BPS = 30.0        # 20bp fees + 10bp basis allowance per on/off cycle


def funding(symbol):
    out = HERE / "data" / "funding" / f"{symbol}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    rows = []
    for ym in pd.period_range("2024-01", "2026-07", freq="M").strftime("%Y-%m"):
        url = (f"https://data.binance.vision/data/futures/um/monthly/"
               f"fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip")
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
        rows.append(df)
    f = pd.concat(rows)
    f.columns = ["ts", "interval_h", "rate"]
    f = f[pd.to_numeric(f.ts, errors="coerce").notna()]
    f["ts"] = pd.to_datetime(pd.to_numeric(f.ts), unit="ms", utc=True)
    f["rate"] = pd.to_numeric(f.rate)
    f = f.sort_values("ts").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out)
    return f


def evaluate(f, s0, s1, filtered):
    d = f[(f.ts >= s0) & (f.ts < s1)].copy()
    if filtered:
        d["on"] = d.rate.rolling(21).mean().shift(1) > 0    # trailing 7d (21x8h)
        d["on"] = d["on"].fillna(False)
    else:
        d["on"] = True
    d["income_bps"] = d.rate.where(d.on, 0.0) * 1e4
    flips = int((d.on != d.on.shift(1).fillna(False)).sum())
    cost_bps = (flips / 2 + (1 if not filtered else 0)) * COST_CYCLE_BPS
    total_bps = d.income_bps.sum() - cost_bps
    days = max((d.ts.iloc[-1] - d.ts.iloc[0]).days, 1)
    ann = total_bps / 1e4 / days * 365 * 100
    monthly = d.set_index("ts").income_bps.resample("ME").sum()
    x = d.income_bps[d.on] if filtered else d.income_bps
    t = float(x.mean() / (x.std(ddof=1) / len(x) ** 0.5)) if len(x) > 2 and x.std() > 0 else 0.0
    return {"n_intervals": int(len(d)), "on_pct": round(100 * d.on.mean(), 1),
            "gross_bps": round(float(d.income_bps.sum()), 1),
            "cost_bps": round(cost_bps, 1), "net_bps": round(float(total_bps), 1),
            "annualized_net_pct": round(ann, 2),
            "months_positive": f"{int((monthly > 0).sum())}/{len(monthly)}",
            "t_stat_per_interval": round(t, 2), "cycles": flips // 2}


def main():
    res = {}
    for sym in ["BTCUSDT", "ETHUSDT"]:
        f = funding(sym)
        res[sym] = {}
        for variant in ["static", "filtered"]:
            res[sym][variant] = {
                sp: evaluate(f, *SPLITS[sp], filtered=(variant == "filtered"))
                for sp in SPLITS}
    # verdict per pre-registration
    passing = []
    for sym, vs in res.items():
        for var, sp in vs.items():
            v = sp["validation"]
            mp = int(v["months_positive"].split("/")[0])
            if v["annualized_net_pct"] > 4.0 and mp >= 2:
                passing.append(f"{sym}:{var}")
    verdict = "PASS" if passing else "FAIL"
    art = {"name": "sprint05_funding_carry", "ts": datetime.now(timezone.utc
           ).isoformat(timespec="seconds"),
           "spec": {"name": "sprint05_funding_carry", "family": "carry",
                    "hypothesis": "delta-neutral funding carry earns documented 8-20%/yr; "
                                  "mechanism is a payment flow, not prediction",
                    "engine": "carry.py (not the signal factory)",
                    "costs": f"{COST_CYCLE_BPS}bp per cycle incl. basis allowance",
                    "limitation": "intra-period basis MTM not modelled (held-to-convergence)",
                    "pass_rule": "validation ann net >4% AND >=2/3 months positive"},
           "results": res, "passing_variants": passing, "verdict": verdict,
           "status": "complete", "red_team": "pending"}
    (HERE / "artifacts" / "sprint05_funding_carry.json").write_text(
        json.dumps(art, indent=1))
    from qr.research.protocol import register, Experiment
    v = res["BTCUSDT"]["static"]["validation"]
    register(Experiment(
        name="sprint05_funding_carry", family="carry",
        hypothesis=art["spec"]["hypothesis"], features=["funding_rate"],
        split="research 2024-01..2026-04 / validation 2026-05..07; 2026-08+ SEALED",
        universe="BTCUSDT,ETHUSDT perp funding vs spot",
        rebalance_days=0, holding_days=999,
        n_observations=v["n_intervals"], metrics=v, alpha=None,
        outcome=verdict,
        failure_reason="" if verdict == "PASS" else "validation carry below gate"))
    print(json.dumps(art, indent=1))


if __name__ == "__main__":
    main()
