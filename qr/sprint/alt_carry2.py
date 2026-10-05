#!/usr/bin/env python3
"""sprint07: alt carry, persistence + hysteresis variant. FINAL carry variant
this week (pre-committed 2026-08-27; a third redesign would be family-mining).

Changes vs sprint06, each motivated by its failure diagnosis:
  - Selection on trailing 30-DAY mean funding (persistent payers, not spikes)
  - MONTHLY rebalance (first Monday), not weekly
  - Hysteresis: holdings stay until they drop below rank 2K (churn control)
Pre-registered decision variant: K=10. Costs 50bp/cycle as sprint06.
PASS: validation ann net > 6% AND >=2/3 months positive.
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from qr.sprint.alt_carry import build_panel, SPLITS, COST_CYCLE_BPS  # noqa: E402


def run(panel, k):
    trail = panel.rolling(90, min_periods=60).mean()          # 30d of 8h bars
    age_ok = panel.notna().cumsum() > 540
    rebs = pd.date_range(panel.index[0], panel.index[-1], freq="MS", tz="UTC")
    rows, prev = [], set()
    for i, rb in enumerate(rebs[:-1]):
        asof = trail.index[trail.index <= rb]
        if len(asof) == 0:
            continue
        t0 = asof[-1]
        elig = trail.loc[t0].where(age_ok.loc[t0]).dropna()
        ranked = elig.sort_values(ascending=False)
        keep = {s for s in prev if s in ranked.index[:2 * k]}
        adds = [s for s in ranked.index if s not in keep][:max(k - len(keep), 0)]
        top = keep | set(adds)
        churn = len(top - prev) + len(prev - top) if prev else len(top)
        mask = (panel.index > rb) & (panel.index <= rebs[i + 1])
        block = panel.loc[mask, sorted(top)] if top else pd.DataFrame()
        inc = block.mean(axis=1).fillna(0.0) * 1e4
        cost = churn * (COST_CYCLE_BPS / max(k, 1)) / 2
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
        out[sp] = {"n": int(len(x)), "net_bps": round(float(x.sum()), 1),
                   "annualized_net_pct": round(float(x.sum())/1e4/days*365*100, 2),
                   "months_positive": f"{int((monthly>0).sum())}/{len(monthly)}",
                   "t_stat": round(float(x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))), 2)
                             if x.std() > 0 else 0.0}
    return out


def main():
    panel = build_panel()
    res = {f"K={k}": run(panel, k) for k in (5, 10, 20)}
    v = res["K=10"]["validation"]
    mp = int(v["months_positive"].split("/")[0])
    verdict = "PASS" if (v["annualized_net_pct"] > 6.0 and mp >= 2) else "FAIL"
    art = {"name": "sprint07_alt_carry_persistent", "ts": datetime.now(
               timezone.utc).isoformat(timespec="seconds"),
           "spec": {"name": "sprint07_alt_carry_persistent", "family": "carry",
                    "hypothesis": "PERSISTENT high-funding alts (30d selection, monthly "
                                  "rotation, hysteresis) retain carry after churn costs",
                    "engine": "alt_carry2.py", "costs": "50bp/position cycle",
                    "limitation": "no ADV floor yet; LAST carry variant this week "
                                  "(pre-committed against family-mining)",
                    "pass_rule": "K=10 validation ann net >6% AND >=2/3 months positive"},
           "results": res, "verdict": verdict, "status": "complete",
           "red_team": "pending"}
    (HERE / "artifacts" / "sprint07_alt_carry_persistent.json").write_text(
        json.dumps(art, indent=1))
    from qr.research.protocol import register, Experiment
    register(Experiment(name="sprint07_alt_carry_persistent", family="carry",
        hypothesis=art["spec"]["hypothesis"], features=["funding_rate_xs_30d"],
        split="research 2024-01..2026-04 / validation 2026-05..07; 2026-08+ SEALED",
        universe="USDT perps w/ spot, >=180d history, hysteresis 2K",
        rebalance_days=30, holding_days=30, n_observations=v["n"],
        metrics=v, alpha=None, outcome=verdict,
        failure_reason="" if verdict == "PASS" else "K=10 validation below 6% gate"))
    print(json.dumps(art, indent=1))


if __name__ == "__main__":
    main()
