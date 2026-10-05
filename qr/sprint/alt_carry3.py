#!/usr/bin/env python3
"""sprint08: the DECISIVE alt-carry re-score. Adjudicates sprint07's
challenged PASS with three fixes, pre-registered before scoring:

 FIX 1 (implementability): a name is eligible at a rebalance only if its
   SPOT pair actually traded in the prior 7 days (spot daily bar exists) —
   point-in-time, ends the BTCST/MATIC ghosts.
 FIX 2 (liquidity): 30d median ADV floors at rebalance: spot >= $2M/day
   AND perp >= $5M/day.
 FIX 3 (accounting): income = sum(alive rates)/K, NOT mean over alive
   names — a dead/halted holding contributes zero instead of silently
   renormalizing the basket (suspected source of sprint07 inflation), and
   pays a forced-exit cycle when it dies.

Decomposition reported: A=fix3 only, B=fix3+fix1, C=all fixes (DECISION).
Selection/hysteresis/K=10/costs/splits/gates unchanged from sprint07:
PASS = variant C, K=10: validation ann net > 6% AND >= 2/3 months positive.
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from qr.sprint.alt_carry import build_panel, SPLITS, COST_CYCLE_BPS  # noqa: E402


def daily_table(dirname):
    out = {}
    for f in (HERE / "data" / dirname).glob("*.parquet"):
        d = pd.read_parquet(f).drop_duplicates("ts", keep="last")
        out[f.stem] = d.set_index("ts").quote_vol
    return pd.DataFrame(out).sort_index()


def run(panel, spot_adv, perp_adv, k=10, use_trade=True, use_adv=True):
    trail = panel.rolling(90, min_periods=60).mean()
    age_ok = panel.notna().cumsum() > 540
    rebs = pd.date_range(panel.index[0], panel.index[-1], freq="MS", tz="UTC")
    rows, prev = [], set()
    for i, rb in enumerate(rebs[:-1]):
        asof = trail.index[trail.index <= rb]
        if len(asof) == 0:
            continue
        t0 = asof[-1]
        elig = trail.loc[t0].where(age_ok.loc[t0]).dropna()
        if use_trade and len(spot_adv):
            week = spot_adv[(spot_adv.index < rb) &
                            (spot_adv.index >= rb - pd.Timedelta(days=7))]
            alive = set(week.columns[week.notna().any()])
            elig = elig[elig.index.isin(alive)]
        if use_adv and len(spot_adv):
            s30 = spot_adv[spot_adv.index < rb].tail(30).median()
            p30 = perp_adv[perp_adv.index < rb].tail(30).median()
            ok = set(s30[s30 >= 2e6].index) & set(p30[p30 >= 5e6].index)
            elig = elig[elig.index.isin(ok)]
        ranked = elig.sort_values(ascending=False)
        keep = {s for s in prev if s in ranked.index[:2 * k]}
        adds = [s for s in ranked.index if s not in keep][:max(k - len(keep), 0)]
        top = keep | set(adds)
        churn = len(top - prev) + len(prev - top) if prev else len(top)
        mask = (panel.index > rb) & (panel.index <= rebs[i + 1])
        block = panel.loc[mask, sorted(top)] if top else pd.DataFrame()
        # FIX 3: fixed capital per slot; dead names pay 0, plus forced exit
        inc = block.sum(axis=1, min_count=1).fillna(0.0) / max(k, 1) * 1e4
        died = int(block.notna().any().eq(False).sum()) if len(block) else 0
        cost = (churn * (COST_CYCLE_BPS / max(k, 1)) / 2
                + died * (COST_CYCLE_BPS / max(k, 1)))
        for ts, v in inc.items():
            rows.append({"ts": ts, "bps": v})
        if len(inc):
            rows[-1]["bps"] -= cost
        prev = {s for s in top if s in block.columns and block[s].notna().any()} \
            if len(block) else top
    s = pd.DataFrame(rows).set_index("ts").bps
    out = {}
    for sp, (a, b) in SPLITS.items():
        x = s[(s.index >= a) & (s.index < b)]
        if not len(x):
            out[sp] = {"n": 0}
            continue
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
    spot_adv = daily_table("spot_daily")
    perp_adv = daily_table("perp_daily")
    print(f"panel {panel.shape[1]} perps | spot_daily {spot_adv.shape[1]} | "
          f"perp_daily {perp_adv.shape[1]}", file=sys.stderr)
    res = {
        "A_fix_accounting_only": run(panel, spot_adv, perp_adv, use_trade=False, use_adv=False),
        "B_plus_tradeability":   run(panel, spot_adv, perp_adv, use_trade=True, use_adv=False),
        "C_full_filters_DECISION": run(panel, spot_adv, perp_adv, use_trade=True, use_adv=True),
    }
    v = res["C_full_filters_DECISION"]["validation"]
    mp = int(v.get("months_positive", "0/0").split("/")[0]) if v.get("n") else 0
    verdict = "PASS" if (v.get("annualized_net_pct", -99) > 6.0 and mp >= 2) else "FAIL"
    art = {"name": "sprint08_alt_carry_rescore",
           "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "spec": {"name": "sprint08_alt_carry_rescore", "family": "carry",
                    "hypothesis": "sprint07's challenged carry survives point-in-time spot "
                                  "tradeability + ADV floors + honest accounting",
                    "engine": "alt_carry3.py",
                    "adjudicates": "sprint07_alt_carry_persistent (PASS-CHALLENGED)",
                    "pass_rule": "variant C K=10: validation ann net >6% AND >=2/3 months positive"},
           "results": res, "verdict": verdict, "status": "complete",
           "red_team": "pending"}
    (HERE / "artifacts" / "sprint08_alt_carry_rescore.json").write_text(
        json.dumps(art, indent=1))
    from qr.research.protocol import register, Experiment
    register(Experiment(name="sprint08_alt_carry_rescore", family="carry",
        hypothesis=art["spec"]["hypothesis"], features=["funding_rate_xs_30d"],
        split="research 2024-01..2026-04 / validation 2026-05..07; 2026-08+ SEALED",
        universe="USDT perps: PIT spot-traded, spot ADV>=$2M, perp ADV>=$5M",
        rebalance_days=30, holding_days=30, n_observations=v.get("n", 0),
        metrics=v, alpha=None, outcome=verdict,
        failure_reason="" if verdict == "PASS" else "decision variant below gate under honest filters"))
    print(json.dumps(art, indent=1))


if __name__ == "__main__":
    main()
