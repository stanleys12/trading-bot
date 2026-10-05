"""
exp20b: Does EXCLUDING the wildest coins improve a crypto basket?
=================================================================

ORIGIN, STATED HONESTLY
    DATA-INSPIRED. exp20's LOWVOL factor showed the strongest rank
    statistic in this project's history (IC +0.087/+0.088, val t=5.5) but
    a zero top-minus-bottom spread. The refinement question — is the whole
    effect "the top-vol quintile crashes"? — was formed AFTER seeing that
    quintile table. It therefore pays a raised validation bar (t >= 2.5
    instead of 2.0) and is flagged data-inspired in the registry.

QUESTION
    Portfolio A: equal-weight all eligible coins, weekly.
    Portfolio B: same, EXCLUDING the top-vol quintile (trailing 30d vol).
    Does B beat A — reliably, in both windows?

USE CASE
    An overlay for anyone holding a broad crypto basket. Not a standalone
    strategy; a filter. Costs are near-identical for both portfolios
    (both rebalance weekly across ~100+ names), so the COMPARISON is
    approximately cost-neutral and is scored gross.

PRE-REGISTERED CRITERIA — all required
    C1  research:   weekly mean of (B - A) > 0 with t >= 2.0
    C2  validation: weekly mean of (B - A) > 0 with t >= 2.5 (raised bar)
    C3  B's max drawdown no deeper than A's in BOTH windows
    2024+ SEALED. Survival -> HOLD only.

POWER (before scoring)
    B and A overlap ~80%, so the difference series is low-variance
    (~0.5-1.5%/wk). 104 validation weeks -> se ~ 0.05-0.15%/wk; t >= 2.5
    detects a ~0.15-0.4%/wk improvement. If Q5 drags EW by what exp20's
    IC implies, the effect lands inside that range. Can pass, can fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register, trial_count  # noqa: E402
from qr.research.exp16_crypto_xs_momentum import (  # noqa: E402
    load_panel, ADV_MIN, MIN_HISTORY, SPLITS)


def main() -> None:
    p = load_panel()
    px = p.pivot_table(index="date", columns="symbol", values="close",
                       aggfunc="last").sort_index()
    dollar = p.pivot_table(index="date", columns="symbol",
                           values="dollar_volume", aggfunc="last").sort_index()
    ret = px.pct_change()
    vol30 = ret.rolling(30, min_periods=20).std()
    adv30 = dollar.rolling(30, min_periods=20).median()
    history = px.notna().cumsum()
    dates = px.index

    rows = []
    for d in dates[::7]:
        i = dates.get_loc(d)
        if i < 35 or i + 8 >= len(dates):
            continue
        elig = (history.iloc[i] >= MIN_HISTORY) & (adv30.iloc[i] >= ADV_MIN)
        v = vol30.iloc[i][elig].dropna()
        if len(v) < 25:
            continue
        fwd = (px.iloc[i + 8] / px.iloc[i + 1] - 1)[v.index]
        q5 = v >= v.quantile(0.8)
        rows.append({"date": d, "A": fwd.mean(), "B": fwd[~q5].mean(),
                     "n": len(v)})
    wk = pd.DataFrame(rows).set_index("date")
    wk["diff"] = wk.B - wk.A
    print(f"weeks scored: {len(wk)}  (K={trial_count()})")

    out, dd_ok = {}, {}
    for split, (lo, hi) in SPLITS.items():
        w = wk[(wk.index >= lo) & (wk.index <= hi)]
        t = w["diff"].mean() / (w["diff"].std() / np.sqrt(len(w)))
        eq = {c: (1 + w[c]).cumprod() for c in ["A", "B"]}
        dd = {c: float((eq[c] / eq[c].cummax() - 1).min()) for c in ["A", "B"]}
        out[split] = {"weeks": len(w), "A_wk": w.A.mean(), "B_wk": w.B.mean(),
                      "diff_wk": w["diff"].mean(), "diff_t": t,
                      "A_dd": dd["A"], "B_dd": dd["B"]}
        dd_ok[split] = bool(dd["B"] >= dd["A"])
        print(f"\n=== {split} ===")
        print(f"  A (all): {w.A.mean()*100:+.3f}%/wk  maxdd {dd['A']:+.2f}")
        print(f"  B (ex top-vol Q): {w.B.mean()*100:+.3f}%/wk  maxdd {dd['B']:+.2f}")
        print(f"  diff: {w['diff'].mean()*100:+.3f}%/wk  t={t:+.2f}")

    r, v = out["research"], out["validation"]
    c1 = bool(r["diff_wk"] > 0 and r["diff_t"] >= 2.0)
    c2 = bool(v["diff_wk"] > 0 and v["diff_t"] >= 2.5)
    c3 = bool(dd_ok["research"] and dd_ok["validation"])
    ok = c1 and c2 and c3
    print(f"\nC1 research diff t>=2.0: {c1} (t={r['diff_t']:+.2f})")
    print(f"C2 validation diff t>=2.5: {c2} (t={v['diff_t']:+.2f})")
    print(f"C3 drawdown not worse, both windows: {c3}")
    print(f"\n{'HOLD (overlay, data-inspired)' if ok else 'REJECTED'}")

    register(Experiment(
        name="exp20b_crypto_vol_exclusion",
        family="crypto",
        hypothesis="DATA-INSPIRED refinement of exp20 LOWVOL: excluding the "
                   "top-vol quintile improves a broad crypto basket (the IC "
                   "was real; is the money all in avoiding Q5?)",
        features=["vol30_exclusion"],
        split="research 2017-2021 / validation 2022-2023; 2024+ SEALED",
        universe="Binance USDT spot incl. delisted, ADV>=$1M",
        rebalance_days=7, holding_days=7,
        n_observations=int(v["weeks"]),
        metrics={f"{s}_{k}": round(float(x), 4)
                 for s, d in out.items() for k, x in d.items()}
        | {"C1": c1, "C2": c2, "C3": c3, "data_inspired": True},
        outcome="HOLD" if ok else "FAIL",
        failure_reason="" if ok else
        f"failed pre-registered criterion(s): C1={c1} C2={c2} C3={c3}",
    ))


if __name__ == "__main__":
    main()
