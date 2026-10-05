"""
exp28: The near-miss composite — is per-signal strictness the bottleneck?
=========================================================================

ORIGIN, STATED HONESTLY
    DATA-INSPIRED by construction and labelled so: the two components are
    the two effects that showed stable, sign-consistent alpha in BOTH
    windows without individual significance (low volatility: +4.65%/+4.82%,
    t=2.81/1.30; low net issuance: +4.56%/+4.97%, t=2.29/0.95). They were
    selected BECAUSE of those seen results, so research and validation
    evidence is contaminated by selection and cannot promote anything.
    The question this trial answers is structural, not statistical:

        Real funds stack individually-unprovable signals; we have only
        ever tested signals alone. Does our strongest available composite
        clear the protocol's own pre-existing promotion gate
        (validation Sharpe >= 0.8 AND alpha_t >= 2.0)?

    IF the gate passes, open_test_set() runs — the first unsealing in
    project history, scored on 2023-2026 data no experiment has touched.
    IF the gate refuses, that refusal IS the answer to "are we too
    strict?": even the best composite of near-misses cannot earn a look,
    meaning signal size, not strictness, is the binding constraint.

SPEC (both components at their frozen definitions; nothing re-tuned)
    Universe    ADV63 >= $50M common stocks
    A: low-vol  lowest quintile trailing 63d vol, EW, quarterly
    B: low-NSI  lowest quintile net share issuance (shares/shares_252d-1),
                EW, quarterly
    Composite   50/50 daily blend, long-only, no leverage
    Alpha       daily regression vs SPY, Newey-West(10) t, per window
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import (  # noqa: E402
    Experiment, ProtocolViolation, freeze, open_test_set, register,
    trial_count)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
ADV_MIN = 5e7
REBAL = 63
RESEARCH = ("2010-01-01", "2017-12-31")
VALIDATION = ("2018-01-01", "2022-12-31")


def hac_alpha(port: pd.Series, mkt: pd.Series, lags: int = 10):
    j = pd.DataFrame({"p": port, "m": mkt}).dropna()
    x, y = j.m.values, j.p.values
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ b
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * u[:, None]).T @ (X * u[:, None])
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        G = (X[k:] * u[k:, None]).T @ (X[:-k] * u[:-k, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return b[0] * 252, b[0] / np.sqrt(V[0, 0]), b[1]


def main() -> None:
    cols = ["symbol", "date", "adj_close", "dollar_volume", "shares"]
    p = pd.read_parquet(PROCESSED / "panel_v5_full.parquet", columns=cols)
    p["date"] = pd.to_datetime(p["date"])
    adj = p.pivot_table(index="date", columns="symbol", values="adj_close",
                        aggfunc="last").sort_index()
    dv = p.pivot_table(index="date", columns="symbol", values="dollar_volume",
                       aggfunc="last").sort_index()
    sh = p.pivot_table(index="date", columns="symbol", values="shares",
                       aggfunc="last").sort_index()
    r = adj.pct_change()
    adv63 = dv.rolling(63, min_periods=40).mean()
    vol63 = r.rolling(63, min_periods=40).std()
    nsi = sh / sh.shift(252) - 1
    dates = adj.index

    spy = (pd.read_parquet(PROCESSED / "etf_panel.parquet")
           .query("symbol=='SPY'").set_index("date")["adj_close"])
    spy.index = pd.to_datetime(spy.index)
    spy_r = spy.reindex(dates).ffill().pct_change()

    ports = {"lowvol": pd.Series(0.0, index=dates),
             "lownsi": pd.Series(0.0, index=dates)}
    hold: dict[str, list] = {"lowvol": [], "lownsi": []}
    for i in range(280, len(dates)):
        if (i - 280) % REBAL == 0:
            elig = adv63.iloc[i - 1] >= ADV_MIN
            v = vol63.iloc[i - 1][elig].dropna()
            hold["lowvol"] = list(v[v <= v.quantile(0.2)].index)
            n = nsi.iloc[i - 1][elig].dropna()
            hold["lownsi"] = list(n[n <= n.quantile(0.2)].index)
        for k in ports:
            if hold[k]:
                ports[k].iloc[i] = r.iloc[i][hold[k]].mean()

    comp = 0.5 * ports["lowvol"] + 0.5 * ports["lownsi"]
    print(f"components built  (K={trial_count()})")

    out = {}
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        m = (dates >= lo) & (dates <= hi)
        row = {}
        for k, s in [("lowvol", ports["lowvol"]), ("lownsi", ports["lownsi"]),
                     ("comp", comp)]:
            a, t, beta = hac_alpha(s[m], spy_r[m])
            row[k] = {"alpha": a, "t": t, "beta": beta}
        sr = comp[m].mean() / comp[m].std() * np.sqrt(252)
        row["comp_sharpe"] = sr
        out[name] = row
        print(f"\n=== {name} ===  composite Sharpe {sr:.2f}")
        for k in ["lowvol", "lownsi", "comp"]:
            d = row[k]
            print(f"  {k:>7}: alpha {d['alpha']*100:+.2f}%/yr  "
                  f"t={d['t']:+.2f}  beta {d['beta']:.2f}")

    v = out["validation"]
    gate_sharpe = float(v["comp_sharpe"])
    gate_alpha_t = float(v["comp"]["t"])
    spec = {"components": ["lowvol_q20_63d", "lownsi_q20_252d"],
            "universe": "ADV63>=$50M", "rebalance_days": REBAL,
            "weights": [0.5, 0.5], "long_only": True}
    print(f"\nPROTOCOL GATE: validation Sharpe {gate_sharpe:.2f} (needs >=0.8), "
          f"alpha_t {gate_alpha_t:.2f} (needs >=2.0)")

    opened = False
    if gate_sharpe >= 0.8 and gate_alpha_t >= 2.0:
        freeze("exp28_composite", spec)
        panel = p
        try:
            open_test_set(panel, "exp28_composite", spec,
                          {"sharpe": gate_sharpe, "alpha_t": gate_alpha_t})
            opened = True
            print("TEST SET OPENED — score on 2023-2026 follows.")
        except ProtocolViolation as e:
            print(f"protocol refused: {e}")
    else:
        print("GATE REFUSED — the composite does not earn a test-set look. "
              "Strictness is not the binding constraint; signal size is.")

    register(Experiment(
        name="exp28_composite_nearmisses",
        family="composite",
        hypothesis="DATA-INSPIRED structural question: does stacking the two "
                   "sign-consistent near-misses (low-vol + low-issuance) "
                   "clear the protocol's pre-existing test-set gate? Fund-"
                   "style diversification of individually-unprovable edges",
        features=["lowvol_q20", "lownsi_q20"],
        split="components' research/validation are selection-contaminated; "
              "only the SEALED test window could score this",
        universe="ADV63>=$50M",
        rebalance_days=REBAL, holding_days=REBAL,
        n_observations=int(((dates >= VALIDATION[0])
                            & (dates <= VALIDATION[1])).sum()),
        metrics={f"{w}_{k}_{m_}": round(float(out[w][k][m_]), 4)
                 for w in out for k in ["lowvol", "lownsi", "comp"]
                 for m_ in ["alpha", "t", "beta"]}
        | {"validation_comp_sharpe": round(gate_sharpe, 3),
           "gate_passed": bool(opened), "data_inspired": True},
        outcome="HOLD" if opened else "FAIL",
        failure_reason="" if opened else
        f"protocol gate refused test-set access (Sharpe {gate_sharpe:.2f} "
        f"vs 0.8 required, alpha_t {gate_alpha_t:.2f} vs 2.0 required); "
        "the strictness question is answered: signal size binds, not "
        "criteria",
    ))


if __name__ == "__main__":
    main()
