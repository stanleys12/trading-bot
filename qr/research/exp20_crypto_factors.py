"""
exp20: Crypto cross-sectional factor screen — reversal, size, low-vol.
======================================================================

CRITERIA FIXED BEFORE SCORING. Same panel, same eligibility, same splits,
same IC-primary powered design as exp16 (which rejected momentum). This
completes the documented crypto factor families on the survivorship-clean
universe rather than leaving them to be tried one at a time forever.

FACTORS (sign convention: positive IC = hypothesis confirmed)
    REV     signal = -(trailing 7d return)          weekly reversal
    SIZE    signal = -log(trailing 30d median $vol) small beats big
                                                    (Liu-Tsyvinski's second
                                                    robust factor)
    LOWVOL  signal = -(trailing 30d daily vol)      low-risk anomaly

POWER (same arithmetic as exp16, computed before scoring)
    Mean weekly Spearman IC; ~104 validation weeks -> se ~ 0.015-0.025;
    t >= 2 detects |IC| >= ~0.04 against literature effects ~0.05.
    Under the null each factor passes F2 ~2.5% of the time.

MULTIPLE TESTING
    Three registered trials (K += 3). Family Bonferroni for the validation
    criterion is |t| >= 2.4 (5%/3, one-sided); reported per factor, and a
    factor is only promoted to HOLD if it ALSO clears the family threshold.

PER-FACTOR CRITERIA (as exp16)
    F1  research IC t >= 3.0
    F2  validation IC t >= 2.0, same sign as research
        (family threshold 2.4 additionally required for promotion)
    F3  top-minus-bottom quintile weekly spread (validation, point est.)
        exceeds median weekly one-way turnover x 22bps
    Survival -> HOLD, never PASS. 2024+ remains SEALED.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register, trial_count  # noqa: E402
from qr.research.exp16_crypto_xs_momentum import (  # noqa: E402
    load_panel, ADV_MIN, MIN_HISTORY, COST_SIDE_BPS, SPLITS)

HOLD_D = 7
T_RESEARCH, T_VALIDATION, T_FAMILY = 3.0, 2.0, 2.4


def weekly_ic_generic(px, dollar, signal, lo, hi):
    dates = px.index
    weeks = dates[::7]
    adv30 = dollar.rolling(30, min_periods=20).median()
    history = px.notna().cumsum()
    rows, prev_top = [], set()
    for d in weeks:
        i = dates.get_loc(d)
        if i < 35 or i + 1 + HOLD_D >= len(dates):
            continue
        if not (lo <= str(d.date()) <= hi):
            continue
        sig = signal.iloc[i]
        fwd = px.iloc[i + 1 + HOLD_D] / px.iloc[i + 1] - 1
        elig = (history.iloc[i] >= MIN_HISTORY) & (adv30.iloc[i] >= ADV_MIN)
        f = pd.DataFrame({"sig": sig, "fwd": fwd})[elig].dropna()
        if len(f) < 20:
            continue
        ic = sstats.spearmanr(f["sig"], f["fwd"]).statistic
        q = pd.qcut(f["sig"], 5, labels=False, duplicates="drop")
        top = set(f.index[q == q.max()])
        turn = (len(top - prev_top) / max(len(top), 1)) if prev_top else np.nan
        prev_top = top
        rows.append({"date": d, "ic": ic, "n": len(f),
                     "spread": f["fwd"][q == q.max()].mean() - f["fwd"][q == 0].mean(),
                     "turnover": turn})
    return pd.DataFrame(rows).set_index("date")


def main() -> None:
    p = load_panel()
    px = p.pivot_table(index="date", columns="symbol", values="close",
                       aggfunc="last").sort_index()
    dollar = p.pivot_table(index="date", columns="symbol",
                           values="dollar_volume", aggfunc="last").sort_index()
    ret = px.pct_change()

    factors = {
        "REV": -(px / px.shift(7) - 1),
        "SIZE": -np.log(dollar.rolling(30, min_periods=20).median()),
        "LOWVOL": -ret.rolling(30, min_periods=20).std(),
    }
    print(f"panel: {px.shape[1]} symbols  (K={trial_count()})")

    for name, sig in factors.items():
        res = {}
        for split, (lo, hi) in SPLITS.items():
            w = weekly_ic_generic(px, dollar, sig, lo, hi)
            t = w["ic"].mean() / (w["ic"].std() / np.sqrt(len(w)))
            res[split] = {"weeks": len(w), "mean_ic": w["ic"].mean(), "ic_t": t,
                          "spread_wk": w["spread"].mean(),
                          "turnover": w["turnover"].median()}
        r, v = res["research"], res["validation"]
        cost_wk = v["turnover"] * 2 * COST_SIDE_BPS / 1e4
        f1 = bool(r["ic_t"] >= T_RESEARCH)
        f2 = bool(v["ic_t"] >= T_VALIDATION
                  and np.sign(v["mean_ic"]) == np.sign(r["mean_ic"]))
        fam = bool(v["ic_t"] >= T_FAMILY)
        f3 = bool(v["spread_wk"] > cost_wk)
        ok = f1 and f2 and fam and f3
        print(f"\n=== {name} ===")
        print(f"  research:   IC {r['mean_ic']:+.4f}  t {r['ic_t']:+.2f}  ({r['weeks']} wks)")
        print(f"  validation: IC {v['mean_ic']:+.4f}  t {v['ic_t']:+.2f}  ({v['weeks']} wks)")
        print(f"  spread {v['spread_wk']*100:+.2f}%/wk vs cost {cost_wk*100:.2f}%/wk")
        print(f"  F1={f1} F2={f2} family(2.4)={fam} F3={f3} -> "
              f"{'HOLD' if ok else 'FAIL'}")

        register(Experiment(
            name=f"exp20_crypto_{name.lower()}",
            family="crypto",
            hypothesis={"REV": "weekly reversal in the crypto cross-section",
                        "SIZE": "small coins outperform (Liu-Tsyvinski size factor)",
                        "LOWVOL": "low-volatility anomaly in crypto"}[name],
            features=[name.lower()],
            split="research 2017-2021 / validation 2022-2023; 2024+ SEALED",
            universe="Binance USDT spot incl. delisted, ADV>=$1M",
            rebalance_days=7, holding_days=7,
            n_observations=int(v["weeks"]),
            metrics={f"{s}_{k}": round(float(x), 4)
                     for s, d in res.items() for k, x in d.items()}
            | {"F1": f1, "F2": f2, "family_2p4": fam, "F3": f3},
            outcome="HOLD" if ok else "FAIL",
            failure_reason="" if ok else
            f"failed pre-registered criterion(s): F1={f1} F2={f2} "
            f"family={fam} F3={f3}",
        ))


if __name__ == "__main__":
    main()
