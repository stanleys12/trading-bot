"""
exp26b: ML feature stacking, GKX-FAITHFUL variant.
==================================================

ORIGIN, STATED HONESTLY
    DATA-INSPIRED METHODOLOGICAL CORRECTION of exp26, prompted by the
    user's challenge ("are you sure their claims aren't right and your
    test wrong?"). Audit of exp26 against Gu-Kelly-Xiu found two genuine
    infidelities that could mask a real effect:
      1. GKX rank-normalize every characteristic CROSS-SECTIONALLY each
         month into [-1, 1]; exp26 fed raw pooled values, letting the
         model waste capacity on era-level drift instead of stock-level
         differences.
      2. GKX include fundamental characteristics; exp26 was price/volume
         only. The panel carries six PIT fundamentals.
    Residual infidelity that remains and is declared: GKX tune
    hyperparameters on a validation split and their best model is a
    neural net; this variant keeps the DEFAULT boosted-tree exactly
    because a tuning loop is a forking path our protocol refuses. A null
    here therefore kills "GKX-style stacking with honest hyperparameters
    in the tradeable universe", not the paper itself. This trial pays
    its own K and the data_inspired label.

CHANGES FROM exp26 (nothing else moves)
    - all features rank-transformed to [-1, 1] within each month
    - six fundamentals added: roe, roa, book_to_market, earnings_yield,
      gross_profitability, accruals (panel columns, filed-date PIT)
    - target unchanged (21d cross-sectionally demeaned return)

PRE-REGISTERED CRITERIA — identical to exp26
    F1  research-OOS (2013-2017) IC t >= 2.0
    F2  validation (2018-2022) IC t >= 2.0, same sign
    F3  validation decile spread > turnover x 2 x 16bp
    2023+ SEALED. Survival -> HOLD.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register, trial_count  # noqa: E402
from qr.research.exp26_ml_stacking import build_features  # noqa: E402

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
ADV_MIN = 1e7
SEED = 26
RESEARCH = ("2013-01-01", "2017-12-31")
VALIDATION = ("2018-01-01", "2022-12-31")

FUNDAMENTALS = ["roe", "roa", "book_to_market", "earnings_yield",
                "gross_profitability", "accruals"]


def main() -> None:
    adj, adv63, F = build_features()

    fcols = ["symbol", "date"] + FUNDAMENTALS
    fp = pd.read_parquet(PROCESSED / "panel_v5_full.parquet", columns=fcols)
    fp["date"] = pd.to_datetime(fp["date"])
    for f in FUNDAMENTALS:
        F[f] = fp.pivot_table(index="date", columns="symbol", values=f,
                              aggfunc="last").sort_index().reindex(
                                  index=adj.index, columns=adj.columns)

    dates = adj.index
    month_ends = pd.Series(dates).groupby(dates.to_period("M")).max().values
    month_pos = [np.searchsorted(dates.values, m) for m in month_ends]
    month_pos = [i for i in month_pos if i + 22 < len(dates) and i > 260]

    feat_names = list(F)
    rows = []
    for i in month_pos:
        elig = adv63.iloc[i] >= ADV_MIN
        fwd = adj.iloc[i + 21] / adj.iloc[i] - 1
        df = pd.DataFrame({k: v.iloc[i] for k, v in F.items()})
        df["fwd"] = fwd
        df = df[elig].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["fwd"] + feat_names[:18])   # price features req
        if len(df) < 300:
            continue
        # GKX cross-sectional rank normalization to [-1, 1], per month;
        # missing fundamentals map to the cross-sectional median (0)
        for k in feat_names:
            r_ = df[k].rank(na_option="keep")
            df[k] = 2 * (r_ - 1) / (r_.max() - 1) - 1 if r_.max() > 1 else 0.0
            df[k] = df[k].fillna(0.0)
        df["fwd_dm"] = df["fwd"] - df["fwd"].mean()
        df["month"] = dates[i]
        rows.append(df.reset_index())
    data = pd.concat(rows, ignore_index=True)
    months = sorted(data["month"].unique())
    print(f"months: {len(months)}, rows: {len(data)}, features: "
          f"{len(feat_names)}  (K={trial_count()})")

    ics, spreads, turns, ic_months = [], [], [], []
    model, last_fit = None, -1
    prev_top: set = set()
    for mi, m in enumerate(months):
        if mi < 36:
            continue
        if model is None or mi - last_fit >= 12:
            train = data[data["month"] < m]
            model = HistGradientBoostingRegressor(random_state=SEED)
            model.fit(train[feat_names], train["fwd_dm"])
            last_fit = mi
        cur = data[data["month"] == m]
        pred = model.predict(cur[feat_names])
        ics.append(sstats.spearmanr(pred, cur["fwd_dm"]).statistic)
        q = pd.qcut(pd.Series(pred), 10, labels=False, duplicates="drop")
        top = set(cur["symbol"].values[q == q.max()])
        if prev_top:
            turns.append(len(top - prev_top) / max(len(top), 1))
        prev_top = top
        spreads.append(cur["fwd_dm"].values[q == q.max()].mean()
                       - cur["fwd_dm"].values[q == q.min()].mean())
        ic_months.append(m)
    ic_s = pd.Series(ics, index=pd.DatetimeIndex(ic_months))
    spread_s = pd.Series(spreads, index=pd.DatetimeIndex(ic_months))

    out = {}
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        w = ic_s[(ic_s.index >= lo) & (ic_s.index <= hi)]
        s = spread_s[(spread_s.index >= lo) & (spread_s.index <= hi)]
        t = w.mean() / (w.std() / np.sqrt(len(w)))
        out[name] = {"months": len(w), "mean_ic": w.mean(), "ic_t": t,
                     "spread_mo": s.mean()}
        print(f"{name}: {len(w)} months | IC {w.mean():+.4f} t={t:+.2f} | "
              f"decile spread {s.mean()*100:+.2f}%/mo")

    turn_med = float(np.median(turns))
    cost_mo = turn_med * 2 * 16e-4
    r, v = out["research"], out["validation"]
    f1 = bool(r["ic_t"] >= 2.0)
    f2 = bool(v["ic_t"] >= 2.0 and np.sign(v["mean_ic"]) == np.sign(r["mean_ic"]))
    f3 = bool(v["spread_mo"] > cost_mo)
    ok = f1 and f2 and f3
    print(f"\nturnover {turn_med:.2f} -> cost {cost_mo*100:.2f}%/mo")
    print(f"F1: {f1} (t={r['ic_t']:+.2f})  F2: {f2} (t={v['ic_t']:+.2f})  "
          f"F3: {f3} ({v['spread_mo']*100:+.2f}% vs {cost_mo*100:.2f}%)")
    print("HOLD" if ok else "REJECTED")

    register(Experiment(
        name="exp26b_ml_stacking_faithful",
        family="ml_stacking",
        hypothesis="GKX-faithful correction of exp26: per-month rank-"
                   "normalized features + six PIT fundamentals; prompted by "
                   "an audit of exp26's infidelities to the paper",
        features=["hgbr_24feat_ranknorm"],
        split="research-OOS 2013-2017 / validation 2018-2022; 2023+ SEALED",
        universe="ADV63>=$10M",
        rebalance_days=21, holding_days=21,
        n_observations=int(v["months"]),
        metrics={f"{s_}_{k}": round(float(x), 4)
                 for s_, d in out.items() for k, x in d.items()}
        | {"turnover": round(turn_med, 3), "F1": f1, "F2": f2, "F3": f3,
           "data_inspired": True},
        outcome="HOLD" if ok else "FAIL",
        failure_reason="" if ok else
        f"failed pre-registered criterion(s): F1={f1} F2={f2} F3={f3}",
    ))


if __name__ == "__main__":
    main()
