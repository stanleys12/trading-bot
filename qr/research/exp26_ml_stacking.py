"""
exp26: Machine-learning feature stacking (Gu-Kelly-Xiu / qlib approach).
========================================================================

CRITERIA AND SPEC FIXED BEFORE SCORING. Everything below — the 18
features, the model, its default hyperparameters, the walk-forward
schedule, the seed — was frozen before any prediction was computed.
There is no tuning loop; there will be no second configuration under
this name.

WHY THIS IS GENUINELY NEW HERE
    All 242 prior trials tested hypotheses ONE AT A TIME with linear/rank
    statistics. The open-source quant mainstream (Microsoft qlib's
    Alpha158 benchmarks) and the strongest recent academic result
    (Gu-Kelly-Xiu 2020 RFS) do the opposite: many weak features, one
    nonlinear model, interactions allowed. GKX report boosted trees and
    neural nets roughly doubling linear strategies' performance. If
    nonlinear interaction effects exist in our panel, this is the
    instrument that can see them and nothing we ran before could.

SPEC
    Universe    ADV63 >= $10M (costs measurable, opens real)
    Features    18, all price/volume (GKX's strongest class), all
                computed from data through the signal date:
                mom21, mom63, mom126, mom252skip21, rev5, vol21, vol63,
                skew63, log_adv, amihud21, maxret21, minret21,
                dist_52w_high, dist_52w_low, range21, overnight_share63,
                log_price, beta_proxy63
    Target      forward 21d return, cross-sectionally demeaned
    Model       sklearn HistGradientBoostingRegressor, DEFAULT
                hyperparameters, random_state=26
    Walk-forward expanding window; first fit after 36 months of history;
                refit every 12 months; predictions are always strictly
                out-of-sample
    Statistic   monthly Spearman IC (prediction vs realised), t across
                months; decile spread and turnover for costs

SPLITS
    research OOS   2013-01 .. 2017-12   (60 monthly ICs)
    validation     2018-01 .. 2022-12   (60 monthly ICs)
    2023+ SEALED.

POWER (before scoring)
    se(mean IC) ~ 0.15/sqrt(60) ~ 0.019 per window; t >= 2 detects
    |IC| >= ~0.039. GKX-class monthly ICs are 0.03-0.06: powered for the
    upper half of that range, marginal below. Null pass ~2.5%/window.

PRE-REGISTERED CRITERIA — all required
    F1  research-OOS IC t >= 2.0
    F2  validation IC t >= 2.0, same sign
    F3  validation decile spread > turnover x 2 x 16bp per month
    Survival -> HOLD.
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

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
ADV_MIN = 1e7
SEED = 26
RESEARCH = ("2013-01-01", "2017-12-31")
VALIDATION = ("2018-01-01", "2022-12-31")


def build_features():
    cols = ["symbol", "date", "open", "close", "adj_close", "true_close",
            "dollar_volume"]
    p = pd.read_parquet(PROCESSED / "panel_v5_full.parquet", columns=cols)
    p["date"] = pd.to_datetime(p["date"])
    adj = p.pivot_table(index="date", columns="symbol", values="adj_close",
                        aggfunc="last").sort_index()
    o = p.pivot_table(index="date", columns="symbol", values="open",
                      aggfunc="last").sort_index()
    c = p.pivot_table(index="date", columns="symbol", values="close",
                      aggfunc="last").sort_index()
    tc = p.pivot_table(index="date", columns="symbol", values="true_close",
                       aggfunc="last").sort_index()
    dv = p.pivot_table(index="date", columns="symbol", values="dollar_volume",
                       aggfunc="last").sort_index()
    r = adj.pct_change()
    spy = (pd.read_parquet(PROCESSED / "etf_panel.parquet")
           .query("symbol=='SPY'").set_index("date")["adj_close"])
    spy.index = pd.to_datetime(spy.index)
    spy_r = spy.reindex(adj.index).ffill().pct_change()

    on = (o / c.shift(1) - 1).clip(-0.5, 0.5)
    intra = (c / o - 1).clip(-0.5, 0.5)
    roll_max = adj.rolling(252, min_periods=126).max()
    roll_min = adj.rolling(252, min_periods=126).min()
    beta = (r.rolling(63).corr(spy_r)
            * r.rolling(63).std().div(spy_r.rolling(63).std(), axis=0))

    F = {
        "mom21": adj / adj.shift(21) - 1,
        "mom63": adj / adj.shift(63) - 1,
        "mom126": adj / adj.shift(126) - 1,
        "mom252skip21": adj.shift(21) / adj.shift(252) - 1,
        "rev5": adj / adj.shift(5) - 1,
        "vol21": r.rolling(21).std(),
        "vol63": r.rolling(63).std(),
        "skew63": r.rolling(63).skew(),
        "log_adv": np.log(dv.rolling(63, min_periods=40).mean()),
        "amihud21": (r.abs() / dv).rolling(21).mean(),
        "maxret21": r.rolling(21).max(),
        "minret21": r.rolling(21).min(),
        "dist_52w_high": adj / roll_max - 1,
        "dist_52w_low": adj / roll_min - 1,
        "range21": ((c - o).abs() / o).rolling(21).mean(),
        "overnight_share63": on.rolling(63).mean() - intra.rolling(63).mean(),
        "log_price": np.log(tc.clip(lower=0.01)),
        "beta_proxy63": beta,
    }
    adv63 = dv.rolling(63, min_periods=40).mean()
    return adj, adv63, F


def main() -> None:
    adj, adv63, F = build_features()
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
        df = df[elig].replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < 300:
            continue
        df["fwd_dm"] = df["fwd"] - df["fwd"].mean()
        df["month"] = dates[i]
        rows.append(df.reset_index())
    data = pd.concat(rows, ignore_index=True)
    months = sorted(data["month"].unique())
    print(f"months: {len(months)}, rows: {len(data)}  (K={trial_count()})")

    # walk-forward: refit every 12 months on all data strictly before the
    # prediction month, first fit after 36 months of history
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
        ic = sstats.spearmanr(pred, cur["fwd_dm"]).statistic
        q = pd.qcut(pd.Series(pred), 10, labels=False, duplicates="drop")
        top = set(cur["symbol"].values[q == q.max()])
        if prev_top:
            turns.append(len(top - prev_top) / max(len(top), 1))
        prev_top = top
        spreads.append(cur["fwd_dm"].values[q == q.max()].mean()
                       - cur["fwd_dm"].values[q == q.min()].mean())
        ics.append(ic)
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
    print(f"F1 research t>=2: {f1} (t={r['ic_t']:+.2f})")
    print(f"F2 validation t>=2, same sign: {f2} (t={v['ic_t']:+.2f})")
    print(f"F3 spread>cost: {f3} ({v['spread_mo']*100:+.2f}% vs {cost_mo*100:.2f}%)")
    print("HOLD" if ok else "REJECTED")

    register(Experiment(
        name="exp26_ml_feature_stacking",
        family="ml_stacking",
        hypothesis="nonlinear interactions among 18 price/volume features "
                   "predict cross-sectional returns where single signals "
                   "failed (Gu-Kelly-Xiu 2020; qlib Alpha158 paradigm); "
                   "frozen default-hyperparameter model, no tuning loop",
        features=["hgbr_18feat_default"],
        split="research-OOS 2013-2017 / validation 2018-2022; 2023+ SEALED",
        universe="ADV63>=$10M",
        rebalance_days=21, holding_days=21,
        n_observations=int(v["months"]),
        metrics={f"{s_}_{k}": round(float(x), 4)
                 for s_, d in out.items() for k, x in d.items()}
        | {"turnover": round(turn_med, 3), "F1": f1, "F2": f2, "F3": f3},
        outcome="HOLD" if ok else "FAIL",
        failure_reason="" if ok else
        f"failed pre-registered criterion(s): F1={f1} F2={f2} F3={f3}",
    ))


if __name__ == "__main__":
    main()
