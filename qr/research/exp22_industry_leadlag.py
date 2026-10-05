"""
exp22: Industry lead-lag — do big names' moves predict small peers?
===================================================================

CRITERIA FIXED BEFORE SCORING. A NEW information-structure family for
this project: not a characteristic of the stock itself but information
diffusing ACROSS stocks with a delay (Hou 2007 JFE; Cohen-Frazzini 2008
economic links). Small caps are followed by few analysts; industry news
lands in the leaders first and walks down the cap ladder over days.

SPEC
    Weekly, decision at close D, positions from close D+1.
    LEADERS   per industry: names with trailing-63d mean dollar volume
              > $200M; industry needs >= 2 leaders that week.
    FOLLOWERS ADV63 in [$2M, $50M], industry-labelled, not a leader.
    SIGNAL    follower's signal = equal-weight trailing 5d return of its
              industry's leaders.
    PRIMARY   pooled weekly Spearman IC of signal vs forward 5d return
              (close D+1 -> close D+6).
    Diagnostic only (never scored): partial IC controlling for the
    follower's own trailing 5d return.

POWER (before scoring)
    ~1,000 followers/week pooled -> weekly IC sd ~0.10-0.20; research has
    ~400 weeks, validation ~260 -> se(mean IC) ~ 0.006-0.012. t >= 2
    detects mean IC >= ~0.02; Hou reports effects well above that in
    1963-2001 data. Passes under the null ~2.5% of the time. Powered.

PRE-REGISTERED CRITERIA — all required
    F1  research mean weekly IC t >= 3.0
    F2  validation mean weekly IC t >= 2.0, same sign as research
    F3  cost arithmetic: validation top-minus-bottom quintile weekly
        spread must exceed median weekly one-way turnover x 2 x 25bps
        (the measured spread band for $2-50M ADV names)
    TEST 2023+ SEALED. Survival -> HOLD.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register, trial_count  # noqa: E402

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

LEADER_ADV, FOLLOW_LO, FOLLOW_HI = 2e8, 2e6, 5e7
RESEARCH = ("2010-01-01", "2017-12-31")
VALIDATION = ("2018-01-01", "2022-12-31")


def main() -> None:
    cols = ["symbol", "date", "adj_close", "dollar_volume", "industry"]
    p = pd.read_parquet(PROCESSED / "panel_v5_full.parquet", columns=cols)
    p["date"] = pd.to_datetime(p["date"])
    industry = p.dropna(subset=["industry"]).groupby("symbol")["industry"] \
                .agg(lambda s: s.mode().iat[0])
    px = p.pivot_table(index="date", columns="symbol", values="adj_close",
                       aggfunc="last").sort_index()
    dv = p.pivot_table(index="date", columns="symbol", values="dollar_volume",
                       aggfunc="last").sort_index()
    adv63 = dv.rolling(63, min_periods=40).mean()
    r5 = px / px.shift(5) - 1
    dates = px.index
    ind = industry.reindex(px.columns)
    ind_codes, ind_names = pd.factorize(ind, use_na_sentinel=True)
    n_ind = len(ind_names)
    print(f"symbols: {px.shape[1]}, industries: {n_ind}  (K={trial_count()})")

    rows, prev_top = [], set()
    for i in range(70, len(dates) - 7, 5):
        d = dates[i]
        adv = adv63.values[i]
        lead_mask = adv > LEADER_ADV
        fol_mask = (adv >= FOLLOW_LO) & (adv <= FOLLOW_HI) & ~lead_mask
        r5v = r5.values[i]
        # leader mean 5d return per industry
        sums = np.zeros(n_ind)
        cnts = np.zeros(n_ind)
        for j in np.where(lead_mask & np.isfinite(r5v) & (ind_codes >= 0))[0]:
            sums[ind_codes[j]] += r5v[j]
            cnts[ind_codes[j]] += 1
        leader_ret = np.where(cnts >= 2, sums / np.maximum(cnts, 1), np.nan)
        fwd = px.values[i + 6] / px.values[i + 1] - 1
        f_idx = np.where(fol_mask & (ind_codes >= 0) & np.isfinite(fwd)
                         & np.isfinite(r5v))[0]
        sig = leader_ret[ind_codes[f_idx]]
        own = r5v[f_idx]
        fw = fwd[f_idx]
        m = np.isfinite(sig)
        if m.sum() < 100:
            continue
        sig, fw, own = sig[m], fw[m], own[m]
        ic = sstats.spearmanr(sig, fw).statistic
        # partial IC: residualize both on follower's own 5d return (ranks)
        rs, rf, ro = (sstats.rankdata(x) for x in (sig, fw, own))
        res_s = rs - np.polyval(np.polyfit(ro, rs, 1), ro)
        res_f = rf - np.polyval(np.polyfit(ro, rf, 1), ro)
        pic = np.corrcoef(res_s, res_f)[0, 1]
        q = pd.qcut(pd.Series(sig), 5, labels=False, duplicates="drop").values
        top = set(f_idx[m][q == np.nanmax(q)])
        turn = (len(top - prev_top) / max(len(top), 1)) if prev_top else np.nan
        prev_top = top
        rows.append({"date": d, "ic": ic, "pic": pic, "n": m.sum(),
                     "spread": fw[q == np.nanmax(q)].mean() - fw[q == 0].mean(),
                     "turnover": turn})
    wk = pd.DataFrame(rows).set_index("date")

    out = {}
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        w = wk[(wk.index >= lo) & (wk.index <= hi)]
        t = w.ic.mean() / (w.ic.std() / np.sqrt(len(w)))
        tp = w.pic.mean() / (w.pic.std() / np.sqrt(len(w)))
        out[name] = {"weeks": len(w), "mean_ic": w.ic.mean(), "ic_t": t,
                     "partial_ic": w.pic.mean(), "partial_t": tp,
                     "avg_n": w.n.mean(), "spread_wk": w.spread.mean(),
                     "turnover": w.turnover.median()}
        print(f"\n=== {name} ===")
        print(f"  IC {w.ic.mean():+.4f} t={t:+.2f} | partial IC "
              f"{w.pic.mean():+.4f} t={tp:+.2f} | {len(w)} wks, "
              f"avg {w.n.mean():.0f} followers")
        print(f"  Q5-Q1 spread {w.spread.mean()*100:+.3f}%/wk, "
              f"turnover {w.turnover.median():.2f}")

    r, v = out["research"], out["validation"]
    cost_wk = v["turnover"] * 2 * 25e-4
    f1 = bool(r["ic_t"] >= 3.0)
    f2 = bool(v["ic_t"] >= 2.0 and np.sign(v["mean_ic"]) == np.sign(r["mean_ic"]))
    f3 = bool(v["spread_wk"] > cost_wk)
    ok = f1 and f2 and f3
    print(f"\nF1 research t>=3: {f1} (t={r['ic_t']:+.2f})")
    print(f"F2 validation t>=2, same sign: {f2} (t={v['ic_t']:+.2f})")
    print(f"F3 spread {v['spread_wk']*100:+.3f}%/wk > cost {cost_wk*100:.3f}%/wk: {f3}")
    print("HOLD" if ok else "REJECTED")

    register(Experiment(
        name="exp22_industry_leadlag",
        family="cross_stock_diffusion",
        hypothesis="industry information lands in large names first and "
                   "diffuses to small peers over days (Hou 2007)",
        features=["leader_5d_return"],
        split="research+validation; 2023+ SEALED",
        universe="followers $2-50M ADV, leaders >$200M ADV, industry-matched",
        rebalance_days=5, holding_days=5,
        n_observations=int(v["weeks"]),
        metrics={f"{s}_{k}": round(float(x), 4)
                 for s, d in out.items() for k, x in d.items()}
        | {"F1": f1, "F2": f2, "F3": f3},
        outcome="HOLD" if ok else "FAIL",
        failure_reason="" if ok else
        f"failed pre-registered criterion(s): F1={f1} F2={f2} F3={f3}",
    ))


if __name__ == "__main__":
    main()
