"""
exp16: Cross-sectional momentum, full survivorship-clean crypto universe.
=========================================================================

WRITTEN BEFORE qr/data/processed/binance_panel.parquet EXISTED.
The Binance Vision download was still running when these criteria were
fixed. Nothing below may change after the panel is built; a changed
criterion means a new experiment name and another unit of K.

WHY THIS EXPERIMENT
    exp14 (two-coin TSMOM) was withdrawn as unpowered: 3 years of two
    correlated coins cannot resolve an edge question, and more years do not
    exist. Breadth is the only lever. This tests the cross-section: does
    RELATIVE past return predict RELATIVE future return across all USDT
    pairs ever listed on Binance — delisted included, so the one bias this
    project could never fix in equities does not exist here by construction.

POWER ARGUMENT (required by protocol since 2026-08-13, computed BEFORE data)
    Primary statistic: mean weekly Spearman IC between 21d formation return
    and forward 7d return, one IC value per week (cross-sectional
    aggregation = the date-clustering lesson of exp09). Weekly IC sd is
    assumed 0.15-0.25 from equity/crypto literature; validation holds ~104
    weeks, so se(mean IC) ~ 0.015-0.025, and the pre-registered t>=2.0
    detects mean IC >= ~0.04. Liu-Tsyvinski (2021) report cross-sectional
    crypto momentum consistent with IC ~ 0.05. So the validation criterion
    CAN fail and CAN pass under realistic effect sizes: under the null,
    P(pass F2) ~ 2.5%; under the literature effect, power ~ 0.6-0.8.
    A long-short annualised-mean criterion was considered and REJECTED as
    the primary: detecting it at t>=2 over 2 years requires a ~40%/yr
    effect — that is exp14's defect again.

SPEC (all fixed now)
    Universe    USDT spot pairs from data.binance.vision, delisted included.
                Excluded: stablecoin bases (USDC/BUSD/TUSD/DAI/FDUSD/PAX/
                USDP/EUR/GBP/AUD prefixes), leveraged tokens (UP/DOWN/BULL/
                BEAR suffixes). Eligibility per week: >=60 prior daily bars,
                trailing-30d median dollar volume >= $1M.
    Formation   trailing 21d return, skipping the most recent 1d
    Forward     7d return starting one full day after formation ends
                (decision close D, position close D+1 — as exp14)
    Frequency   weekly, anchored at the first panel date
    Portfolios  quintiles by formation return, equal weight (descriptive
                table + cost arithmetic; the IC is the primary statistic)

SPLITS
    research    2017-08-17 .. 2021-12-31   (~228 weeks)
    validation  2022-01-01 .. 2023-12-31   (~104 weeks)
    test        2024-01-01 ..              SEALED (same seal as exp14)

PRE-REGISTERED CRITERIA — all three required
    F1  research mean weekly IC, t >= 3.0
    F2  validation mean weekly IC, t >= 2.0, same sign as research
    F3  cost viability arithmetic: median top-quintile weekly one-way
        turnover x 22bps (11bps taker+spread each side) must be smaller
        than the validation top-minus-bottom weekly spread POINT ESTIMATE.
        (Deterministic arithmetic on measured turnover, not a noise test;
        it can fail if the spread is thin or turnover extreme.)
    Reported, not required: Bonferroni threshold at current K; long-only
    top-quintile vs equal-weight benchmark; per-year IC breakdown.
    Any survival earns HOLD (paper forward test), never PASS — the exp14
    lesson is that this family's point estimates flatter.

SHORT LEG CAVEAT (stated now)
    The IC and quintile spread measure the cross-sectional EFFECT. Shorting
    alts is operationally constrained (borrow scarce, funding volatile), so
    any tradeable implication is long-leg-tilt only; that limitation is
    permanent and stated before scoring.
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

FORMATION, SKIP, HOLD = 21, 1, 7
ADV_MIN, MIN_HISTORY = 1e6, 60
COST_SIDE_BPS = 11.0
SPLITS = {"research": ("2017-08-17", "2021-12-31"),
          "validation": ("2022-01-01", "2023-12-31")}
T_RESEARCH, T_VALIDATION = 3.0, 2.0

STABLE_PREFIX = ("USDC", "BUSD", "TUSD", "DAI", "FDUSD", "PAX", "USDP",
                 "EUR", "GBP", "AUD", "SUSD", "UST")
LEVERAGED_SUFFIX = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(PROCESSED / "binance_panel.parquet")
    keep = ~(p["symbol"].str.startswith(STABLE_PREFIX)
             | p["symbol"].str.endswith(LEVERAGED_SUFFIX))
    return p[keep]


def weekly_ic(px: pd.DataFrame, dollar: pd.DataFrame,
              lo: str, hi: str) -> pd.DataFrame:
    """One row per week: Spearman IC, n, and quintile returns/turnover."""
    dates = px.index
    weeks = dates[::7]
    adv30 = dollar.rolling(30, min_periods=20).median()
    history = px.notna().cumsum()
    rows, prev_top = [], set()
    for d in weeks:
        i = dates.get_loc(d)
        if i < FORMATION + SKIP or i + 1 + HOLD >= len(dates):
            continue
        if not (lo <= str(d.date()) <= hi):
            continue
        form = px.iloc[i - SKIP] / px.iloc[i - SKIP - FORMATION] - 1
        fwd = px.iloc[i + 1 + HOLD] / px.iloc[i + 1] - 1
        elig = (history.iloc[i] >= MIN_HISTORY) & (adv30.iloc[i] >= ADV_MIN)
        f = pd.DataFrame({"form": form, "fwd": fwd})[elig].dropna()
        if len(f) < 20:
            continue
        ic = sstats.spearmanr(f["form"], f["fwd"]).statistic
        q = pd.qcut(f["form"], 5, labels=False, duplicates="drop")
        top = set(f.index[q == q.max()])
        turn = (len(top - prev_top) / max(len(top), 1)) if prev_top else np.nan
        prev_top = top
        rows.append({"date": d, "ic": ic, "n": len(f),
                     "q_top": f["fwd"][q == q.max()].mean(),
                     "q_bot": f["fwd"][q == 0].mean(),
                     "turnover": turn})
    return pd.DataFrame(rows).set_index("date")


def main() -> None:
    p = load_panel()
    px = p.pivot_table(index="date", columns="symbol", values="close",
                       aggfunc="last").sort_index()
    dollar = p.pivot_table(index="date", columns="symbol",
                           values="dollar_volume", aggfunc="last").sort_index()
    print(f"panel: {px.shape[1]} symbols x {px.shape[0]} days  (K={trial_count()})")

    res = {}
    for split, (lo, hi) in SPLITS.items():
        w = weekly_ic(px, dollar, lo, hi)
        t = w["ic"].mean() / (w["ic"].std() / np.sqrt(len(w)))
        res[split] = {"weeks": len(w), "mean_ic": w["ic"].mean(),
                      "ic_t": t, "avg_n": w["n"].mean(),
                      "spread_wk": (w["q_top"] - w["q_bot"]).mean(),
                      "turnover": w["turnover"].median()}
        print(f"\n=== {split} ===")
        for k, v in res[split].items():
            print(f"  {k:>10}: {v:,.4f}" if isinstance(v, float) else f"  {k:>10}: {v}")

    r, v = res["research"], res["validation"]
    cost_wk = v["turnover"] * 2 * COST_SIDE_BPS / 1e4
    f1 = bool(r["ic_t"] >= T_RESEARCH)
    f2 = bool(v["ic_t"] >= T_VALIDATION
              and np.sign(v["mean_ic"]) == np.sign(r["mean_ic"]))
    f3 = bool(v["spread_wk"] > cost_wk)
    ok = f1 and f2 and f3
    print(f"\nF1 research IC t>={T_RESEARCH}: {f1} (t={r['ic_t']:+.2f})")
    print(f"F2 validation IC t>={T_VALIDATION}, same sign: {f2} (t={v['ic_t']:+.2f})")
    print(f"F3 spread {v['spread_wk']*100:.2f}%/wk > cost {cost_wk*100:.2f}%/wk: {f3}")
    print(f"\n{'HOLD -- paper forward test next; never PASS from backtest' if ok else 'REJECTED'}")

    register(Experiment(
        name="exp16_crypto_xs_momentum",
        family="crypto",
        hypothesis="relative past return predicts relative future return "
                   "across the full survivorship-clean Binance USDT "
                   "cross-section (Liu-Tsyvinski 2021); IC-primary design "
                   "per the power rule",
        features=["xs_mom_21d_skip1"],
        split="research 2017-2021 / validation 2022-2023; 2024+ SEALED",
        universe="Binance USDT spot incl. delisted, ADV>=$1M",
        rebalance_days=7, holding_days=7,
        n_observations=int(v["weeks"]),
        metrics={f"{s}_{k}": round(float(x), 4)
                 for s, d in res.items() for k, x in d.items()}
        | {"F1": f1, "F2": f2, "F3": f3},
        outcome="HOLD" if ok else "FAIL",
        failure_reason="" if ok else
        f"failed pre-registered criterion(s): F1={f1} F2={f2} F3={f3}",
    ))


if __name__ == "__main__":
    main()
