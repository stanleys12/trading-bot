"""
EXPERIMENT 09 — the question this project opened with, asked properly.
======================================================================

PRE-REGISTERED BEFORE SEEING ANY PREMARKET DATA.

THE HISTORY
The inherited system claimed 86% win rate on "gap up + RVOL >= 5x, buy at the
open, exit at the close". That RVOL used the day's ENTIRE volume -- a quantity
that does not exist at 09:30 and is mechanically correlated with the return it
predicts. Substituting the lagged equivalent gave 42.9%, against a 43.7%
no-filter baseline. The signal was the leak.

But the strategy was never wrong to WANT a volume confirmation. Premarket
volume is real, observable at 09:25, and was simply not in the daily bars the
original research used. Every conclusion since has been conditioned on data
that could not express the question.

Now it can.

HYPOTHESIS
Genuine premarket volume, measured through 09:25 ET and normalised against
that stock's own trailing premarket norm, predicts open-to-close continuation
on gap-up days.

    pm_rvol = pm_volume(09:25) / median(pm_volume, trailing 20 sessions)

FALSIFICATION -- any one of these is a rejection
  F1  pm_rvol shows no monotone relationship with open-to-close returns
  F2  the relationship exists but reverses out of sample
  F3  it survives statistically but not after measured spreads (we now have
      real spreads by ADV bucket, so this is no longer an estimate)
  F4  the effect is entirely explained by the gap itself -- i.e. pm_rvol adds
      nothing incremental to pm_gap, which the original system also had

F4 matters most. The original claim was specifically that VOLUME confirmed the
gap. If gap alone does the work, the volume story was never the mechanism.

AMENDMENT 2026-08-13, BEFORE ANY PREMARKET DATA EXISTED
The thresholds above were calibrated by simulating the null, and the original
F1 did not survive that calibration. Two defects:

  1. F1 was |corr(decile_index, decile_mean)| > 0.5. That correlation runs
     across TEN points, so its null sd is 0.34 -- the threshold sat half a
     standard deviation from zero. Monte Carlo over 4,000 draws of pure noise:
     it fires 14.5% of the time. A criterion that passes on noise once every
     seven attempts is not a falsification criterion.

  2. Every statistic here was computed as though symbol-days were independent.
     They are not: gap-ups cluster hard on market-wide days, so the effective
     sample is the number of DATES, not the number of rows. Uncorrected
     t-statistics inflate by roughly sqrt(names per day).

The primary statistic is therefore now the regression coefficient on
log(pm_rvol) with standard errors clustered by DATE -- which the same Monte
Carlo confirms is correctly sized (null sd 0.99, P(|t|>2) = 4.1%). The decile
table is retained as description, with its threshold raised to the simulated
95th percentile of the null (0.63) rather than an eyeballed 0.5.

Thresholds are also raised for multiplicity: this is trial 213. Bonferroni at
5% over that many trials needs |t| > 3.7, which is reported alongside.

This amendment was written before qr/data/processed/premarket_features.parquet
existed. It tightens every criterion; it loosens none.

SPLITS
  RESEARCH    2016-2021
  VALIDATION  2022-2024
  TEST        2025-2026  SEALED

WHAT WOULD MAKE THIS DIFFERENT FROM 212 FAILURES
Nothing about the method. The same protocol, the same scorecard, the same
sealed test set. The only difference is that the question is finally
answerable with data that exists. That improves the odds of a TRUE answer,
not the odds of a POSITIVE one.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register, trial_count

logger = logging.getLogger("exp09")
HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
PROCESSED = HERE.parent / "data" / "processed"

SPLITS = {"research": ("2016-01-01", "2021-12-31"),
          "validation": ("2022-01-01", "2024-12-31"),
          "test": ("2025-01-01", "2026-08-11")}

# calibrated by Monte Carlo against the null -- see AMENDMENT in the docstring
T_RESEARCH = 3.0        # primary, clustered by date
T_VALIDATION = 2.0
MONO_MIN = 0.63         # 95th pctile of |r| under the null, not an eyeball
T_BONFERRONI = 3.7      # 5% over 213 trials, reported not required


# measured spread model fitted to 2.05M SIP quotes (qr/data/spread_audit.py)
def spread_bps(adv: float) -> float:
    return 10 ** (4.288 - 0.416 * np.log10(max(adv, 1e5)))


def ols_clustered(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> tuple:
    """
    OLS with standard errors clustered on `groups` (here: the date).

    Gap-up days are not independent draws. On a day the whole market gaps,
    hundreds of names gap together and their open-to-close returns share a
    common factor. Treating those as independent observations overstates the
    sample by roughly the number of names per day and inflates every
    t-statistic by its square root. Clustering on date is the minimum
    correction; it is what makes the number below comparable to the simulated
    null.
    """
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)

    meat = np.zeros((k, k))
    order = np.argsort(groups, kind="stable")
    gs, Xs, us = groups[order], X[order], u[order]
    edges = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        sg = Xs[a:b].T @ us[a:b]
        meat += np.outer(sg, sg)

    G = len(edges) - 1
    if G > 1:
        meat *= (G / (G - 1)) * ((n - 1) / (n - k))
    V = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    r2 = 1 - u.var() / y.var() if y.var() > 0 else np.nan
    return beta, np.divide(beta, se, out=np.zeros_like(beta), where=se > 0), r2, G


def load() -> pd.DataFrame:
    f = pd.read_parquet(PROCESSED / "premarket_features.parquet")
    f["date"] = pd.to_datetime(f["date"])
    f["adv_dollar"] = f["adv20"] * f["prev_close"]
    f["cost_bps"] = f["adv_dollar"].apply(spread_bps)      # full spread
    return f


def cut(f: pd.DataFrame, split: str) -> pd.DataFrame:
    lo, hi = SPLITS[split]
    return f[(f["date"] >= lo) & (f["date"] <= hi)]


def decile_table(d: pd.DataFrame, feat: str, outcome: str = "open_to_close_pct",
                 n: int = 10) -> pd.DataFrame:
    x = d[[feat, outcome, "cost_bps"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 500:
        return pd.DataFrame()
    x = x.assign(bucket=pd.qcut(x[feat].rank(method="first"), n, labels=False))
    return (x.groupby("bucket")
            .agg(n=(outcome, "size"), mean=(outcome, "mean"),
                 median=(outcome, "median"),
                 win=(outcome, lambda s: (s > 0).mean() * 100),
                 cost=("cost_bps", "median")).reset_index())


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    f = load()
    logger.info("premarket dataset: %d symbol-days, %d symbols, %s..%s  (K=%d)",
                len(f), f.symbol.nunique(), f.date.min().date(),
                f.date.max().date(), trial_count())

    # the original strategy's population: gap ups
    gap = f[(f["pm_gap_pct"] >= 2) & (f["prev_close"] >= 2)]
    logger.info("gap-up >=2%% at 09:25: %d sessions (%.1f%% of all)\n",
                len(gap), len(gap) / max(len(f), 1) * 100)

    res = cut(gap, "research")
    val = cut(gap, "validation")
    logger.info("research %d sessions | validation %d sessions\n",
                len(res), len(val))

    # ---------------------------------------------------- F1: monotone?
    logger.info("=== F1  OPEN-TO-CLOSE BY PREMARKET RVOL DECILE (research) ===")
    t = decile_table(res, "pm_rvol")
    if t.empty:
        logger.error("insufficient data")
        return
    logger.info("  %-8s %7s %9s %9s %8s %8s", "decile", "n", "mean%", "median%",
                "win%", "cost bps")
    for _, r in t.iterrows():
        logger.info("  D%-7d %7d %+9.3f %+9.3f %7.1f %8.0f",
                    int(r.bucket) + 1, r.n, r["mean"], r["median"], r.win, r.cost)
    mono = np.corrcoef(t.bucket, t["mean"])[0, 1]
    spread_hi_lo = t["mean"].iloc[-1] - t["mean"].iloc[0]
    logger.info("\n  monotonicity r = %+.2f | D10-D1 spread = %+.3f%% | "
                "round-trip cost = %.2f%%", mono, spread_hi_lo,
                t["cost"].median() / 1e4 * 2 * 100)

    # ------------------------------- PRIMARY + F4: incremental to the gap?
    logger.info("\n=== PRIMARY  REGRESSION, SE CLUSTERED BY DATE ===")
    logger.info("  (uncorrected t is shown only to expose the clustering "
                "inflation)")

    def fit(d: pd.DataFrame, cols: list[str]) -> dict:
        x = d[["pm_rvol", "pm_gap_pct", "open_to_close_pct", "date"]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        x = x[x.pm_rvol > 0].assign(log_rvol=lambda z: np.log(z.pm_rvol))
        if len(x) < 500:
            return {}
        X = np.column_stack([np.ones(len(x))] + [x[c].values for c in cols])
        y = x.open_to_close_pct.values
        beta, tcl, r2, G = ols_clustered(y, X, x["date"].values)
        u = y - X @ beta
        se_iid = np.sqrt(u.var() * np.diag(np.linalg.inv(X.T @ X)))
        t_iid = beta / np.where(se_iid > 0, se_iid, np.nan)
        return {"cols": cols, "beta": beta[1:], "t": tcl[1:],
                "t_iid": t_iid[1:], "r2": r2, "n": len(x), "G": G}

    fits = {}
    for label, cols in (("gap only", ["pm_gap_pct"]),
                        ("rvol only", ["log_rvol"]),
                        ("gap + rvol", ["pm_gap_pct", "log_rvol"])):
        r = fit(res, cols)
        fits[label] = r
        if not r:
            continue
        logger.info("  %-12s n=%d dates=%d R2=%.5f", label, r["n"], r["G"], r["r2"])
        for c, b, tc, ti in zip(cols, r["beta"], r["t"], r["t_iid"]):
            logger.info("      %-12s b=%+.4f  t_clustered=%+.2f  (t_iid=%+.2f)",
                        c, b, tc, ti)

    t_primary = (fits.get("rvol only") or {}).get("t", [np.nan])[0]
    joint = fits.get("gap + rvol") or {}
    t_rvol_given_gap = joint.get("t", [np.nan, np.nan])[-1] if joint else np.nan
    logger.info("\n  F4 -- pm_rvol controlling for the gap: t = %+.2f",
                t_rvol_given_gap)
    logger.info("  Bonferroni threshold over %d trials: |t| > %.1f",
                trial_count(), T_BONFERRONI)

    # ------------------------------------------ F2: does it hold out of sample?
    logger.info("\n=== F2  VALIDATION (2022-2024) ===")
    fv = fit(val, ["log_rvol"])
    fv_joint = fit(val, ["pm_gap_pct", "log_rvol"])
    t_val = fv.get("t", [np.nan])[0] if fv else np.nan
    t_val_given_gap = (fv_joint.get("t", [np.nan, np.nan])[-1]
                       if fv_joint else np.nan)
    if fv:
        logger.info("  rvol only   n=%d dates=%d  b=%+.4f  t_clustered=%+.2f",
                    fv["n"], fv["G"], fv["beta"][0], t_val)
    if fv_joint:
        logger.info("  gap + rvol  t on log_rvol = %+.2f", t_val_given_gap)

    tv = decile_table(val, "pm_rvol")
    if not tv.empty:
        mono_v = np.corrcoef(tv.bucket, tv["mean"])[0, 1]
        spread_v = tv["mean"].iloc[-1] - tv["mean"].iloc[0]
        logger.info("  monotonicity r = %+.2f | D10-D1 spread = %+.3f%%",
                    mono_v, spread_v)
        logger.info("  D1 mean %+.3f%% (n=%d) | D10 mean %+.3f%% (n=%d)",
                    tv["mean"].iloc[0], tv.n.iloc[0], tv["mean"].iloc[-1], tv.n.iloc[-1])
    else:
        mono_v, spread_v = np.nan, np.nan

    # ------------------------------------- F3: net of MEASURED spread
    logger.info("\n=== F3  TOP-DECILE STRATEGY, NET OF MEASURED SPREAD ===")
    logger.info("  %-12s %8s %10s %10s %10s", "split", "n", "gross%", "cost%", "net%")
    net_ok = {}
    for label, d in (("research", res), ("validation", val)):
        x = d.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["pm_rvol", "open_to_close_pct", "cost_bps"])
        if len(x) < 500:
            continue
        top = x[x.pm_rvol >= x.pm_rvol.quantile(0.90)]
        gross = top.open_to_close_pct.mean()
        cost = top.cost_bps.median() / 1e4 * 100          # full spread, round trip
        logger.info("  %-12s %8d %+10.3f %10.3f %+10.3f",
                    label, len(top), gross, cost, gross - cost)
        net_ok[label] = gross - cost > 0

    # ------------------------------------------------------------ verdict
    logger.info("\n=== PRE-REGISTERED FALSIFICATION (amended thresholds) ===")
    f1 = (np.isfinite(t_primary) and abs(t_primary) >= T_RESEARCH
          and abs(mono) >= MONO_MIN)
    f2 = (np.isfinite(t_val) and np.isfinite(t_primary)
          and np.sign(t_val) == np.sign(t_primary) and abs(t_val) >= T_VALIDATION)
    f3 = all(net_ok.values()) and len(net_ok) == 2
    f4 = (np.isfinite(t_rvol_given_gap) and np.isfinite(t_primary)
          and np.sign(t_rvol_given_gap) == np.sign(t_primary)
          and abs(t_rvol_given_gap) >= T_RESEARCH)
    logger.info("  F1 clustered |t|>=%.1f and mono>=%.2f : %s "
                "(t=%+.2f, r=%+.2f)", T_RESEARCH, MONO_MIN, f1, t_primary, mono)
    logger.info("  F2 holds in validation |t|>=%.1f     : %s (t=%+.2f)",
                T_VALIDATION, f2, t_val)
    logger.info("  F3 net positive after measured spread: %s", f3)
    logger.info("  F4 incremental to the gap alone      : %s (t=%+.2f)",
                f4, t_rvol_given_gap)
    ok = f1 and f2 and f3 and f4

    reason = "" if ok else (
        f"F1={f1} F2={f2} F3={f3} F4={f4}. Premarket RVOL failed at least one "
        "pre-registered criterion on gap-up days (date-clustered SEs, "
        "null-calibrated thresholds)")
    register(Experiment(
        name="exp09_premarket_rvol", family="intraday_microstructure",
        hypothesis="genuine premarket volume through 09:25, normalised to the "
                   "stock's own premarket norm, predicts open-to-close "
                   "continuation on gap-up days -- the signal the inherited "
                   "system meant to use before it used the day's total volume",
        features=["pm_rvol", "pm_gap_pct"], split="research+validation",
        universe="ADV $3M-2B, gap>=2% at 09:25, 2016-2024",
        rebalance_days=1, holding_days=1, n_observations=int(len(res) + len(val)),
        metrics={"t_clustered_research": float(t_primary),
                 "t_clustered_validation": float(t_val),
                 "t_rvol_given_gap": float(t_rvol_given_gap),
                 "mono_research": float(mono), "mono_validation": float(mono_v),
                 "d10_d1_research": float(spread_hi_lo)},
        outcome="HOLD" if ok else "FAIL", failure_reason=reason))
    logger.info("\n%s   K = %d", "ADVANCE" if ok else "REJECTED", trial_count())


if __name__ == "__main__":
    main()
