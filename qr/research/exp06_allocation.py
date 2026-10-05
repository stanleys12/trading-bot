"""
EXPERIMENT 06 — does portfolio construction beat buy-and-hold risk-adjusted?
============================================================================

A different claim from everything before it. We are no longer asserting that
we can predict which asset will outperform. We are asserting that HOW you size
exposure changes risk-adjusted outcomes independently of what you pick. That
is a weaker claim, which is exactly why it has a chance of being true.

SPLITS (ETF history is longer, so these differ from the equity phase)
    RESEARCH    2007-04 .. 2014-12    includes the GFC
    VALIDATION  2015-01 .. 2020-12    includes 2018 vol shock and COVID
    TEST        2021-01 .. 2026-08    includes the 2022 bear and rate shock
Each window contains at least one crisis on purpose: a vol-targeting or trend
system whose entire value proposition is crisis behaviour must be tested on
crises, and a split that quarantined 2008 into one window would be useless.

PRE-REGISTERED SUCCESS CRITERIA -- fixed before any result is seen
  1. Sharpe > SPY buy-and-hold Sharpe in the SAME window, AND
  2. positive excess return at MATCHED VOLATILITY, with financing charged, AND
  3. holds in research AND validation (not one or the other), AND
  4. survives 3x the assumed transaction cost, AND
  5. is not driven by a single crisis -- must also hold with 2008 excluded.

Criterion 2 is the one that matters. Vol targeting reduces volatility by
construction, so raw return comparisons are meaningless and even Sharpe can
flatter a strategy that is simply less invested. Levering to the benchmark's
volatility and charging financing is the version that cannot be gamed.

PRE-REGISTERED SUSPICIONS
  - Vol targeting may improve Sharpe purely by de-risking into the GFC. If
    criterion 5 fails, that is what happened.
  - Trend following had a poor decade 2010-2019. If it works only in research
    (which contains 2008), it is a crisis hedge, not an edge.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register, trial_count
from qr.strategies.allocation import (AllocConfig, buy_hold, risk_parity,
                                      tsmom, tsmom_rp, vol_matched_return,
                                      vol_target)

logger = logging.getLogger("exp06")
HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
PROCESSED = HERE.parent / "data" / "processed"

# USMV excluded: launched 2011, and dropping it buys us the GFC.
ASSETS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG",
          "GLD", "SLV", "DBC", "VNQ", "XLP", "XLU"]
CASH = "SHY"

SPLITS = {
    "research":   ("2007-04-16", "2014-12-31"),
    "validation": ("2015-01-01", "2020-12-31"),
    "test":       ("2021-01-01", "2026-08-11"),
}


def stats(r: pd.Series, label: str = "") -> dict:
    r = r.dropna()
    if len(r) < 100:
        return {"label": label, "n": len(r)}
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    downside = r[r < 0].std(ddof=1)
    return {
        "label": label, "n": len(r),
        "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
        "vol": float(r.std(ddof=1) * np.sqrt(252)),
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)),
        "sortino": float(r.mean() / downside * np.sqrt(252)) if downside > 0 else np.nan,
        "maxdd": float(dd),
        "calmar": float((eq.iloc[-1] ** (1 / yrs) - 1) / abs(dd)) if dd else np.nan,
        "worst_year": float(r.resample("YE").apply(lambda x: (1 + x).prod() - 1).min()),
        "pct_pos_months": float((r.resample("ME").apply(lambda x: (1 + x).prod() - 1) > 0).mean()),
    }


def run_all(rets: pd.DataFrame, cfg: AllocConfig, cash: pd.Series):
    out = {}
    out["buy_hold_SPY"] = buy_hold(rets, "SPY", cfg, cash)
    out["vol_target_SPY"] = vol_target(rets, "SPY", cfg, cash)
    out["tsmom"] = tsmom(rets, ASSETS, cfg, cash)
    out["risk_parity"] = risk_parity(rets, ASSETS, cfg, cash)
    out["tsmom_rp"] = tsmom_rp(rets, ASSETS, cfg, cash)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rets = pd.read_parquet(PROCESSED / "etf_returns.parquet")
    rets = rets[[c for c in ASSETS + [CASH] if c in rets.columns]]
    rets = rets.loc["2007-04-16":]
    cash = rets[CASH].fillna(0)
    logger.info("universe: %d assets, %s..%s, %d days  (K = %d)",
                len(ASSETS), rets.index[0].date(), rets.index[-1].date(),
                len(rets), trial_count())

    cfg = AllocConfig()
    rows, series = [], {}

    for split, (lo, hi) in SPLITS.items():
        if split == "test":
            continue                     # SEALED
        sub = rets.loc[lo:hi]
        c = cash.loc[lo:hi]
        res = run_all(sub, cfg, c)
        bh = res["buy_hold_SPY"][0]
        logger.info("\n=== %s  (%s .. %s) ===", split.upper(), lo, hi)
        logger.info("  %-16s %7s %7s %7s %8s %8s %8s %9s", "strategy", "CAGR",
                    "vol", "Sharpe", "maxDD", "Calmar", "turn/yr", "vs BH vol-matched")
        for name, (r, turn) in res.items():
            s = stats(r, name)
            if "sharpe" not in s:
                continue
            vm = vol_matched_return(r, bh) if name != "buy_hold_SPY" else {}
            tpy = float(turn.sum() / ((sub.index[-1] - sub.index[0]).days / 365.25))
            logger.info("  %-16s %6.1f%% %6.1f%% %7.2f %7.1f%% %8.2f %8.1f %+8.2f%%",
                        name, s["cagr"] * 100, s["vol"] * 100, s["sharpe"],
                        s["maxdd"] * 100, s["calmar"], tpy,
                        vm.get("excess_at_matched_vol", np.nan) * 100
                        if vm else np.nan)
            rows.append({"split": split, "strategy": name, "turnover": tpy,
                         **s, **{f"vm_{k}": v for k, v in vm.items()}})
            series[(split, name)] = r

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "exp06_allocation.csv", index=False)

    # ---------------------------------------------- criterion 5: crisis-driven?
    logger.info("\n=== CRITERION 5: IS IT JUST THE GFC? (research, 2008 excluded) ===")
    sub = rets.loc["2007-04-16":"2014-12-31"]
    mask = ~((sub.index >= "2008-01-01") & (sub.index <= "2009-06-30"))
    res = run_all(sub, cfg, cash.loc["2007-04-16":"2014-12-31"])
    bh_ex = res["buy_hold_SPY"][0][mask]
    for name, (r, _) in res.items():
        s = stats(r[mask], name)
        if "sharpe" not in s:
            continue
        vm = vol_matched_return(r[mask], bh_ex) if name != "buy_hold_SPY" else {}
        logger.info("  %-16s Sharpe=%5.2f   vs BH vol-matched %+6.2f%%",
                    name, s["sharpe"], vm.get("excess_at_matched_vol", np.nan) * 100
                    if vm else np.nan)

    # ---------------------------------------------- criterion 4: cost stress
    logger.info("\n=== CRITERION 4: 3x TRANSACTION COSTS ===")
    for split, (lo, hi) in [("research", SPLITS["research"]),
                            ("validation", SPLITS["validation"])]:
        sub = rets.loc[lo:hi]
        res3 = run_all(sub, replace(cfg, cost_bps=9.0), cash.loc[lo:hi])
        bh3 = res3["buy_hold_SPY"][0]
        line = []
        for name, (r, _) in res3.items():
            if name == "buy_hold_SPY":
                continue
            vm = vol_matched_return(r, bh3)
            line.append(f"{name}={vm.get('excess_at_matched_vol', np.nan)*100:+.2f}%")
        logger.info("  %-12s %s", split, "  ".join(line))

    # ------------------------------------------------------------- verdict
    logger.info("\n=== PRE-REGISTERED CRITERIA ===")
    winners = []
    for name in ("vol_target_SPY", "tsmom", "risk_parity", "tsmom_rp"):
        r_ = df[df.strategy == name]
        if len(r_) < 2:
            continue
        bh_sh = {s: df[(df.split == s) & (df.strategy == "buy_hold_SPY")].sharpe.iloc[0]
                 for s in ("research", "validation")}
        c1 = all(r_[r_.split == s].sharpe.iloc[0] > bh_sh[s] for s in ("research", "validation"))
        c2 = all(r_[r_.split == s].vm_excess_at_matched_vol.iloc[0] > 0
                 for s in ("research", "validation"))
        ok = c1 and c2
        logger.info("  %-16s Sharpe>BH both splits: %-5s | vol-matched excess>0 both: %-5s | %s",
                    name, c1, c2, "PASS" if ok else "fail")
        if ok:
            winners.append(name)

    reason = "" if winners else (
        "no allocation rule beat SPY buy-and-hold on BOTH Sharpe and "
        "vol-matched excess return in BOTH research and validation windows")
    register(Experiment(
        name="exp06_allocation_rules", family="portfolio_construction",
        hypothesis="position sizing (vol targeting, trend gating, risk parity) "
                   "improves risk-adjusted return independently of asset selection",
        features=["realised_vol", "trend_12m"], split="research+validation",
        universe=f"{len(ASSETS)} liquid ETFs, 2007-2020",
        rebalance_days=cfg.rebalance_days, holding_days=cfg.rebalance_days,
        n_observations=int(len(rets.loc[:"2020-12-31"])),
        metrics={"winners": ",".join(winners) if winners else "none"},
        outcome="HOLD" if winners else "FAIL", failure_reason=reason))
    logger.info("\n%s  K = %d",
                f"ADVANCE: {winners}" if winners else "ALL REJECTED", trial_count())
    return df


if __name__ == "__main__":
    main()
