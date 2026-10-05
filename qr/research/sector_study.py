"""
Sector-relative and size-conditioned search.
============================================

Systematic pattern hunting, but with a prior on where to spend the search
budget. Every hypothesis below names a reason it might work; that is not
purity, it is statistical power. 110 unmotivated tests means the winner must
clear a Bonferroni bar of 0.0005. Twelve motivated ones means 0.004.

H1  SECTOR-NEUTRAL vs RAW
    Every "relative strength" signal so far compared a biotech to a regional
    bank. Most of a small-cap's move is its sector's move. Demeaning within
    sector isolates the idiosyncratic part, which is the part that carries
    firm-specific information. If our signals were drowning in sector beta,
    this recovers them; if they were empty, it changes nothing.

H2  SIZE CONDITIONING
    The whole thesis is that inefficiency lives where big money will not go.
    That is a testable claim, not an assumption: if true, signals should get
    STRONGER as you move down the liquidity distribution, and the gradient
    should be visible. If signal strength is flat across size buckets, the
    thesis is wrong and small caps are just expensive to trade.

H3  INTRA-SECTOR LEAD-LAG          <- the one I actually believe in
    Hou (2007), Lo-MacKinlay: information diffuses gradually within an
    industry. Large, heavily-covered firms price news first; small,
    under-covered firms in the same industry follow with a lag. The mechanism
    is limited analyst attention, which is EXACTLY a small-cap condition, and
    the capacity is far too small for a large fund to bother with -- which is
    why it can persist.
    Concretely: today's return of the big names in a sector should predict
    tomorrow's return of the small names in that same sector.

H4  SECTOR-BASKET MOMENTUM
    Cost has killed everything so far because we trade 200+ illiquid names.
    Trading ~11 sector baskets inverts that: few instruments, each an average
    of many names, far lower turnover per dollar. Worth knowing whether the
    signal survives when the cost structure is favourable, even if the
    implementation would need ETFs rather than baskets.

H5  DISPERSION CONDITIONING
    Cross-sectional signals should pay more when cross-sectional dispersion is
    high, because there is more to separate. A conditional that trades rarely
    amortises the cost floor over a bigger edge.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger("sector")
HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
PROCESSED = HERE.parent / "data" / "processed"

COST_BPS = 60.0


def prep(panel: pd.DataFrame) -> pd.DataFrame:
    """Monthly-sampled feature frame with sector labels."""
    out = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        px = g["true_close"] * g["adj_factor"]
        r = px.pct_change(fill_method=None)

        g["ret_fwd_21"] = px.shift(-22) / px.shift(-1) - 1.0
        g["ret_fwd_5"] = px.shift(-6) / px.shift(-1) - 1.0
        g["ret_fwd_1"] = px.shift(-2) / px.shift(-1) - 1.0

        g["mom_252"] = px.shift(1) / px.shift(253) - 1.0
        g["mom_21"] = px.shift(1) / px.shift(22) - 1.0
        g["rev_5"] = -(px.shift(1) / px.shift(6) - 1.0)
        g["vol_63"] = -r.rolling(63).std().shift(1)
        g["ret_1d"] = r.shift(1)
        g["ret_5d"] = px.shift(1) / px.shift(6) - 1.0
        g["dv21"] = g["dollar_volume"].rolling(21).mean().shift(1)
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    return df


def demean_by(df: pd.DataFrame, cols: list[str], by: list[str]) -> pd.DataFrame:
    """Subtract the group mean -- the sector-neutralising operation."""
    d = df.copy()
    for c in cols:
        d[c + "_sn"] = d[c] - d.groupby(by)[c].transform("mean")
    return d


def ic(df: pd.DataFrame, feat: str, fwd: str, by_date: str = "date") -> dict:
    """Rank information coefficient: per-date Spearman(feature, forward return)."""
    rows = []
    for d, g in df.groupby(by_date):
        g = g[[feat, fwd]].dropna()
        if len(g) < 50:
            continue
        c = stats.spearmanr(g[feat], g[fwd]).statistic
        if np.isfinite(c):
            rows.append(c)
    if len(rows) < 8:
        return {"n": len(rows), "ic": np.nan, "t": np.nan}
    a = np.array(rows)
    t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))
    return {"n": len(a), "ic": float(a.mean()), "t": float(t),
            "ir": float(a.mean() / a.std(ddof=1)) if a.std(ddof=1) else np.nan}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    panel = pd.read_parquet(PROCESSED / "panel_v3_sector.parquet")
    panel = panel[panel["true_close"].between(2, 1000)]
    logger.info("panel %d rows, %d symbols, %d sectors",
                len(panel), panel.symbol.nunique(), panel.sector.nunique())

    df = prep(panel)
    d = np.sort(df["date"].unique())
    d_test = pd.Timestamp(d[int(len(d) * 0.8)])
    dv = df[(df["date"] < d_test) & (df["dv21"] > 2e6)
            & (df["sector"] != "Unknown")].copy()

    month_ends = dv.groupby(dv["date"].dt.to_period("M"))["date"].max()
    mon = dv[dv["date"].isin(month_ends)].copy()
    logger.info("dev+val: %d rows, %d dates, %d sectors\n",
                len(mon), mon.date.nunique(), mon.sector.nunique())

    SIGS = ["mom_252", "mom_21", "rev_5", "vol_63"]
    mon = demean_by(mon, SIGS, ["date", "sector"])
    results = []

    # ------------------------------------------------------------- H1
    logger.info("=== H1  SECTOR-NEUTRAL vs RAW  (IC on 21d forward return) ===")
    logger.info("  %-12s %>18s %>18s", "signal", "raw", "sector-neutral")
    for s in SIGS:
        a, b = ic(mon, s, "ret_fwd_21"), ic(mon, s + "_sn", "ret_fwd_21")
        logger.info("  %-12s  IC=%+.4f t=%5.2f   IC=%+.4f t=%5.2f",
                    s, a["ic"], a["t"], b["ic"], b["t"])
        results += [{"test": "H1", "signal": s, "variant": "raw", **a},
                    {"test": "H1", "signal": s, "variant": "sector_neutral", **b}]

    # ------------------------------------------------------------- H2
    logger.info("\n=== H2  SIZE CONDITIONING (IC by liquidity quintile) ===")
    mon["liq_q"] = mon.groupby("date")["dv21"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    for s in ["mom_252_sn", "rev_5_sn", "vol_63_sn"]:
        line = []
        for q in range(5):
            r = ic(mon[mon.liq_q == q], s, "ret_fwd_21")
            line.append(f"Q{q+1}:{r['ic']:+.3f}")
            results.append({"test": "H2", "signal": s, "variant": f"liq_q{q+1}", **r})
        logger.info("  %-14s %s   (Q1=least liquid)", s, "  ".join(line))

    # ------------------------------------------------------------- H3
    logger.info("\n=== H3  INTRA-SECTOR LEAD-LAG ===")
    logger.info("  do the BIG names in a sector predict the SMALL ones next period?")
    dd = dv.copy()
    dd["liq_q"] = dd.groupby("date")["dv21"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop"))
    big = dd[dd.liq_q == 4]
    for lead_col, fwd_col, label in [("ret_1d", "ret_fwd_1", "1d leader -> 1d follower"),
                                     ("ret_5d", "ret_fwd_5", "5d leader -> 5d follower")]:
        lead = (big.groupby(["date", "sector"])[lead_col].mean()
                .rename("sector_leader").reset_index())
        small = dd[dd.liq_q <= 1].merge(lead, on=["date", "sector"], how="left")
        r = ic(small, "sector_leader", fwd_col)
        logger.info("  %-28s IC=%+.4f  t=%5.2f  n=%d", label, r["ic"], r["t"], r["n"])
        results.append({"test": "H3", "signal": label, "variant": "lead_lag", **r})

    # ------------------------------------------------------------- H4
    logger.info("\n=== H4  SECTOR-BASKET MOMENTUM (11 instruments, low cost) ===")
    sec = (dv.groupby(["date", "sector"])
           .apply(lambda g: pd.Series({
               "ret_fwd_21": g["ret_fwd_21"].mean(),
               "mom_252": g["mom_252"].mean(),
               "mom_21": g["mom_21"].mean()}), include_groups=False)
           .reset_index())
    sec_m = sec[sec["date"].isin(month_ends)]
    for s in ("mom_252", "mom_21"):
        rows = []
        for d_, g in sec_m.groupby("date"):
            g = g[[s, "ret_fwd_21"]].dropna()
            if len(g) < 6:
                continue
            rows.append(stats.spearmanr(g[s], g["ret_fwd_21"]).statistic)
        rows = [x for x in rows if np.isfinite(x)]
        if len(rows) > 8:
            a = np.array(rows)
            t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))
            logger.info("  sector %-10s IC=%+.4f  t=%5.2f  n=%d", s, a.mean(), t, len(a))
            results.append({"test": "H4", "signal": f"sector_{s}",
                            "variant": "basket", "n": len(a),
                            "ic": float(a.mean()), "t": float(t)})

    # ------------------------------------------------------------- H5
    logger.info("\n=== H5  DISPERSION CONDITIONING ===")
    disp = mon.groupby("date")["ret_fwd_21"].std().rename("disp")
    mon2 = mon.merge(disp, on="date")
    hi = mon2[mon2.disp > mon2.disp.median()]
    lo = mon2[mon2.disp <= mon2.disp.median()]
    for s in ["mom_252_sn", "rev_5_sn"]:
        a, b = ic(hi, s, "ret_fwd_21"), ic(lo, s, "ret_fwd_21")
        logger.info("  %-14s high-dispersion IC=%+.4f (t=%.2f) | low IC=%+.4f (t=%.2f)",
                    s, a["ic"], a["t"], b["ic"], b["t"])
        results += [{"test": "H5", "signal": s, "variant": "high_disp", **a},
                    {"test": "H5", "signal": s, "variant": "low_disp", **b}]

    out = pd.DataFrame(results)
    out.to_csv(REPORTS / "sector_study.csv", index=False)
    logger.info("\n=== ANYTHING WITH |t| > 3 ? ===")
    strong = out[out["t"].abs() > 3].sort_values("t", key=abs, ascending=False)
    logger.info("\n%s", strong.to_string(index=False) if len(strong) else "  nothing")
    return out


if __name__ == "__main__":
    main()
