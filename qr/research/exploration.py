"""
Free exploration. Weirder hypotheses, looser structure.
=======================================================

The structured screen swept the standard factor zoo and found a cost floor.
This file goes looking somewhere else. The organising idea: at ~120bps a
trade, a good RANKING is worth little. What is worth something is

  (a) an AVOIDANCE rule -- long-only, no borrow needed, and it works by not
      owning disasters rather than by picking winners, so it can pay off
      without any turnover at all; or
  (b) a CONDITIONAL -- a signal that only fires in specific states, so it
      trades rarely and the cost floor is amortised over a bigger edge.

Hypotheses, each with a reason to exist rather than a curve to fit:

  H-MAX     Lottery preference. Investors overpay for names with recent
            extreme single-day gains (Bali/Cakici/Whitelaw). Those names then
            underperform. Long-only AVOIDANCE rule.
  H-52W     Proximity to the 52-week high (George & Hwang). An anchoring
            effect, distinct from raw momentum: traders under-react to news
            that pushes a stock near its high.
  H-DD      Deep-drawdown zombies. Names far below their own 52w high with
            falling volume are dying, not cheap. AVOIDANCE.
  H-TAX     Tax-loss selling. Small-cap losers are dumped for tax reasons in
            December and bounce in January. A calendar CONDITIONAL on
            reversal, which otherwise loses money.
  H-TOM     Turn-of-month. Documented flows around month boundaries.
  H-LEAD    Sympathy / lead-lag. When a name moves hard, does its correlated
            cohort follow the next day? Retail-scale, capacity-limited, and
            exactly the kind of thing large funds ignore.
  H-DISP    Dispersion regime. Cross-sectional signals should pay more when
            cross-sectional dispersion is high. A CONDITIONAL on everything.
  H-VOLDRY  Liquidity evaporation. Falling dollar volume as a leading marker
            of distress, independent of price.

Nothing here is promoted on the strength of one number. Anything interesting
goes to the same four hurdles that killed the last candidate.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger("explore")
HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
PROCESSED = HERE.parent / "data" / "processed"

COST_BPS = 60.0        # one-way, deliberately below the microcap estimate;
                       # this is exploration, and anything that fails even at
                       # 60bps is dead at 120.


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """All strictly-lagged. Wide-form per symbol, then stacked."""
    out = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        px = g["close"] * g["adj_factor"]
        r = px.pct_change(fill_method=None)

        g["ret_fwd_21"] = px.shift(-22) / px.shift(-1) - 1.0   # entry t+1, exit t+22

        # H-MAX: biggest single-day gain over the past month
        g["max_ret_21"] = r.rolling(21).max().shift(1)
        # H-52W
        hi52 = px.rolling(252, min_periods=120).max().shift(1)
        g["pct_of_52w_high"] = (px.shift(1) / hi52)
        # H-DD
        g["drawdown_52w"] = g["pct_of_52w_high"] - 1.0
        # H-VOLDRY: dollar volume trend
        dv = g["dollar_volume"]
        g["dv_trend"] = (dv.rolling(21).mean() / dv.rolling(126).mean()).shift(1)
        g["dv_level"] = dv.rolling(21).mean().shift(1)
        # generic
        g["ret_21"] = (px.shift(1) / px.shift(22) - 1.0)
        g["ret_252"] = (px.shift(1) / px.shift(253) - 1.0)
        g["vol_21"] = r.rolling(21).std().shift(1)
        g["price"] = g["close"].shift(1)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def decile_spread(df: pd.DataFrame, feat: str, fwd: str = "ret_fwd_21",
                  n_bins: int = 10, min_names: int = 100) -> pd.DataFrame:
    """Cross-sectional decile means of `fwd` sorted on `feat`, per date."""
    rows = []
    for d, g in df.groupby("date"):
        g = g[[feat, fwd]].dropna()
        if len(g) < min_names:
            continue
        try:
            q = pd.qcut(g[feat].rank(method="first"), n_bins, labels=False)
        except ValueError:
            continue
        m = g.groupby(q)[fwd].mean()
        rows.append(m.rename(d))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def report(name: str, dec: pd.DataFrame, hypothesis: str,
           direction: str = "low_minus_high") -> dict:
    """Summarise a decile table into a tradeable long-only statement."""
    if dec.empty or dec.shape[1] < 3:
        logger.info("  %-12s no data", name)
        return {"name": name, "status": "no_data"}

    lo, hi = dec.iloc[:, 0], dec.iloc[:, -1]
    spread = (lo - hi) if direction == "low_minus_high" else (hi - lo)
    # long-only versions: own the favoured decile, minus costs
    fav = lo if direction == "low_minus_high" else hi
    unfav = hi if direction == "low_minus_high" else lo
    net_fav = fav - COST_BPS / 1e4
    universe = dec.mean(axis=1)

    t, p = stats.ttest_1samp(spread.dropna(), 0)
    ppy = 252 / 21
    out = {
        "name": name, "hypothesis": hypothesis,
        "n_periods": int(len(dec)),
        "spread_bps_per_21d": float(spread.mean() * 1e4),
        "spread_t": float(t), "spread_p": float(p / 2 if t > 0 else 1 - p / 2),
        "favoured_decile_ret": float(fav.mean()),
        "avoided_decile_ret": float(unfav.mean()),
        "universe_ret": float(universe.mean()),
        "fav_minus_universe_bps": float((fav.mean() - universe.mean()) * 1e4),
        "long_only_net_vs_universe_bps": float(
            (net_fav.mean() - universe.mean()) * 1e4),
        "monotonic_r": float(np.corrcoef(np.arange(dec.shape[1]),
                                         dec.mean(axis=0).values)[0, 1]),
        "sharpe_spread": float(spread.mean() / spread.std(ddof=1) * np.sqrt(ppy))
                         if spread.std(ddof=1) > 0 else np.nan,
    }
    logger.info("  %-12s spread=%+7.0fbps/21d  t=%5.2f  mono_r=%+.2f  "
                "fav-univ=%+6.0fbps  net=%+6.0fbps",
                name, out["spread_bps_per_21d"], out["spread_t"],
                out["monotonic_r"], out["fav_minus_universe_bps"],
                out["long_only_net_vs_universe_bps"])
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    REPORTS.mkdir(exist_ok=True)

    panel = pd.read_parquet(PROCESSED / "panel_v1_frozen.parquet")
    # tradeable-ish universe, point in time; keeps the split junk out
    panel = panel[(panel["close"].between(2, 1000))]
    logger.info("panel: %d rows, %d symbols", len(panel), panel.symbol.nunique())

    df = build_features(panel)
    d = np.sort(df["date"].unique())
    d_test = pd.Timestamp(d[int(len(d) * 0.8)])
    dv = df[df["date"] < d_test]                      # TEST sealed
    dv = dv[dv["dv_level"] > 2e6]
    # sample monthly to keep the decile tables independent-ish
    month_ends = dv.groupby(dv["date"].dt.to_period("M"))["date"].max()
    dv = dv[dv["date"].isin(month_ends)]
    logger.info("dev+val: %d rows, %d dates\n", len(dv), dv.date.nunique())

    results = []

    logger.info("=== AVOIDANCE RULES (long only, no borrow needed) ===")
    results.append(report("H-MAX", decile_spread(dv, "max_ret_21"),
                          "lottery-like recent extreme gains underperform",
                          "low_minus_high"))
    results.append(report("H-DD", decile_spread(dv, "drawdown_52w"),
                          "deep drawdown = dying, not cheap", "high_minus_low"))
    results.append(report("H-VOLDRY", decile_spread(dv, "dv_trend"),
                          "liquidity evaporation precedes distress",
                          "high_minus_low"))
    results.append(report("H-VOL", decile_spread(dv, "vol_21"),
                          "high idiosyncratic vol underperforms",
                          "low_minus_high"))
    results.append(report("H-PRICE", decile_spread(dv, "price"),
                          "low nominal price underperforms", "high_minus_low"))

    logger.info("\n=== ANCHORING / MOMENTUM VARIANTS ===")
    results.append(report("H-52W", decile_spread(dv, "pct_of_52w_high"),
                          "proximity to 52w high (George-Hwang anchoring)",
                          "high_minus_low"))
    results.append(report("H-MOM12", decile_spread(dv, "ret_252"),
                          "12m momentum, for comparison", "high_minus_low"))

    logger.info("\n=== CALENDAR CONDITIONALS ===")
    dv2 = dv.copy()
    dv2["month"] = dv2["date"].dt.month
    for label, months, feat, direc in [
        ("H-TAX(dec)", [12], "ret_252", "low_minus_high"),
        ("H-TAX(jan)", [1], "ret_252", "low_minus_high"),
        ("H-TAX(rest)", list(range(2, 12)), "ret_252", "low_minus_high"),
    ]:
        sub = dv2[dv2["month"].isin(months)]
        results.append(report(label, decile_spread(sub, feat),
                              "tax-loss selling reversal in losers", direc))

    pd.DataFrame(results).to_csv(REPORTS / "exploration.csv", index=False)
    logger.info("\nwrote reports/exploration.csv")
    return pd.DataFrame(results)


if __name__ == "__main__":
    main()
