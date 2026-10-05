"""
Stress test on the one strong result: the low-price premium.

The exploration found low-priced stocks beating high-priced ones by ~327bps
per 21 days, t = -5.34, monotonic across deciles. That is the strongest
statistic produced anywhere in this project.

It is also, a priori, the result most likely to be pure survivorship artefact:
low-priced small caps are precisely the population that delists, and 37% of
our point-in-time universe has no recoverable price history. Every name that
went to zero and vanished would have sat in the low-price decile.

So rather than report it, we try to break it three ways.

T1  FATE SPLIT      Compute the effect separately for names that survived and
                    for the 416 delisted names we DID recover. If the dead
                    behave much worse, the 3,489 we cannot see would drag the
                    effect down.

T2  REWEIGHTING     Our sample contains dead names at ~7% when the true rate
                    is ~42%. Re-weight the dead up to their true frequency and
                    recompute. This is a crude correction and it assumes the
                    recovered dead resemble the unrecovered ones -- which is
                    itself optimistic, because the recoverable deaths skew
                    toward mergers and acquisitions rather than bankruptcies.

T3  TERMINAL DROP   What happens in a name's final months before it vanishes?
                    If the effect is driven by the survivors' rebounds while
                    the dying names are absent, the low-price decile's true
                    return is far worse than measured.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qr.research.exploration import build_features, decile_spread  # noqa: E402

logger = logging.getLogger("stress")
HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
PROCESSED = HERE.parent / "data" / "processed"
INTERIM = HERE.parent / "data" / "interim"


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    panel = pd.read_parquet(PROCESSED / "panel_v1_frozen.parquet")
    panel = panel[panel["close"].between(2, 1000)]
    memb = pd.read_parquet(INTERIM / "universe_membership.parquet")

    dead = set(memb[memb.delisted].symbol)
    n_dead_true = int(memb.delisted.sum())
    n_universe = int(len(memb))

    df = build_features(panel)
    d = np.sort(df["date"].unique())
    d_test = pd.Timestamp(d[int(len(d) * 0.8)])
    dv = df[(df["date"] < d_test) & (df["dv_level"] > 2e6)].copy()
    month_ends = dv.groupby(dv["date"].dt.to_period("M"))["date"].max()
    dv = dv[dv["date"].isin(month_ends)]
    dv["is_dead"] = dv["symbol"].isin(dead)

    logger.info("=== SAMPLE COMPOSITION ===")
    logger.info("  point-in-time universe   : %d symbols, %d delisted (%.1f%%)",
                n_universe, n_dead_true, n_dead_true / n_universe * 100)
    logger.info("  in this panel            : %d symbols, %d delisted (%.1f%%)",
                dv.symbol.nunique(), dv[dv.is_dead].symbol.nunique(),
                dv[dv.is_dead].symbol.nunique() / dv.symbol.nunique() * 100)
    logger.info("  -> the dead are under-represented by ~%.0fx\n",
                (n_dead_true / n_universe) /
                (dv[dv.is_dead].symbol.nunique() / dv.symbol.nunique()))

    # ------------------------------------------------------------- T1
    logger.info("=== T1  EFFECT BY EVENTUAL FATE ===")
    logger.info("  forward 21d return by price decile (1 = cheapest)")
    logger.info("  %-10s %>8s %>8s %>8s %>8s", "cohort", "D1", "D5", "D10", "D1-D10")
    res = {}
    for label, sub in (("survivors", dv[~dv.is_dead]), ("delisted", dv[dv.is_dead])):
        dec = decile_spread(sub, "price", min_names=40)
        if dec.empty or dec.shape[1] < 10:
            logger.info("  %-10s insufficient", label)
            continue
        d1, d5, d10 = dec.iloc[:, 0].mean(), dec.iloc[:, 4].mean(), dec.iloc[:, -1].mean()
        res[label] = {"d1": d1, "d10": d10, "spread": d1 - d10, "dec": dec}
        logger.info("  %-10s %+7.2f%% %+7.2f%% %+7.2f%% %+7.2f%%",
                    label, d1 * 100, d5 * 100, d10 * 100, (d1 - d10) * 100)

    if "survivors" in res and "delisted" in res:
        logger.info("\n  cheapest-decile 21d return:")
        logger.info("    survivors : %+.2f%%", res["survivors"]["d1"] * 100)
        logger.info("    delisted  : %+.2f%%", res["delisted"]["d1"] * 100)
        logger.info("    gap       : %+.2f%% per 21 days",
                    (res["delisted"]["d1"] - res["survivors"]["d1"]) * 100)

    # ------------------------------------------------------------- T2
    logger.info("\n=== T2  REWEIGHTED TO THE TRUE DELISTING RATE ===")
    if "survivors" in res and "delisted" in res:
        w_dead = n_dead_true / n_universe
        for k in ("d1", "d10"):
            corrected = (1 - w_dead) * res["survivors"][k] + w_dead * res["delisted"][k]
            observed = ((1 - 0.075) * res["survivors"][k] + 0.075 * res["delisted"][k])
            logger.info("  decile %-3s observed %+6.2f%%  ->  reweighted %+6.2f%%",
                        k.upper(), observed * 100, corrected * 100)
        s_obs = ((1 - 0.075) * res["survivors"]["d1"] + 0.075 * res["delisted"]["d1"]) - \
                ((1 - 0.075) * res["survivors"]["d10"] + 0.075 * res["delisted"]["d10"])
        s_cor = ((1 - w_dead) * res["survivors"]["d1"] + w_dead * res["delisted"]["d1"]) - \
                ((1 - w_dead) * res["survivors"]["d10"] + w_dead * res["delisted"]["d10"])
        logger.info("\n  low-minus-high spread:")
        logger.info("    as measured   : %+.0f bps / 21d", s_obs * 1e4)
        logger.info("    reweighted    : %+.0f bps / 21d", s_cor * 1e4)
        logger.info("    bias          : %+.0f bps  (%.0f%% of the effect)",
                    (s_obs - s_cor) * 1e4,
                    abs((s_obs - s_cor) / s_obs) * 100 if s_obs else np.nan)
        logger.info("\n  NOTE: this correction is OPTIMISTIC. The delisted names we")
        logger.info("  can price skew toward mergers/acquisitions, which end at a")
        logger.info("  premium. The ones we cannot price are the bankruptcies.")

    # ------------------------------------------------------------- T3
    logger.info("\n=== T3  WHAT HAPPENS JUST BEFORE A NAME VANISHES ===")
    last_seen = panel[panel.symbol.isin(dead)].groupby("symbol")["date"].max()
    rows = []
    for sym, last in last_seen.items():
        g = df[(df.symbol == sym)].sort_values("date")
        if len(g) < 200:
            continue
        tail = g[g["date"] >= last - pd.Timedelta(days=180)]
        if len(tail) < 20:
            continue
        px = tail["close"] * tail["adj_factor"]
        rows.append({"symbol": sym,
                     "final_6m_ret": px.iloc[-1] / px.iloc[0] - 1,
                     "price_at_start": tail["close"].iloc[0]})
    t3 = pd.DataFrame(rows)
    if len(t3):
        t3["cheap"] = t3["price_at_start"] < t3["price_at_start"].median()
        logger.info("  n = %d delisted names with a full final 6 months", len(t3))
        logger.info("  median final-6m return, all      : %+.1f%%",
                    t3.final_6m_ret.median() * 100)
        logger.info("  median final-6m return, cheap    : %+.1f%%",
                    t3[t3.cheap].final_6m_ret.median() * 100)
        logger.info("  median final-6m return, expensive: %+.1f%%",
                    t3[~t3.cheap].final_6m_ret.median() * 100)
        logger.info("  %% of cheap names down >50%%        : %.0f%%",
                    (t3[t3.cheap].final_6m_ret < -0.5).mean() * 100)
        logger.info("\n  and these are only the deaths we can SEE. The 3,489 we")
        logger.info("  cannot price are the ones that ended at zero.")


if __name__ == "__main__":
    main()
