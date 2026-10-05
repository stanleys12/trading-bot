"""
EXPERIMENT 03 — net share issuance (dilution) in small caps.
=============================================================

RESEARCH SPLIT ONLY (2010-2017). Validation untouched. Test sealed.

HYPOTHESIS
Firms issue equity when management believes the stock is overvalued and
repurchase when undervalued. Net share issuance should therefore predict
returns NEGATIVELY: heavy diluters underperform, net repurchasers outperform.

    NSI(t) = log( shares_outstanding(t) / shares_outstanding(t - 12m) )
    signal = -NSI          (higher score = fewer shares issued)

WHY THIS ONE, AFTER 205 FAILURES
Four properties none of the previous candidates had together:

1. ORTHOGONAL. It is a corporate action, not a price or valuation transform.
   Everything rejected so far was derived from prices, volumes, or accounting
   ratios of the same prices. This is a different information source.

2. NATURALLY LOW TURNOVER. Share counts change on filings, so the signal
   changes quarterly at most. The cost floor -- which killed the 5-day
   reversal at 85x turnover -- is structurally not the binding constraint.

3. SMALL-CAP SPECIFIC BY CONSTRUCTION. Dilution is the defining financial
   behaviour of microcaps: ATMs, toxic converts, serial secondaries. If an
   effect should be concentrated where big money will not go, this is the
   most plausible candidate we have tested.

4. THE SURVIVORSHIP BIAS RUNS AGAINST IT. Every prior candidate was flattered
   by the missing 37% of the universe. Here it is the reverse: the firms that
   dilute hardest are the ones that go to zero and vanish from the panel, so
   the measured effect is a LOWER bound. That asymmetry is the single best
   reason to take a positive result here more seriously than the last six.

WHAT WOULD FALSIFY IT
- IC not reliably negative on raw NSI (i.e. the sign is wrong or absent).
- Effect NOT stronger in small/illiquid names -- that would mean it is a
  large-cap corporate-finance effect we have no edge in.
- Effect explained by beta, sector, or by value/profitability loadings.
- Effect concentrated in the contaminated 2010-2017 window only.

A NOTE ON THE SHARE-COUNT SERIES
`shares` comes from several XBRL tags with different meanings (period-end
outstanding vs weighted-average diluted). Mixing them creates fake jumps, so
we require the SAME tag on both endpoints of the 12-month change and drop any
observation where the implied change exceeds a factor of 4, which is almost
always a tag switch or a split we already handle elsewhere.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.backtesting.evaluation import alpha_vs_benchmarks
from qr.research.exp01_sector_neutral import portfolio
from qr.research.protocol import (Experiment, register, slice_split,
                                  split_start, trial_count)

logger = logging.getLogger("exp03")
HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
PROCESSED = HERE.parent / "data" / "processed"
INTERIM = HERE.parent / "data" / "interim"

SPLIT = "research"


def build_nsi(panel: pd.DataFrame) -> pd.DataFrame:
    """Net share issuance over the trailing 12 months, point-in-time."""
    out = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        px = g["true_close"] * g["adj_factor"]
        sh = g["shares"].replace(0, np.nan)

        # 12-month change in the reported share count. `shares` already comes
        # from an as-of join on filing date, so this is strictly PIT.
        sh_lag = sh.shift(252)
        ratio = sh / sh_lag
        # tag switches and unhandled splits show up as implausible jumps
        ratio = ratio.where((ratio > 0.25) & (ratio < 4.0))
        g["nsi_12m"] = np.log(ratio)
        g["signal"] = -g["nsi_12m"]              # fewer shares issued = better

        g["fwd_63"] = px.shift(-64) / px.shift(-1) - 1.0
        g["fwd_252"] = px.shift(-253) / px.shift(-1) - 1.0
        g["dv21"] = g["dollar_volume"].rolling(21).mean().shift(1)
        g["px"] = px
        out.append(g)
    return pd.concat(out, ignore_index=True)


def ic(df: pd.DataFrame, feat: str, fwd: str, min_names: int = 50) -> dict:
    vals = []
    for d, g in df.groupby("date"):
        g = g[[feat, fwd]].dropna()
        if len(g) < min_names:
            continue
        c = stats.spearmanr(g[feat], g[fwd]).statistic
        if np.isfinite(c):
            vals.append(c)
    if len(vals) < 8:
        return {"n": len(vals), "ic": np.nan, "t": np.nan}
    a = np.array(vals)
    return {"n": len(a), "ic": float(a.mean()),
            "t": float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))),
            "ir": float(a.mean() / a.std(ddof=1))}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    panel = pd.read_parquet(PROCESSED / "panel_v5_full.parquet")
    panel = slice_split(panel, SPLIT)
    panel = panel[panel["true_close"].between(2, 1000)]
    logger.info("SPLIT=%s  %d rows, %d symbols  (K so far = %d)",
                SPLIT, len(panel), panel.symbol.nunique(), trial_count())

    df = build_nsi(panel)
    df = df[(df["date"] >= split_start(SPLIT)) & (df["dv21"] > 1e6)]
    cov = df["signal"].notna().mean()
    logger.info("NSI coverage: %.0f%% of rows\n", cov * 100)

    q_ends = df.groupby(df["date"].dt.to_period("Q"))["date"].max()
    q = df[df["date"].isin(q_ends)].copy()
    q = q.dropna(subset=["signal"])
    logger.info("quarterly obs with NSI: %d rows, %d dates, %d symbols",
                len(q), q.date.nunique(), q.symbol.nunique())
    logger.info("NSI distribution: p10=%.3f  median=%.3f  p90=%.3f",
                q.nsi_12m.quantile(.1), q.nsi_12m.median(), q.nsi_12m.quantile(.9))
    logger.info("  share of names diluting >10%%/yr: %.0f%%\n",
                (q.nsi_12m > 0.095).mean() * 100)

    # ------------------------------------------------------------ core IC
    logger.info("=== IC OF -NSI (higher = fewer shares issued) ===")
    for fwd, lab in (("fwd_63", "1 quarter"), ("fwd_252", "1 year")):
        r = ic(q, "signal", fwd)
        logger.info("  vs %-10s IC=%+.4f  t=%5.2f  IR=%+.2f  n=%d",
                    lab, r["ic"], r["t"], r.get("ir", np.nan), r["n"])

    # ------------------------------- falsification 1: is it a small-cap effect?
    logger.info("\n=== BY LIQUIDITY QUINTILE (must be stronger in small) ===")
    q["liq_q"] = q.groupby("date")["dv21"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop"))
    rows = []
    for lq in range(5):
        sub = q[q.liq_q == lq]
        r = ic(sub, "signal", "fwd_252", min_names=30)
        med = sub.dv21.median()
        logger.info("  Q%d (ADV $%6.1fM)  IC=%+.4f  t=%5.2f  n=%d",
                    lq + 1, med / 1e6, r["ic"], r["t"], r["n"])
        rows.append({"liq_q": lq + 1, "adv": med, **r})

    # ------------------------------- falsification 2: sector, not stock?
    logger.info("\n=== SECTOR-NEUTRAL VERSION ===")
    qs = q[q["sector"].notna() & (q["sector"] != "Unknown")].copy()
    qs["signal_sn"] = qs["signal"] - qs.groupby(["date", "sector"])["signal"].transform("mean")
    for f, lab in (("signal", "raw"), ("signal_sn", "sector-neutral")):
        r = ic(qs, f, "fwd_252")
        logger.info("  %-16s IC=%+.4f  t=%5.2f", lab, r["ic"], r["t"])

    # ------------------------------- falsification 3: is it just value?
    logger.info("\n=== CONTROLLED FOR VALUE / PROFITABILITY ===")
    ctrl = q.dropna(subset=["signal", "fwd_252", "book_to_market", "roe"]).copy()
    resid_ic = []
    for d, g in ctrl.groupby("date"):
        if len(g) < 60:
            continue
        X = np.column_stack([np.ones(len(g)),
                             stats.zscore(g["book_to_market"].clip(
                                 *g["book_to_market"].quantile([.01, .99]))),
                             stats.zscore(g["roe"].clip(
                                 *g["roe"].quantile([.01, .99])))])
        y = stats.zscore(g["signal"].clip(*g["signal"].quantile([.01, .99])))
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r_ = y - X @ beta
            c = stats.spearmanr(r_, g["fwd_252"]).statistic
            if np.isfinite(c):
                resid_ic.append(c)
        except np.linalg.LinAlgError:
            continue
    if len(resid_ic) > 8:
        a = np.array(resid_ic)
        logger.info("  NSI orthogonalised to B/M and ROE: IC=%+.4f  t=%5.2f  n=%d",
                    a.mean(), a.mean() / (a.std(ddof=1) / np.sqrt(len(a))), len(a))

    # ------------------------------------------------- portfolio + alpha
    logger.info("\n=== PORTFOLIOS (quarterly rebalance, 30bps, daily-marked) ===")
    bench = pd.read_parquet(PROCESSED / "benchmarks.parquet")
    bench.index = pd.to_datetime(bench.index).tz_localize(None)
    bret = {c: bench[c].pct_change(fill_method=None) for c in bench.columns}

    pf_rows = []
    for label, sub in (("all names", q),
                       ("illiquid half", q[q.liq_q <= 1]),
                       ("liquid half", q[q.liq_q >= 3])):
        if sub.date.nunique() < 12:
            continue
        r = portfolio(df, sub, "signal", q=0.20)
        if len(r) < 200:
            logger.warning("  %-16s SKIPPED (%d obs)", label, len(r))
            continue
        eq = (1 + r.fillna(0)).cumprod()
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        a = alpha_vs_benchmarks(r, bret, 252)
        sp, tf = a.get("vs_SPY", {}), a.get("vs_SPY_IWM", {})
        logger.info("  %-16s CAGR=%+6.1f%%  SR=%5.2f  DD=%6.1f%%  "
                    "aSPY=%+5.2f%% (t=%4.2f)  a3F=%+5.2f%% (t=%4.2f)",
                    label, (eq.iloc[-1] ** (1 / yrs) - 1) * 100,
                    r.mean() / r.std(ddof=1) * np.sqrt(252),
                    ((eq - eq.cummax()) / eq.cummax()).min() * 100,
                    sp.get("alpha_annual", np.nan) * 100, sp.get("alpha_t", np.nan),
                    tf.get("alpha_annual", np.nan) * 100, tf.get("alpha_t", np.nan))
        pf_rows.append({"cohort": label, "cagr": eq.iloc[-1] ** (1 / yrs) - 1,
                        "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(252),
                        "alpha_3f": tf.get("alpha_annual"), "t_3f": tf.get("alpha_t")})

    pd.DataFrame(rows).to_csv(REPORTS / "exp03_by_liquidity.csv", index=False)
    pd.DataFrame(pf_rows).to_csv(REPORTS / "exp03_portfolios.csv", index=False)

    core = ic(q, "signal", "fwd_252")
    small = ic(q[q.liq_q <= 1], "signal", "fwd_252", min_names=30)
    large = ic(q[q.liq_q >= 3], "signal", "fwd_252", min_names=30)
    advance = (np.isfinite(core["t"]) and core["t"] > 2.0
               and np.isfinite(small["ic"]) and np.isfinite(large["ic"])
               and small["ic"] > large["ic"])
    reason = "" if advance else (
        f"IC t={core['t']:.2f} (need >2) or effect not concentrated in small caps "
        f"(illiquid IC={small['ic']:.4f} vs liquid IC={large['ic']:.4f}); a dilution "
        f"effect that is not stronger in microcaps is a large-cap corporate-finance "
        f"effect we have no edge in")
    register(Experiment(
        name="exp03_net_share_issuance", family="corporate_events",
        hypothesis="firms issue equity when overvalued; net share issuance "
                   "predicts returns negatively, most strongly in microcaps "
                   "where dilution is endemic",
        features=["nsi_12m"], split=SPLIT,
        universe="ADV>$1M, price $2-1000, 2010-2017",
        rebalance_days=63, holding_days=63, n_observations=int(q.date.nunique()),
        metrics={"ic_1y": core["ic"], "t_1y": core["t"],
                 "ic_illiquid": small["ic"], "ic_liquid": large["ic"]},
        outcome="HOLD" if advance else "FAIL", failure_reason=reason))
    logger.info("\n%s.  K = %d", "ADVANCE to validation" if advance else "REJECTED",
                trial_count())


if __name__ == "__main__":
    main()
