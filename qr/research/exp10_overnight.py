"""
exp10: Overnight vs intraday return decomposition on liquid ETFs.
=================================================================

HYPOTHESIS
    Equity index returns accrue disproportionately overnight (close -> open),
    while intraday (open -> close) earns roughly nothing. Documented since
    Cliff/Cooper/Gulen (2008) and Lou-Polk-Skouras (2019, JFE): the overnight
    anomaly. Economic story: (a) news, including earnings and macro, releases
    outside RTH; (b) retail order flow concentrates at the open; (c) market
    makers demand an overnight inventory premium.

WHY THIS IS NOT ANOTHER DEAD CROSS-SECTIONAL SCREEN
    Every prior null in this project was cross-sectional selection paying
    the spread. This is a TIMING decomposition on the single most liquid
    instrument in the world, implementable with opening/closing AUCTION
    orders (MOO/MOC). Auction executions are single-price crosses: no
    bid-ask spread is paid. The cost wall that killed 212 trials does not
    apply in the same form. Residual costs: SEC fee on sells (~0.28bp),
    auction slippage (sensitivity-tested 0/1/2 bp per side), zero commission
    at Alpaca.

INFORMATION CUTOFF
    None needed: the "signal" is the calendar. Buy every close, sell every
    open (or the reverse). Nothing about the future is used.

DATA QUALITY GUARD
    Yahoo daily opens are known to be unreliable in early years for some
    symbols (stale opens equal to prior close). We measure the fraction of
    exact stale opens per year and the overnight share of total variance;
    a year with >5% stale opens is excluded from scoring.

SPLITS (protocol.py)
    research    2010-2017  explore
    validation  2018-2022  confirm
    test        2023+      NOT TOUCHED

Pre-registered promotion criteria (set before looking at validation):
    - overnight mean > 0 in validation with t >= 2.0
    - overnight Sharpe (net, 1bp/side slippage) >= buy-and-hold Sharpe
      in BOTH research and validation
    - sign agreement research/validation
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"

SYMBOLS = ["SPY", "QQQ", "IWM"]
SEC_FEE = 0.0000278          # on sale proceeds, i.e. once per day for either leg
SLIPPAGE_BPS = [0.0, 1.0, 2.0]   # per side, auction slippage sensitivity

RESEARCH = ("2010-01-01", "2017-12-31")
VALIDATION = ("2018-01-01", "2022-12-31")


def load_etf() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "etf_panel.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


def decompose(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Per-day intraday / overnight / total(with divs) returns for one symbol."""
    s = (df[df["symbol"] == symbol]
         .sort_values("date")
         .set_index("date")[["open", "close", "adj_close"]]
         .dropna())
    out = pd.DataFrame(index=s.index)
    # total return close->close including dividends
    out["total"] = s["adj_close"].pct_change()
    # intraday: open->close, raw prices (no corporate action lands intraday)
    out["intraday"] = s["close"] / s["open"] - 1.0
    # overnight is what's left; dividends go ex at the open, so they belong here
    out["overnight"] = (1.0 + out["total"]) / (1.0 + out["intraday"]) - 1.0
    out["stale_open"] = (s["open"] == s["close"].shift(1))
    return out.dropna()


def year_quality(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(d.index.year)
    q = pd.DataFrame({
        "stale_frac": g["stale_open"].mean(),
        "on_var_share": g.apply(
            lambda x: x["overnight"].var() / max(x["total"].var(), 1e-12),
            include_groups=False),
    })
    return q


def _ann(mean_d: float, std_d: float) -> tuple[float, float]:
    return mean_d * 252, std_d * np.sqrt(252)


def stats_block(r: pd.Series) -> dict:
    n = len(r)
    mu, sd = r.mean(), r.std()
    t = mu / (sd / np.sqrt(n)) if sd > 0 else np.nan
    ann_mu, ann_sd = _ann(mu, sd)
    return {"n": n, "ann_ret": ann_mu, "ann_vol": ann_sd,
            "sharpe": ann_mu / ann_sd if ann_sd > 0 else np.nan, "t": t}


def net_overnight(r: pd.Series, slip_bps: float) -> pd.Series:
    """Overnight leg net of auction slippage both sides + SEC fee on the sell."""
    per_day = 2 * slip_bps / 1e4 + SEC_FEE
    return r - per_day


def window(d: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    return d[(d.index >= lo) & (d.index <= hi)]


def main() -> None:
    df = load_etf()
    verdicts = {}
    for sym in SYMBOLS:
        d = decompose(df, sym)
        q = year_quality(d)
        bad_years = q[q["stale_frac"] > 0.05].index.tolist()
        d = d[~d.index.year.isin(bad_years)]

        print(f"\n=== {sym} ===  (excluded stale-open years: {bad_years or 'none'})")
        res = window(d, *RESEARCH)
        val = window(d, *VALIDATION)

        rows = []
        for name, w in [("research", res), ("validation", val)]:
            for leg in ["overnight", "intraday", "total"]:
                s = stats_block(w[leg])
                rows.append({"window": name, "leg": leg, **s})
            # net overnight at each slippage level
            for slip in SLIPPAGE_BPS:
                s = stats_block(net_overnight(w["overnight"], slip))
                rows.append({"window": name, "leg": f"overnight_net_{slip:.0f}bp", **s})
        tab = pd.DataFrame(rows)
        with pd.option_context("display.float_format", "{:8.3f}".format,
                               "display.width", 120):
            print(tab.to_string(index=False))

        # pre-registered criteria
        vn = stats_block(net_overnight(val["overnight"], 1.0))
        rn = stats_block(net_overnight(res["overnight"], 1.0))
        vb = stats_block(val["total"])
        rb = stats_block(res["total"])
        crit = {
            "val_t_ge_2": bool(stats_block(val["overnight"])["t"] >= 2.0),
            "net_sharpe_beats_bh_research": bool(rn["sharpe"] >= rb["sharpe"]),
            "net_sharpe_beats_bh_validation": bool(vn["sharpe"] >= vb["sharpe"]),
            "sign_agreement": bool(np.sign(rn["ann_ret"]) == np.sign(vn["ann_ret"])),
        }
        passed = all(crit.values())
        verdicts[sym] = (passed, crit, rn, vn)
        print(f"criteria: {crit}  ->  {'PASS' if passed else 'FAIL'}")

    any_pass = any(v[0] for v in verdicts.values())
    for sym, (passed, crit, rn, vn) in verdicts.items():
        register(Experiment(
            name=f"exp10_overnight_{sym}",
            family="intraday_timing",
            hypothesis="index returns accrue overnight; implementable at ~zero "
                       "spread via MOO/MOC auction orders (Lou-Polk-Skouras 2019)",
            features=["overnight_return"],
            split="research+validation",
            universe=sym,
            rebalance_days=1, holding_days=1,
            n_observations=vn["n"],
            metrics={"research_net_sharpe_1bp": round(rn["sharpe"], 3),
                     "validation_net_sharpe_1bp": round(vn["sharpe"], 3),
                     "validation_net_ann_ret_1bp": round(vn["ann_ret"], 4),
                     **{k: bool(v) for k, v in crit.items()}},
            outcome="PASS" if passed else "FAIL",
            failure_reason="" if passed else
            "failed pre-registered criterion(s): "
            + ", ".join(k for k, v in crit.items() if not v),
        ))
    print(f"\nregistered {len(verdicts)} trials; any PASS: {any_pass}")


if __name__ == "__main__":
    main()
