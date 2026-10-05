"""
exp14: Time-series momentum on BTC/ETH, long-or-flat.
=====================================================

HYPOTHESIS
    Crypto is retail-dominated, herding-prone, and lacks the institutional
    arbitrage capital that flattened equity anomalies; trend-following on the
    majors is the most documented effect in the asset class (Liu-Tsyvinski
    2021 RFS: 1-4 week momentum). Long-or-flat TSMOM should retain most
    upside while sitting out the -75% bear legs.

WHY THIS MARKET IS NEW TERRAIN FOR THIS PROJECT
    The 212-trial null is about US equity DAILY data: institutionally
    arbitraged, spread-bound in exactly the names where alpha lives. Crypto
    majors invert both constraints: taker cost on BTC/ETH is ~10-12bps TOTAL
    per side (less than half the small-cap HALF-spread), the market never
    closes, and the marginal participant is a retail trend-chaser.

DATA
    Yahoo daily closes for BTC-USD (2014->), ETH-USD (2017->). Yahoo crypto
    is survivor-clean for THIS test: the pair choice (top-2 by cap, already
    top-2 in 2016) is defensible point-in-time; no delisting censoring for
    a fixed-pair time-series test. A full cross-sectional build must use
    data.binance.vision (delisted pairs included) — verified accessible.

SPEC (fixed before scoring)
    Signal      sign of trailing 30-day return, evaluated weekly (every 7
                calendar days); long 1.0 if positive, else flat (0), per coin
    Portfolio   equal-weight across coins with data
    Cutoff      signal from close D, position from close D+1 (one full day
                of lag — conservative even for a 24/7 market)
    Costs       11 bps per side per flip (Binance taker 10bp + ~1bp spread)
    Annualise   365 days (crypto trades every day)

SPLITS
    research    2015-01-01 .. 2020-12-31   (one full halving cycle)
    validation  2021-01-01 .. 2023-12-31   (full bull-bear-chop sequence)
    test        2024-01-01 ..              SEALED, not scored here, ever

Pre-registered promotion criteria:
    - net Sharpe >= buy-and-hold Sharpe in BOTH windows
    - net MaxDD shallower than buy-and-hold in BOTH windows
    - validation net annual return > 0
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register  # noqa: E402

INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim"

COINS = ["BTC-USD", "ETH-USD"]
LOOKBACK = 30
REBAL_DAYS = 7
COST_SIDE = 11e-4
RESEARCH = ("2015-01-01", "2020-12-31")
VALIDATION = ("2021-01-01", "2023-12-31")


def load() -> pd.DataFrame:
    cache = INTERIM / "crypto_majors_daily.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    px = {}
    for c in COINS:
        h = yf.Ticker(c).history(start="2014-01-01", auto_adjust=False)
        h.index = pd.to_datetime(h.index).tz_localize(None)
        px[c] = h["Close"]
    df = pd.DataFrame(px)
    df.to_parquet(cache)
    return df


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    n, mu, sd = len(r), r.mean(), r.std()
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    ann = mu * 365
    vol = sd * np.sqrt(365)
    return {"n": n, "ann_ret": ann, "ann_vol": vol,
            "sharpe": ann / vol if vol > 0 else np.nan, "maxdd": dd,
            "t": mu / (sd / np.sqrt(n)) if sd > 0 else np.nan}


def main() -> None:
    px = load()
    ret = px.pct_change()

    # weekly-evaluated signal: sign of trailing 30d return at close D,
    # held constant for the next 7 days, applied from close D+1
    mom = px / px.shift(LOOKBACK) - 1.0
    on_rebal = pd.Series(False, index=px.index)
    on_rebal.iloc[::REBAL_DAYS] = True
    sig = (mom > 0).where(on_rebal).ffill().astype(float)
    pos = sig.shift(1)                      # close D+1 execution lag

    flips = pos.diff().abs()
    strat_by_coin = pos * ret - flips * COST_SIDE
    # equal weight across coins that have data that day
    strat = strat_by_coin.mean(axis=1)
    bh = ret.mean(axis=1)

    results = {}
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        idx = (px.index >= lo) & (px.index <= hi)
        results[name] = {"strat": perf(strat[idx]), "bh": perf(bh[idx]),
                         "flips_per_yr": float(np.nansum(flips[idx].values) / (idx.sum() / 365))}
        print(f"\n=== {name} ===  (position flips/yr, both coins: "
              f"{results[name]['flips_per_yr']:.1f})")
        for k in ["strat", "bh"]:
            p = results[name][k]
            print(f"{k:>6}: n={p['n']:5d} ann={p['ann_ret']:+7.3f} vol={p['ann_vol']:.3f} "
                  f"sharpe={p['sharpe']:+.2f} maxdd={p['maxdd']:+.3f} t={p['t']:+.2f}")

    r, v = results["research"], results["validation"]
    crit = {
        "sharpe_beats_bh_research": bool(r["strat"]["sharpe"] >= r["bh"]["sharpe"]),
        "sharpe_beats_bh_validation": bool(v["strat"]["sharpe"] >= v["bh"]["sharpe"]),
        "dd_shallower_research": bool(r["strat"]["maxdd"] > r["bh"]["maxdd"]),
        "dd_shallower_validation": bool(v["strat"]["maxdd"] > v["bh"]["maxdd"]),
        "val_net_ret_positive": bool(v["strat"]["ann_ret"] > 0),
    }
    passed = all(crit.values())
    print(f"\ncriteria: {crit}\n-> {'PASS' if passed else 'FAIL'}")

    register(Experiment(
        name="exp14_crypto_tsmom_btc_eth",
        family="crypto",
        hypothesis="retail-dominated 24/7 market retains trend inefficiency "
                   "institutions arbitraged out of equities (Liu-Tsyvinski 2021)",
        features=["tsmom_30d"],
        split="research+validation (crypto splits: 2015-20 / 2021-23; 2024+ sealed)",
        universe="BTC-USD, ETH-USD",
        rebalance_days=REBAL_DAYS, holding_days=REBAL_DAYS,
        n_observations=v["strat"]["n"],
        metrics={f"{w}_{s}_{k}": round(float(results[w][s][k]), 3)
                 for w in ["research", "validation"] for s in ["strat", "bh"]
                 for k in ["ann_ret", "sharpe", "maxdd"]}
        | {k: bool(x) for k, x in crit.items()},
        outcome="PASS" if passed else "FAIL",
        failure_reason="" if passed else
        "failed pre-registered criterion(s): "
        + ", ".join(k for k, x in crit.items() if not x),
    ))


if __name__ == "__main__":
    main()
