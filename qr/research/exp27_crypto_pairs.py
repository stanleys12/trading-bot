"""
exp27: Crypto pairs trading (distance method) — relative value, first test.
===========================================================================

CRITERIA FIXED BEFORE SCORING.

WHY
    Relative-value / statistical arbitrage is an entire mechanism class
    this project has never tested in ANY market: bet on the SPREAD
    between two co-moving assets, direction-agnostic. Equity pairs
    profits (Gatev-Goetzmann-Rouwenhorst) decayed after 2002; recent
    studies on 2022-2024 CRYPTO data still report strong results
    (BTC-ETH cointegration pairs, Sharpe ~2+). Retail-dominated market,
    low fees on majors, survivorship-clean panel in hand.

SPEC (GGR distance method, one configuration)
    Universe    coins with trailing-30d median dollar volume >= $5M
    Formation   60 days: normalize each coin's cumulative return path to
                1.0 at window start; pair distance = sum of squared path
                differences; take the 15 closest pairs
    Trading     next 30 days: open a pair when |spread z| > 2 (z uses the
                formation window's spread mean/sd); long the laggard,
                short the leader, equal legs; close when z crosses 0 or
                at window end. One open per pair per window.
    Costs       11bp per side x 4 legs = 44bp per completed round trip,
                charged at close. Short-leg funding is NOT modelled and
                is a stated caveat (perp funding averages positive but
                varies); any survival is conditional on it.
    Portfolio   equal weight across active pairs; days with none = 0.
    Windows     sequential non-overlapping 30d trading windows.

SPLITS
    research    2017-08 .. 2021-12    validation 2022-01 .. 2023-12
    2024+ SEALED (the crypto seal).

POWER (before scoring)
    Monthly-clustered t. Validation has ~24 windows; pair-portfolio
    monthly vol ~1.5-3%. t >= 2 detects ~0.6-1.2%/month = 8-15%/yr.
    The recent literature claims ~16%/yr: powered for the claimed size.

PRE-REGISTERED CRITERIA — all required
    C1  research: net monthly mean > 0, t >= 2.5
    C2  validation: net monthly mean > 0, t >= 2.0
    Survival -> HOLD (with the funding caveat attached).
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register, trial_count  # noqa: E402
from qr.research.exp16_crypto_xs_momentum import load_panel  # noqa: E402

ADV_MIN = 5e6
FORM, TRADE = 60, 30
N_PAIRS = 15
Z_OPEN = 2.0
COST_RT = 44e-4
RESEARCH = ("2017-08-01", "2021-12-31")
VALIDATION = ("2022-01-01", "2023-12-31")


def main() -> None:
    p = load_panel()
    px = p.pivot_table(index="date", columns="symbol", values="close",
                       aggfunc="last").sort_index()
    dollar = p.pivot_table(index="date", columns="symbol",
                           values="dollar_volume", aggfunc="last").sort_index()
    adv30 = dollar.rolling(30, min_periods=20).median()
    dates = px.index
    print(f"panel {px.shape}  (K={trial_count()})")

    daily_pnl = pd.Series(0.0, index=dates)
    n_trades = 0
    for start in range(FORM, len(dates) - TRADE, TRADE):
        f_lo, f_hi = start - FORM, start
        elig = (adv30.iloc[f_hi] >= ADV_MIN)
        cols = [s for s in px.columns[elig.fillna(False)]
                if px[s].iloc[f_lo:f_hi].notna().all()
                and px[s].iloc[f_hi:start + TRADE].notna().all()]
        if len(cols) < 10:
            continue
        norm = px[cols].iloc[f_lo:f_hi]
        norm = norm / norm.iloc[0]
        # closest pairs by path distance
        dists = []
        arr = norm.values
        idx = {s: k for k, s in enumerate(cols)}
        for a, b in combinations(cols, 2):
            d = np.mean((arr[:, idx[a]] - arr[:, idx[b]]) ** 2)
            dists.append((d, a, b))
        dists.sort()
        pairs = dists[:N_PAIRS]

        for _, a, b in pairs:
            spread_f = norm[a] - norm[b]
            mu, sd = spread_f.mean(), spread_f.std()
            if not np.isfinite(sd) or sd == 0:
                continue
            pa = px[a].iloc[f_hi - 1:start + TRADE] / px[a].iloc[f_hi - 1]
            pb = px[b].iloc[f_hi - 1:start + TRADE] / px[b].iloc[f_hi - 1]
            z = ((pa * norm[a].iloc[-1]) - (pb * norm[b].iloc[-1]) - mu) / sd
            open_i = None
            sign = 0
            for k in range(1, len(z)):
                if open_i is None and abs(z.iloc[k]) > Z_OPEN:
                    open_i, sign = k, -np.sign(z.iloc[k])
                    continue
                if open_i is not None and (np.sign(z.iloc[k]) != -sign
                                           or k == len(z) - 1):
                    ra = pa.iloc[k] / pa.iloc[open_i] - 1
                    rb = pb.iloc[k] / pb.iloc[open_i] - 1
                    pnl = sign * 0.5 * (ra - rb) - COST_RT
                    day = z.index[k]
                    daily_pnl.loc[day] += pnl / N_PAIRS
                    n_trades += 1
                    open_i = None
                    break
    print(f"completed pair trades: {n_trades}")

    monthly = daily_pnl.groupby(daily_pnl.index.to_period("M")).sum()
    out = {}
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        w = monthly[(monthly.index >= pd.Period(lo[:7]))
                    & (monthly.index <= pd.Period(hi[:7]))]
        t = w.mean() / (w.std() / np.sqrt(len(w))) if len(w) > 5 else np.nan
        out[name] = {"months": len(w), "net_monthly": w.mean(),
                     "ann_pct": w.mean() * 12 * 100, "t": t}
        print(f"{name}: {len(w)} months | net {w.mean()*100:+.3f}%/mo "
              f"({w.mean()*12*100:+.1f}%/yr) t={t:+.2f}")

    r, v = out["research"], out["validation"]
    c1 = bool(r["net_monthly"] > 0 and r["t"] >= 2.5)
    c2 = bool(v["net_monthly"] > 0 and v["t"] >= 2.0)
    ok = c1 and c2
    print(f"\nC1 research t>=2.5: {c1} (t={r['t']:+.2f})  "
          f"C2 validation t>=2: {c2} (t={v['t']:+.2f})")
    print("HOLD (funding caveat attached)" if ok else "REJECTED")

    register(Experiment(
        name="exp27_crypto_pairs_distance",
        family="relative_value",
        hypothesis="co-moving coin pairs mean-revert exploitably (GGR "
                   "distance method; recent 2022-24 crypto literature "
                   "claims Sharpe ~2); first relative-value trial in any "
                   "market for this project",
        features=["pair_distance_60_30"],
        split="research 2017-2021 / validation 2022-2023; 2024+ SEALED",
        universe="Binance USDT incl. delisted, ADV>=$5M",
        rebalance_days=30, holding_days=30,
        n_observations=int(n_trades),
        metrics={f"{s}_{k}": (round(float(x), 4) if x == x else None)
                 for s, d in out.items() for k, x in d.items()}
        | {"n_trades": n_trades, "C1": c1, "C2": c2},
        outcome="HOLD" if ok else "FAIL",
        failure_reason="" if ok else
        f"failed pre-registered criterion(s): C1={c1} C2={c2}",
    ))


if __name__ == "__main__":
    main()
