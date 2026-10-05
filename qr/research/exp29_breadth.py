"""
exp29: Institutional breadth changes (Chen-Hong-Stein) — the last family.
=========================================================================

CRITERIA FIXED BEFORE SCORING.

HYPOTHESIS
    Chen-Hong-Stein (2002 JFE): when the NUMBER of institutions holding a
    stock falls, short-sale constraints imply the price reflects only the
    optimists — overpricing, low future returns. Breadth INCREASES
    predict high returns. Found on 1979-1998 data; the last major free
    data family this project has never touched (positioning at the
    holder level rather than the short-interest level).

DATA / JOIN
    qr/data/processed/inst_breadth.parquet (13F, 40 quarters 2013-2022).
    CUSIP -> ticker via the SEC fails-to-deliver files (official, free);
    quarters map to the nearest-in-time available yearly crosswalk, and
    match rates are printed — a quarter mapping under 60% is excluded
    rather than silently thinned.

PIT
    Holdings for quarter end Q are FILED up to 45 days later; the signal
    is usable at Q + 50 calendar days. Outcome: the following 63 trading
    sessions (to the next quarter's usable date).

SPEC
    Signal      dBreadth = (n_holders_Q - n_holders_{Q-1}) / max(n_{Q-1}, 1)
    Universe    panel ADV63 >= $10M at the usable date (liquid: breadth
                is meaningful and costs measurable)
    Portfolio   decile long-short on dBreadth (science stat), quarterly
    Statistic   t of quarterly long-short returns per window; IC reported

POWER (before scoring)
    ~19/20 usable quarters per window; L-S quarterly sd ~4-6%; t >= 2
    detects >= ~2.5%/quarter (~10%/yr). CHS-era effect was ~4-6%/yr:
    THIS TEST IS POWERED ONLY FOR AN ABOVE-LITERATURE EFFECT, stated
    plainly. A FAIL means "no large effect", and that is the strongest
    statement 40 quarters can buy. It still closes the family honestly.

PRE-REGISTERED CRITERIA — all required
    C1  research (2013-2017) L-S mean > 0 with t >= 2.0
    C2  validation (2018-2022) L-S mean > 0 with t >= 2.0
    2023+ SEALED. Survival -> HOLD.
"""
from __future__ import annotations
import os

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qr.research.protocol import Experiment, register, trial_count  # noqa: E402

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim"
UA = {"User-Agent": f"stanley-research ({os.environ.get('RESEARCH_CONTACT_EMAIL', 'contact@example.com')})"}
ADV_MIN = 1e7
RESEARCH = ("2013-01-01", "2017-12-31")
VALIDATION = ("2018-01-01", "2022-12-31")


def cusip_map() -> pd.DataFrame:
    cache = INTERIM / "cusip_ticker_map.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    for y in range(2013, 2023):
        got = False
        for stem in (f"cnsfails{y}06b", f"cnsfails{y}12b", f"cnsfails{y}09b"):
            url = f"https://www.sec.gov/files/data/fails-deliver-data/{stem}.zip"
            try:
                r = requests.get(url, headers=UA, timeout=60)
            except requests.RequestException:
                continue
            if not (r.ok and r.content[:2] == b"PK"):
                continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            raw = z.read(z.namelist()[0]).decode("latin1", errors="replace")
            for line in raw.splitlines()[1:]:
                parts = line.split("|")
                if len(parts) >= 3 and len(parts[1]) >= 8:
                    rows.append({"year": y, "cusip8": parts[1][:8].upper(),
                                 "symbol": parts[2].strip().upper()})
            got = True
            break
        if not got:
            print(f"  no FTD file for {y}")
    df = (pd.DataFrame(rows).drop_duplicates(["year", "cusip8"])
          .reset_index(drop=True))
    df.to_parquet(cache)
    return df


def main() -> None:
    br = pd.read_parquet(PROCESSED / "inst_breadth.parquet")
    br["period"] = pd.to_datetime(br["period"])
    cmap = cusip_map()
    years_avail = sorted(cmap.year.unique())
    print(f"crosswalk years: {years_avail}  (K={trial_count()})")

    cols = ["symbol", "date", "adj_close", "dollar_volume"]
    p = pd.read_parquet(PROCESSED / "panel_v5_full.parquet", columns=cols)
    p["date"] = pd.to_datetime(p["date"])
    px = p.pivot_table(index="date", columns="symbol", values="adj_close",
                       aggfunc="last").sort_index()
    dv = p.pivot_table(index="date", columns="symbol",
                       values="dollar_volume", aggfunc="last").sort_index()
    adv63 = dv.rolling(63, min_periods=40).mean()
    dates = px.index

    quarters = sorted(br.period.unique())
    ls_rets, ls_dates, ics = [], [], []
    for qi in range(1, len(quarters) - 1):
        q, q_prev = quarters[qi], quarters[qi - 1]
        ymap = cmap[cmap.year == min(years_avail,
                                     key=lambda y: abs(y - q.year))]
        m = ymap.set_index("cusip8")["symbol"]
        cur = br[br.period == q].set_index("cusip8")["n_holders"]
        prev = br[br.period == q_prev].set_index("cusip8")["n_holders"]
        common = cur.index.intersection(prev.index)
        # INSTRUMENT REPAIR (first run scored zero quarters): the raw 13F
        # table is ~2/3 unmappable non-equity CUSIPs (bonds, options,
        # median 3 holders), which polluted the match-rate denominator and
        # tripped the crosswalk-failure gate on every quarter. The gate now
        # measures match rate where it is meant to: names with >= 5
        # holders, the only ones the signal or the universe can ever use.
        common = common[(cur[common] >= 5) | (prev[common] >= 5)]
        db = ((cur[common] - prev[common]) / prev[common].clip(lower=1))
        db.index = db.index.map(m)
        db = db[db.index.notna()]
        db = db.groupby(db.index).mean()
        match = len(db) / max(len(common), 1)
        usable = q + pd.Timedelta(days=50)
        u = np.searchsorted(dates.values, np.datetime64(usable), "left")
        if u + 64 >= len(dates) or match < 0.60:
            continue
        elig = adv63.iloc[u] >= ADV_MIN
        names = db.index.intersection(px.columns[elig.fillna(False)])
        f = pd.DataFrame({"sig": db[names],
                          "fwd": px.iloc[u + 63][names] / px.iloc[u][names] - 1}
                         ).dropna()
        if len(f) < 150:
            continue
        ics.append(sstats.spearmanr(f.sig, f.fwd).statistic)
        d10 = pd.qcut(f.sig.rank(method="first"), 10, labels=False)
        ls_rets.append(f.fwd[d10 == 9].mean() - f.fwd[d10 == 0].mean())
        ls_dates.append(q)
    ls = pd.Series(ls_rets, index=pd.DatetimeIndex(ls_dates))
    ic_s = pd.Series(ics, index=pd.DatetimeIndex(ls_dates))
    print(f"usable quarters: {len(ls)}")

    out = {}
    for name, (lo, hi) in [("research", RESEARCH), ("validation", VALIDATION)]:
        w = ls[(ls.index >= lo) & (ls.index <= hi)]
        wi = ic_s[(ic_s.index >= lo) & (ic_s.index <= hi)]
        t = w.mean() / (w.std() / np.sqrt(len(w))) if len(w) > 5 else np.nan
        out[name] = {"quarters": len(w), "ls_q": w.mean(), "t": t,
                     "mean_ic": wi.mean()}
        print(f"{name}: {len(w)} qtrs | L-S {w.mean()*100:+.2f}%/q t={t:+.2f}"
              f" | IC {wi.mean():+.4f}")

    r, v = out["research"], out["validation"]
    c1 = bool(r["ls_q"] > 0 and (r["t"] or 0) >= 2.0)
    c2 = bool(v["ls_q"] > 0 and (v["t"] or 0) >= 2.0)
    ok = c1 and c2
    print(f"\nC1 research t>=2: {c1} (t={r['t']:+.2f})  "
          f"C2 validation t>=2: {c2} (t={v['t']:+.2f})")
    print("HOLD" if ok else "REJECTED")

    register(Experiment(
        name="exp29_institutional_breadth",
        family="positioning",
        hypothesis="breadth-of-ownership changes predict returns (Chen-Hong-"
                   "Stein 2002); powered only for above-literature effects "
                   "(~10%/yr) given 40 quarters — stated before scoring",
        features=["d_breadth_q"],
        split="research 2013-2017 / validation 2018-2022; 2023+ SEALED",
        universe="ADV63>=$10M, FTD-crosswalk-mapped",
        rebalance_days=63, holding_days=63,
        n_observations=int(v["quarters"]),
        metrics={f"{s}_{k}": (round(float(x), 4) if x == x else None)
                 for s, d in out.items() for k, x in d.items()}
        | {"C1": c1, "C2": c2},
        outcome="HOLD" if ok else "FAIL",
        failure_reason="" if ok else
        f"failed pre-registered criterion(s): C1={c1} C2={c2}",
    ))


if __name__ == "__main__":
    main()
