#!/usr/bin/env python3
"""binance.vision minute-data layer for the sprint.

Survivorship handling: the S3 listing is enumerated for ALL pairs that ever
have kline folders (delisted pairs keep their history), and the point-in-time
top-N universe is built from trailing dollar volume computed from the data
itself — never from today's listing page.
"""
import io, json, sys, zipfile
from pathlib import Path
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DL = "https://data.binance.vision"
COLS = ["open_time","open","high","low","close","volume","close_time",
        "quote_volume","n_trades","taker_base","taker_quote","ignore"]


def list_all_symbols() -> list[str]:
    """Every spot pair with 1m kline history (S3 listing, paginated)."""
    syms, marker = [], ""
    while True:
        params = {"delimiter": "/", "prefix": "data/spot/monthly/klines/"}
        if marker:
            params["marker"] = marker
        r = requests.get(S3, params=params, timeout=30)
        r.raise_for_status()
        pres = [p.split("/")[-2] for p in
                pd.Series(r.text.split("<Prefix>")).str.split("</Prefix>").str[0]
                if p.endswith("/")]
        syms += pres
        if "<IsTruncated>true" not in r.text:
            break
        marker = f"data/spot/monthly/klines/{syms[-1]}/"
    return sorted(set(syms))


def fetch_month(symbol: str, ym: str) -> pd.DataFrame | None:
    """One month of 1m klines -> DataFrame, cached as parquet."""
    out = CACHE / symbol / f"{ym}.v2.parquet"   # v2: adds taker_buy
    if out.exists():
        return pd.read_parquet(out)
    url = f"{DL}/data/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{ym}.zip"
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COLS)
    if df.iloc[0].open_time == "open_time":          # some files carry headers
        df = df.iloc[1:].reset_index(drop=True)
    df = df.astype({c: "float64" for c in
                    ["open","high","low","close","volume","quote_volume"]})
    df["open_time"] = pd.to_numeric(df.open_time)
    # 2025+ files use microseconds; older use ms. Normalize to UTC ts.
    unit = "us" if df.open_time.iloc[0] > 1e14 else "ms"
    df["ts"] = pd.to_datetime(df.open_time, unit=unit, utc=True)
    # taker_base = volume bought by aggressive (taker) buyers. The order-flow
    # imbalance literature (EFMA 2025; arXiv 2607.09426) makes this the most
    # information-dense free field in the file — v1 of this loader dropped it.
    df["taker_buy"] = pd.to_numeric(df["taker_base"], errors="coerce")
    df = df[["ts","open","high","low","close","volume","quote_volume","n_trades","taker_buy"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


def load(symbol: str, start: str, end: str) -> pd.DataFrame:
    months = pd.period_range(start[:7], end[:7], freq="M").strftime("%Y-%m")
    parts = [fetch_month(symbol, m) for m in months]
    parts = [p for p in parts if p is not None]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts).drop_duplicates("ts").sort_values("ts")
    return df[(df.ts >= start) & (df.ts < end)].reset_index(drop=True)


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "symbols":
        s = list_all_symbols()
        (CACHE / "all_symbols.json").write_text(json.dumps(s))
        usdt = [x for x in s if x.endswith("USDT")]
        print(f"{len(s)} pairs total, {len(usdt)} USDT pairs")
    else:
        d = load("BTCUSDT", "2026-06-01", "2026-06-08")
        print(d.head(3).to_string()); print(f"{len(d)} rows")
