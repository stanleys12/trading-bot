#!/usr/bin/env python3
"""Weather-market recorder: Kalshi daily-high temperature ladders vs the
free NWS forecast, every 30 minutes. Phase 1 = MEASURE (same discipline as
the crypto recorder): after a few days we compare market-implied highs vs
NWS forecasts vs realized highs, and only a demonstrated calibration gap
earns a paper-trading phase. Documented precedent: retail bots profitably
priced these markets off free government forecast ensembles.
Stations = the exact ones Kalshi resolves against."""
import os
import json, time
from datetime import datetime, timezone
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "reports" / "weather_snapshots"
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": f"stanley-research ({os.environ.get('RESEARCH_CONTACT_EMAIL', 'contact@example.com')})"}

CITIES = {
    "KXHIGHNY":  ("Central Park NYC", 40.7789, -73.9692),
    "KXHIGHCHI": ("Chicago Midway",   41.7842, -87.7553),
    "KXHIGHMIA": ("Miami Intl",       25.7906, -80.3164),
    "KXHIGHAUS": ("Austin Camp Mabry",30.3208, -97.7603),
    "KXHIGHDEN": ("Denver Intl",      39.8467, -104.6562),
}
_grid = {}


def grid_for(lat, lon):
    key = (lat, lon)
    if key not in _grid:
        p = requests.get(f"https://api.weather.gov/points/{lat},{lon}",
                         headers=UA, timeout=20).json()["properties"]
        _grid[key] = (p["gridId"], p["gridX"], p["gridY"])
    return _grid[key]


def nws_forecast(lat, lon):
    g = grid_for(lat, lon)
    r = requests.get(f"https://api.weather.gov/gridpoints/{g[0]}/{g[1]},{g[2]}/forecast",
                     headers=UA, timeout=20).json()
    out = []
    for per in r.get("properties", {}).get("periods", [])[:4]:
        out.append({"name": per["name"], "temp": per["temperature"],
                    "unit": per["temperatureUnit"], "daytime": per["isDaytime"]})
    return out


def kalshi_ladder(series):
    r = requests.get("https://api.elections.kalshi.com/trade-api/v2/markets",
                     params={"series_ticker": series, "status": "open",
                             "limit": 100}, timeout=20).json()
    rows = []
    for m in r.get("markets", []):
        rows.append({"id": m["ticker"], "q": m.get("title"),
                     "close": m.get("close_time"),
                     "yes_bid": m.get("yes_bid_dollars"),
                     "yes_ask": m.get("yes_ask_dollars"),
                     "bid_sz": m.get("yes_bid_size_fp"),
                     "ask_sz": m.get("yes_ask_size_fp"),
                     "vol24": m.get("volume_24h_fp")})
    return rows


def main():
    while True:
        try:
            snap = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "cities": {}}
            for series, (name, lat, lon) in CITIES.items():
                try:
                    snap["cities"][series] = {
                        "station": name,
                        "nws": nws_forecast(lat, lon),
                        "ladder": kalshi_ladder(series)}
                except Exception as e:
                    snap["cities"][series] = {"error": str(e)[:120]}
            f = OUT / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
            with f.open("a") as fh:
                fh.write(json.dumps(snap) + "\n")
        except Exception as e:
            with (OUT / "recorder.err").open("a") as fh:
                fh.write(f"{datetime.now()} {str(e)[:200]}\n")
        time.sleep(1800)


if __name__ == "__main__":
    main()
