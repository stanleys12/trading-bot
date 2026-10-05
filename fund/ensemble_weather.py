#!/usr/bin/env python3
"""Ensemble weather paper-track (v2 of the weather idea).

The v1 weather test used a single-POINT NWS forecast, which can't price a
probability market — and its apparent edge was correlated-day luck (t 2.64
-> 1.45 -> noise). This version uses ENSEMBLE forecasts (GFS 31 + ECMWF 51
+ ICON 40 members) to get a real calibrated P(high > threshold), the actual
documented-profitable approach.

Discipline learned the hard way:
 * DATE ALIGNMENT: match each market's close DATE to the ensemble forecast
   for THAT date (forecast_days index), never today's forecast vs a future
   market — that mismatch produced fake 70c edges.
 * MORNING ONLY: only take a position from a snapshot taken >6h before the
   market's local day ends, so the market can't have intraday high info the
   forecast lacks (the Denver-1c artifact).
 * INDEPENDENCE: one aggregate P&L per (city, day); never count 6 correlated
   thresholds as 6 trades.
 * FORWARD ONLY: no historical ensemble archive, so this proves out only on
   real forward sample. PAPER. Verdict needs ~30+ independent city-days.
"""
import os
import requests, json, re, datetime, statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
UA = {"User-Agent": f"stanley-research ({os.environ.get('RESEARCH_CONTACT_EMAIL', 'contact@example.com')})"}
CITY={"KXHIGHNY":(40.78,-73.97,"KNYC"),"KXHIGHCHI":(41.78,-87.75,"KMDW"),
      "KXHIGHMIA":(25.79,-80.32,"KMIA"),"KXHIGHAUS":(30.32,-97.76,"KATT"),
      "KXHIGHDEN":(39.85,-104.66,"KDEN")}
kfee=lambda p:0.07*p*(1-p)
PEND=HERE/"reports"/"ensemble_pending.jsonl"   # positions awaiting resolution
TRACK=HERE/"reports"/"ensemble_track.jsonl"    # resolved, scored

def ens_by_date(lat,lon):
    """{iso_date: [member max-temps]} for the next few days."""
    out={}
    for model in ("gfs025","ecmwf_ifs025","icon_seamless"):
        try:
            d=requests.get("https://ensemble-api.open-meteo.com/v1/ensemble",
                params={"latitude":lat,"longitude":lon,"daily":"temperature_2m_max",
                        "models":model,"temperature_unit":"fahrenheit","forecast_days":4},timeout=25).json()
            dates=d.get("daily",{}).get("time",[])
            for k in d.get("daily",{}):
                if "temperature_2m_max" not in k: continue
                for i,day in enumerate(dates):
                    v=d["daily"][k][i]
                    if v is not None: out.setdefault(day,[]).append(v)
        except: pass
    return out

def realized(stn,day):
    try:
        r=requests.get(f"https://api.weather.gov/stations/{stn}/observations",
            params={"start":f"{day}T00:00:00Z","end":f"{day}T23:59:59Z","limit":200},headers=UA,timeout=25).json()
        t=[o["properties"]["temperature"]["value"] for o in r.get("features",[]) if o["properties"]["temperature"]["value"] is not None]
        return max(x*9/5+32 for x in t) if t else None
    except: return None

def open_positions():
    now=datetime.datetime.now(datetime.timezone.utc)
    seen={json.loads(l)["key"] for l in PEND.read_text().splitlines()} if PEND.exists() else set()
    for series,(lat,lon,stn) in CITY.items():
        edates=ens_by_date(lat,lon)
        if not edates: continue
        r=requests.get("https://api.elections.kalshi.com/trade-api/v2/markets",
            params={"series_ticker":series,"status":"open","limit":100},timeout=20).json()
        for m in r.get("markets",[]):
            tk=m.get("ticker",""); tm=re.search(r"T(\d+)",tk)
            b,a=m.get("yes_bid_dollars"),m.get("yes_ask_dollars")
            close=m.get("close_time","")
            if not tm or b in (None,"0.0000") or a is None or not close: continue
            cday=close[:10]; key=f"{tk}"
            if key in seen or cday not in edates: continue
            # MORNING-ONLY guard: need >6h before close
            try: ct=datetime.datetime.fromisoformat(close.replace("Z","+00:00"))
            except: continue
            if (ct-now).total_seconds()<6*3600: continue
            mem=edates[cday]
            if len(mem)<20: continue
            thr=int(tm.group(1)); ens_p=sum(1 for v in mem if v>thr)/len(mem)
            mid=(float(b)+float(a))/2; edge=ens_p-mid
            if abs(edge)<0.15 or not (0.05<ens_p<0.95): continue
            side="YES" if edge>0 else "NO"
            price=float(a) if edge>0 else 1-float(b)
            with PEND.open("a") as fh:
                fh.write(json.dumps({"key":key,"series":series,"stn":stn,"day":cday,
                    "thr":thr,"ens_p":round(ens_p,3),"mid":round(mid,3),"side":side,
                    "price":round(price,3),"recorded":now.isoformat(timespec="seconds")})+"\n")

def resolve():
    if not PEND.exists(): return
    pend=[json.loads(l) for l in PEND.read_text().splitlines()]
    done={json.loads(l)["key"] for l in TRACK.read_text().splitlines()} if TRACK.exists() else set()
    today=datetime.date.today(); still=[]
    for p in pend:
        if p["key"] in done: continue
        if datetime.date.fromisoformat(p["day"])>=today: still.append(p); continue
        rh=realized(p["stn"],p["day"])
        if rh is None: still.append(p); continue
        win=(rh>p["thr"]) if p["side"]=="YES" else (rh<=p["thr"])
        pnl=(1-p["price"]-kfee(p["price"])) if win else (-p["price"]-kfee(p["price"]))
        with TRACK.open("a") as fh:
            fh.write(json.dumps({**p,"realized":round(rh),"win":win,"pnl":round(pnl,4)})+"\n")
    PEND.write_text("\n".join(json.dumps(p) for p in still)+("\n" if still else ""))

def report():
    if not TRACK.exists(): return "ensemble weather: no resolved positions yet"
    rows=[json.loads(l) for l in TRACK.read_text().splitlines()]
    byday={}
    for r in rows: byday.setdefault((r["series"],r["day"]),[]).append(r["pnl"])
    dayp=[sum(v)/len(v) for v in byday.values()]   # one independent unit per city-day
    n=len(dayp); net=sum(r["pnl"] for r in rows)
    if n>1:
        m=statistics.mean(dayp); se=statistics.pstdev(dayp)/n**0.5
        return f"ensemble weather: {len(rows)} trades / {n} indep city-days, net {net:+.2f}, t={m/se:.2f} (need ~2 & n>=30)"
    return f"ensemble weather: {len(rows)} trades / {n} city-day (accumulating)"

if __name__=="__main__":
    resolve(); open_positions()
    msg=report()
    with (HERE/"reports"/"floor_feed.log").open("a") as fh:
        fh.write(f"{datetime.datetime.now():%H:%M:%S}  {msg}\n")
    print(msg)
    print("pending positions:", len(PEND.read_text().splitlines()) if PEND.exists() else 0)
