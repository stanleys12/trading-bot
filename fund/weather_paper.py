#!/usr/bin/env python3
"""Weather paper-track: each day, resolve YESTERDAY's Kalshi high-temp markets
against realized station highs, book the forecast-vs-market paper trades, and
append to a growing, self-scoring track record. Runs autonomously so n grows
toward a powered verdict without a human in the loop.

Strategy (frozen 2026-09-23, pilot): for each threshold market, take the last
recorded pre-close quote; if the NWS daytime-high forecast implies the outcome
(fc>thr+1 => YES, fc<thr-1 => NO) and disagrees with the market mid by >15c,
take our side at the market ask/bid; settle at realized high; net Kalshi fees.
PAPER ONLY. Pilot P&L at build time: +0.60/trade, 75% win, t=2.64, n=8 —
promising but UNDERPOWERED and likely weather-correlated; this job exists to
find out if it survives real forward sample."""
import os
import json, glob, re, requests, datetime, statistics
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
UA = {"User-Agent": f"stanley-research ({os.environ.get('RESEARCH_CONTACT_EMAIL', 'contact@example.com')})"}
STN={"KXHIGHNY":"KNYC","KXHIGHCHI":"KMDW","KXHIGHMIA":"KMIA","KXHIGHAUS":"KATT","KXHIGHDEN":"KDEN"}
kfee=lambda p:0.07*p*(1-p)
TRACK=HERE/"reports"/"weather_track.jsonl"
def realized(stn,day):
    try:
        r=requests.get(f"https://api.weather.gov/stations/{stn}/observations",
            params={"start":f"{day}T00:00:00Z","end":f"{day}T23:59:59Z","limit":200},headers=UA,timeout=25).json()
        t=[o["properties"]["temperature"]["value"] for o in r.get("features",[]) if o["properties"]["temperature"]["value"] is not None]
        return max(x*9/5+32 for x in t) if t else None
    except: return None
def main():
    done={json.loads(l)["key"] for l in TRACK.read_text().splitlines()} if TRACK.exists() else set()
    book=defaultdict(dict)
    for f in sorted(glob.glob(str(HERE/"reports"/"weather_snapshots"/"*.jsonl"))):
        for line in open(f):
            try:s=json.loads(line)
            except:continue
            for series,c in s.get("cities",{}).items():
                if "error" in c:continue
                fc=next((p["temp"] for p in c.get("nws",[]) if p.get("daytime")),None)
                for m in c.get("ladder",[]):
                    tm=re.search(r"T(\d+)",m["id"]); b,a=m.get("yes_bid"),m.get("yes_ask")
                    if not tm or b in (None,"0.0000") or a is None:continue
                    day=m.get("close","")[:10]
                    if fc is not None: book[(series,day)]["nws"]=fc
                    book[(series,day)][int(tm.group(1))]=(float(b),float(a))
    today=datetime.date.today(); new=0
    for (series,day),d in book.items():
        key=f"{series}:{day}"
        if key in done or "nws" not in d:continue
        try:dd=datetime.date.fromisoformat(day)
        except:continue
        if dd>=today:continue
        rh=realized(STN[series],day)
        if rh is None:continue
        fc=d["nws"]; trades=[]
        for thr,v in d.items():
            if not isinstance(thr,int):continue
            bid,ask=v; mid=(bid+ask)/2
            our=1.0 if fc>thr+1 else (0.0 if fc<thr-1 else 0.5)
            if our==0.5 or abs(our-mid)<0.15:continue
            outcome=1.0 if rh>thr else 0.0
            pnl=(outcome-ask-kfee(ask)) if our-mid>0 else ((1-outcome)-(1-bid)-kfee(bid))
            trades.append(round(pnl,4))
        if trades:
            with TRACK.open("a") as fh:
                fh.write(json.dumps({"key":key,"series":series,"day":day,"nws":fc,
                    "realized":round(rh),"n":len(trades),"pnl":round(sum(trades),4)})+"\n")
            new+=1
    # report cumulative
    if TRACK.exists():
        rows=[json.loads(l) for l in TRACK.read_text().splitlines()]
        allp=[r["pnl"] for r in rows]; nt=sum(r["n"] for r in rows)
        m=statistics.mean(allp) if allp else 0
        se=statistics.pstdev(allp)/len(allp)**0.5 if len(allp)>1 else 0
        msg=(f"weather paper-track: +{new} city-days | cumulative {len(rows)} days, "
             f"{nt} trades, net {sum(allp):+.2f}, t={m/se:.2f}" if se else
             f"weather paper-track: {len(rows)} days")
        with (HERE/"reports"/"floor_feed.log").open("a") as fh:
            fh.write(f"{datetime.datetime.now():%H:%M:%S}  {msg}\n")
        print(msg)
if __name__=="__main__": main()
