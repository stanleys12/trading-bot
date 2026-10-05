#!/usr/bin/env python3
"""One-screen sprint status for the dashboard."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
arts = sorted((HERE.parent / "qr/sprint/artifacts").glob("*.json"),
              key=lambda f: f.stat().st_mtime, reverse=True)
print(f"{'EXPERIMENT':30}{'VERDICT':8}{'N':>7}{'NET bps':>9}{'T':>8}  RED TEAM")
for f in arts[:12]:
    a = json.loads(f.read_text())
    v = a.get("results", {}).get("validation", {})
    print(f"{a['name']:30}{a.get('verdict','?'):8}{v.get('n','-'):>7}"
          f"{v.get('mean_net_bps','-'):>9}{v.get('t_stat','-'):>8}  {a.get('red_team','-')}")
pend = list((HERE.parent / "qr/sprint/proposals").glob("*.json"))
print(f"\nproposals pending: {[p.stem for p in pend] or 'none'}")
