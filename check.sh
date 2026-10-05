#!/bin/bash
# How is the paper forward-test doing?
#
# 2026-08-26: reads daily_v2. The old `daily` table conflated all three books
# in one account and benchmarked against a ONE-MINUTE SPY bar; its numbers
# are not usable. It is retained in the DB as a record of what was reported.
cd /Users/stanleyshen/trading-bot
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
$PY -W ignore - <<'PYEOF'
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from qr.execution.live_paper import positions, market_open, _db
from qr.execution.daily_run import performance, RESERVED

pos  = positions()
sleeve = pos[~pos.symbol.isin(RESERVED)] if len(pos) else pos
iwm    = pos[pos.symbol.isin(RESERVED)]  if len(pos) else pos
crypto = positions(asset_class="crypto")
opt    = positions(asset_class="us_option")

def mv(df):  return float(df.market_value.sum()) if len(df) else 0.0

print("=== THE THREE BOOKS (separate strategies, one shared account) ===")
print(f"  low-vol equity   {len(sleeve):>4} names   ${mv(sleeve):>12,.0f}   <-- the strategy under test")
print(f"  crypto TSMOM     {len(crypto):>4} pairs   ${mv(crypto):>12,.0f}")
print(f"  covered call      IWM + short call  ${mv(iwm)+mv(opt):>12,.0f}")
print(f"  market_open={market_open()}")

con = _db()
try:
    t = pd.read_sql("SELECT COUNT(*) n, SUM(target_shares>0) live FROM targets "
                    "WHERE run_id=(SELECT run_id FROM targets ORDER BY ts DESC LIMIT 1)", con)
    held = set(sleeve.symbol) if len(sleeve) else set()
    tg = pd.read_sql("SELECT symbol,target_shares FROM targets WHERE run_id="
                     "(SELECT run_id FROM targets ORDER BY ts DESC LIMIT 1)", con)
    want = {r.symbol: r.target_shares for r in tg.itertuples() if r.target_shares > 0}
    missing = [s for s in want if s not in held]
    print(f"\n=== BOOK vs TARGET ===")
    print(f"  target names {len(want)}   held {len(held)}   MISSING {len(missing)}")
    ra = pd.read_sql("SELECT COUNT(*) n FROM repair_attempts", con).n.iloc[0]
    print(f"  names with repair attempts logged: {ra}")
except Exception as e:
    print(f"\n(no stored target book yet: {e})")

d = performance()
if len(d):
    print(f"\n=== LOW-VOL BOOK, COMPLETED SESSIONS ({len(d)}) ===")
    cols = ['session_date','n_positions','sleeve_mv','book_day_pct','spy_day_pct',
            'excess_vs_spy_pct','cum_pct','cum_spy_pct','cum_excess_pct','coverage_pct']
    print(d[[c for c in cols if c in d.columns]].round(3).to_string(index=False))
    r = d.book_day_pct.dropna()/100
    if len(r) > 2 and r.std() > 0:
        print(f"\n  ann. vol {r.std()*np.sqrt(252)*100:.1f}%   "
              f"Sharpe {r.mean()/r.std()*np.sqrt(252):.2f} "
              f"(MEANINGLESS below ~100 sessions)")
    print(f"\n  NOTE: {len(d)} session(s). A ~276-name low-vol book runs roughly 11%")
    print( "  annualised vol, so daily moves are noise. Nothing here is evidence")
    print( "  about the strategy until the sample is months long.")
else:
    print("\nno completed sessions recorded in daily_v2 yet")
PYEOF
