#!/usr/bin/env python3
"""EVENT-DRIVEN DISPATCHER: agents wake when something happens, not on timers.

Cheap deterministic watchers poll every ~20s (free). When a condition trips,
the relevant seat is summoned with the trigger stated in its briefing, files
its memo (streamed, journaled, debated if HIGH), and stands down.

Watchers -> who wakes:
  order fills / expiries / rejections        -> execution_verifier
  MISSING-names count changes                -> equity_pm
  margin sign flip or equity jump >1%        -> risk_officer
  SPY moves >0.6% in 15min                   -> scout_macro
  BTC/ETH moves >1.5% in 15min (mkt hours)   -> scout_crypto
     or >3.0% in 60min (any hour)
  new/updated sprint artifact                -> red_team (production review;
     its APPROVED/objection verdict is written back onto the artifact)
  fresh morning run line in a book log       -> that book's PM
  >=4 filings since last synthesis           -> chief_of_desk
  90 min of total silence in market hours    -> risk_officer patrol

Guards: per-seat cooldown 10 min, wakes run one at a time, 120/day cap.
"""
import json, re, subprocess, sys, time
from collections import deque
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import requests                                                     # noqa: E402
from qr.execution.live_paper import (_headers, PAPER_BASE, DATA_BASE,
                                     positions, latest_targets)     # noqa: E402
from orchestrator import feed, run_solo, SEATS                      # noqa: E402
from briefing import build                                          # noqa: E402

COOLDOWN, MAX_DAY = 600, 120
ART = ROOT / "qr/sprint/artifacts"
BOOKS = {"STOCKS": (ROOT / "qr/reports/live/cron.log", "equity_pm"),
         "CRYPTO": (ROOT / "qr/reports/live/crypto_cron.log", "crypto_pm"),
         "OPTIONS": (ROOT / "qr/reports/live/options_cron.log", "options_pm")}


class D:
    def __init__(self):
        self.last_wake = {}          # seat -> ts
        self.wakes_today, self.day = 0, datetime.now().date()
        self.orders = {}
        self.missing = None
        self.equity = None
        self.cash = None
        self.px = {"SPY": deque(maxlen=50), "BTC": deque(maxlen=200),
                   "ETH": deque(maxlen=200)}
        self.offs = {k: (f.stat().st_size if f.exists() else 0)
                     for k, (f, _) in BOOKS.items()}
        self.art_seen = {}
        for f in ART.glob("*.json"):
            try:
                pending = json.loads(f.read_text()).get("red_team") == "pending"
            except Exception:
                pending = False
            # pending artifacts trigger on the first pass (clears the backlog)
            self.art_seen[f.name] = 0 if pending else f.stat().st_mtime
        self.filings_since_chief = 0
        self.last_any_wake = time.time()
        self.exp_today, self.exp_day = 0, datetime.now().date()
        self.last_lead_wake = 0.0

    def can(self, seat):
        if datetime.now().date() != self.day:
            self.day, self.wakes_today = datetime.now().date(), 0
        return (self.wakes_today < MAX_DAY
                and time.time() - self.last_wake.get(seat, 0) > COOLDOWN)

    def wake(self, seat, reason):
        if not self.can(seat):
            return
        self.last_wake[seat] = time.time()
        self.wakes_today += 1
        self.last_any_wake = time.time()
        feed(f"TRIGGER: waking {seat} — {reason}")
        try:
            briefing = build() + f"\n\n## WHY YOU WERE WOKEN (event trigger)\n{reason}\n"
            run_solo(seat, briefing, datetime.now().astimezone())
            if seat != "chief_of_desk":
                self.filings_since_chief += 1
        except Exception as e:
            feed(f"TRIGGER ERROR for {seat}: {str(e)[:100]}")

    # ---------------- watchers (all deterministic, all cheap) ----------------
    def market_open(self):
        try:
            c = requests.get(f"{PAPER_BASE}/clock", headers=_headers(),
                             timeout=15).json()
            return bool(c.get("is_open"))
        except Exception:
            return False

    def w_orders(self):
        o = requests.get(f"{PAPER_BASE}/orders", headers=_headers(),
                         params={"status": "all", "limit": 500,
                                 "after": datetime.now().strftime("%Y-%m-%dT07:00:00Z")},
                         timeout=20).json()
        fills = dead = news = 0
        for x in o:
            oid, s = x["id"], x["status"]
            prev = self.orders.get(oid)
            if prev is None:
                news += 1
            elif prev != s:
                if s == "filled":
                    fills += 1
                elif s in ("expired", "canceled", "rejected"):
                    dead += 1
            self.orders[oid] = s
        if fills:
            self.wake("execution_verifier",
                      f"{fills} order(s) just FILLED — verify against targets and cost model")
        elif dead:
            self.wake("execution_verifier",
                      f"{dead} order(s) expired/cancelled unfilled — assess repair progress")

    def w_book(self):
        pos = positions(); tgt = latest_targets()
        want = {r.symbol for r in tgt.itertuples() if r.target_shares > 0}
        m = len(want - set(pos.symbol)) if len(pos) else len(want)
        if self.missing is not None and m != self.missing:
            self.wake("equity_pm",
                      f"portfolio completion changed: MISSING {self.missing} -> {m}")
        self.missing = m

    def w_account(self):
        a = requests.get(f"{PAPER_BASE}/account", headers=_headers(), timeout=20).json()
        eq, cash = float(a["equity"]), float(a["cash"])
        if self.equity and abs(eq / self.equity - 1) > 0.01:
            self.wake("risk_officer",
                      f"account equity moved {100*(eq/self.equity-1):+.1f}% "
                      f"(${self.equity:,.0f} -> ${eq:,.0f})")
        if self.cash is not None and (cash >= 0) != (self.cash >= 0):
            self.wake("risk_officer",
                      "margin status flipped: " +
                      ("borrowing cleared" if cash >= 0 else "now borrowing"))
        self.equity, self.cash = eq, cash

    def w_prices(self, mkt):
        now = time.time()
        try:
            q = requests.get(f"{DATA_BASE}/stocks/quotes/latest", headers=_headers(),
                             params={"symbols": "SPY", "feed": "iex"},
                             timeout=15).json()["quotes"]["SPY"]
            mid = (q["bp"] + q["ap"]) / 2 if q.get("bp") and q.get("ap") else None
            if mid:
                self.px["SPY"].append((now, mid))
        except Exception:
            pass
        try:
            c = requests.get("https://data.alpaca.markets/v1beta3/crypto/us/latest/quotes",
                             headers=_headers(),
                             params={"symbols": "BTC/USD,ETH/USD"}, timeout=15
                             ).json().get("quotes", {})
            for sym, key in [("BTC/USD", "BTC"), ("ETH/USD", "ETH")]:
                if sym in c and c[sym].get("bp"):
                    self.px[key].append((now, (c[sym]["bp"] + c[sym]["ap"]) / 2))
        except Exception:
            pass

        def move(key, mins):
            buf = self.px[key]
            if len(buf) < 2:
                return 0.0
            cutoff = now - mins * 60
            old = next((p for t, p in buf if t >= cutoff), None)
            return (buf[-1][1] / old - 1) * 100 if old else 0.0

        if mkt and abs(move("SPY", 15)) > 0.6:
            self.wake("scout_macro", f"SPY moved {move('SPY',15):+.2f}% in 15 minutes")
        for key in ("BTC", "ETH"):
            m15, m60 = move(key, 15), move(key, 60)
            if (mkt and abs(m15) > 1.5) or abs(m60) > 3.0:
                self.wake("scout_crypto",
                          f"{key} moved {m15:+.2f}% (15m) / {m60:+.2f}% (60m)")

    def w_booklogs(self):
        for tag, (f, seat) in BOOKS.items():
            if not f.exists():
                continue
            size = f.stat().st_size
            if size > self.offs.get(tag, 0):
                with f.open() as fh:
                    fh.seek(self.offs[tag])
                    chunk = fh.read()
                self.offs[tag] = size
                if "attempt 1 succeeded" in chunk or "rebalanced" in chunk:
                    self.wake(seat, f"your book ({tag}) just completed its "
                                    "scheduled run — review what it did")

    def w_artifacts(self):
        for f in ART.glob("*.json"):
            mt = f.stat().st_mtime
            if mt > self.art_seen.get(f.name, 0):
                self.art_seen[f.name] = mt
                try:
                    a = json.loads(f.read_text())
                except Exception:
                    continue
                if a.get("red_team") == "pending":
                    self.wake("red_team",
                              f"new/updated experiment artifact {a['name']} "
                              f"(verdict {a.get('verdict')}) awaits your review "
                              f"under the charter accuracy rules. Artifact:\n"
                              + json.dumps(a)[:3000])
                    self._stamp_artifact(f)

    def _stamp_artifact(self, f):
        memos = sorted(HERE.glob("floor/*/*/red_team.md"),
                       key=lambda x: x.stat().st_mtime)
        if not memos:
            return
        txt = memos[-1].read_text().lower()
        try:
            a = json.loads(f.read_text())
            a["red_team"] = ("approved " if "approved" in txt else
                             "objection ") + datetime.now().strftime("%m-%d")
            f.write_text(json.dumps(a, indent=1))
        except Exception:
            pass

    EXP_CAP_DAY = 6          # experiment quality > quantity; protocol holds
    PY = "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"

    def w_sprint(self):
        """Continuous daytime experiment loop (principal's direction 08-27):
        registrar drains proposals; factory runs unrun specs; when the queue
        is dry and the cap allows, the research lead is woken to propose the
        next one from the sourced BACKLOG. All verdicts flow through the
        existing artifact->red_team trigger."""
        if datetime.now().date() != self.exp_day:
            self.exp_day, self.exp_today = datetime.now().date(), 0
        if not (6 <= datetime.now().hour < 20):
            return
        props = list((ROOT / "qr/sprint/proposals").glob("*.json"))
        if props:
            r = subprocess.run([self.PY, "-W", "ignore",
                                str(ROOT / "qr/sprint/registrar.py")],
                               capture_output=True, text=True, timeout=120,
                               cwd=ROOT)
            for line in r.stdout.strip().splitlines():
                feed(f"SPRINT registrar: {line}")
        arts = {f.stem for f in ART.glob("*.json")}
        pending = [s for s in (ROOT / "qr/sprint/specs").glob("*.json")
                   if s.stem not in arts and not s.stem.startswith("C")]
        if pending and self.exp_today < self.EXP_CAP_DAY:
            spec = pending[0].stem
            self.exp_today += 1
            feed(f"SPRINT factory: running {spec} "
                 f"(experiment {self.exp_today}/{self.EXP_CAP_DAY} today)")
            r = subprocess.run([self.PY, "-W", "ignore",
                                str(ROOT / "qr/sprint/factory.py"), spec],
                               capture_output=True, text=True, timeout=1800,
                               cwd=ROOT)
            feed(f"SPRINT factory: {spec} finished rc={r.returncode}")
            return                      # artifact trigger handles the review
        if (not pending and not props and self.exp_today < self.EXP_CAP_DAY
                and time.time() - self.last_lead_wake > 5400):
            self.last_lead_wake = time.time()
            self.wake("research_lead",
                      "the experiment queue is EMPTY and today's cap allows "
                      f"more ({self.exp_today}/{self.EXP_CAP_DAY} run). Consult "
                      "the BACKLOG in your briefing and propose the single "
                      "best next spec as ```json``` (new signals available: "
                      "taker_imbalance, qhour_open — order-flow data is live).")

    def w_chief(self, mkt):
        if self.filings_since_chief >= 4 and self.can("chief_of_desk"):
            self.filings_since_chief = 0
            self.wake("chief_of_desk",
                      "several seats have filed since your last synthesis — "
                      "synthesize the recent memos into a fresh desk note")
        elif mkt and time.time() - self.last_any_wake > 5400:
            self.wake("risk_officer", "patrol: no events for 90 minutes — "
                                      "routine sweep of the account and books")


def main():
    d = D()
    feed("DISPATCHER online — event-driven mode: agents wake on triggers only")
    n = 0
    while True:
        try:
            mkt = d.market_open() if n % 6 == 0 else mkt          # noqa: F821
        except NameError:
            mkt = d.market_open()
        try:
            d.w_prices(mkt)
            if n % 3 == 0:                       # ~60s: broker-state watchers
                d.w_orders(); d.w_book(); d.w_account()
                d.w_booklogs(); d.w_artifacts()
                d.w_sprint()
                d.w_chief(mkt)
        except Exception as e:
            feed(f"DISPATCHER watcher error: {str(e)[:90]}")
        n += 1
        time.sleep(20)


if __name__ == "__main__":
    main()
