#!/usr/bin/env python3
"""Red-team calibration: seeded artifacts with KNOWN ground truth.

The principal's directive (2026-08-26): the red team must be accurate, not
a random killer. So it is benchmarked on BOTH error types:
  - recall: seeded-defect artifacts it must object to, naming the right class
  - precision: clean artifacts it must APPROVE (killing them counts against)
Scores append to fund/evals/scores.jsonl — same trust ledger as every seat.
"""
import json, re, sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "fund"))
from orchestrator import run_seat                      # noqa: E402

CASES = {
  "RT1_lookahead": {
    "truth": "OBJECT", "defect_rx": r"lookahead|look-ahead|same bar|same-bar|entry.*signal bar",
    "artifact": {
      "name": "seed_rt1", "verdict": "PASS",
      "spec": {"signal": "breakout", "hold_minutes": 30, "fee_tier": "T2",
               "entry_rule": "enter at the CLOSE of the bar that triggered the breakout signal",
               "symbols": ["BTCUSDT", "ETHUSDT"]},
      "results": {"research": {"n": 4100, "mean_net_bps": 9.1, "t_stat": 6.0},
                  "validation": {"n": 1450, "mean_net_bps": 8.4, "t_stat": 4.9}},
      "powered": True, "same_sign": True,
      "notes": "entry price = close of signal bar; exit = close of bar t+30"}},
  "RT2_cost_omission": {
    "truth": "OBJECT", "defect_rx": r"cost|fee|slippage|T0|zero.?bps",
    "artifact": {
      "name": "seed_rt2", "verdict": "PASS",
      "spec": {"signal": "meanrev", "hold_minutes": 15, "fee_tier": "T0",
               "entry_rule": "next bar open", "symbols": ["BTCUSDT"]},
      "results": {"research": {"n": 9800, "mean_net_bps": 3.2, "t_stat": 5.1},
                  "validation": {"n": 3100, "mean_net_bps": 2.9, "t_stat": 3.4}},
      "powered": True, "same_sign": True,
      "notes": "scored at fee tier T0 (0bps, no slippage); mean edge 2.9bps"}},
  "RT3_clean": {
    "truth": "APPROVE", "defect_rx": None,
    "artifact": {
      "name": "seed_rt3", "verdict": "PASS",
      "spec": {"signal": "volspike_drift", "hold_minutes": 30, "fee_tier": "T2",
               "entry_rule": "next bar open after signal bar closes",
               "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]},
      "results": {"research": {"n": 5200, "mean_net_bps": 7.8, "t_stat": 4.4},
                  "validation": {"n": 1900, "mean_net_bps": 6.9, "t_stat": 3.1}},
      "powered": True, "same_sign": True,
      "notes": "next-bar-open entries, T2 both sides + range-based slippage, "
               "splits verified 2024-01..2026-04 / 2026-05..07, params frozen "
               "pre-scoring, first and only scoring of this spec"}},
}

PROMPT = ("Red-team review under qr/sprint/CHARTER.md accuracy rules. Below is "
          "the complete experiment artifact. Object ONLY with (a) a defect class "
          "from the charter taxonomy, (b) specific evidence in this artifact, "
          "(c) the predicted result if fixed. If no confirmed-class defect "
          "exists, your memo must say APPROVED. Artifact:\n")


def main(seat="red_team"):
    out = Path(__file__).resolve().parents[2] / "fund" / "evals" / "scores.jsonl"
    rows = []
    for cname, c in CASES.items():
        r = run_seat(seat, PROMPT + json.dumps(c["artifact"], indent=1))
        memo = r["memo"].lower()
        objected = bool(r["tail"].get("alerts")) or "approved" not in memo
        if c["truth"] == "OBJECT":
            correct = objected and bool(re.search(c["defect_rx"], memo))
        else:
            correct = (not r["tail"].get("alerts")) and "approved" in memo
        row = {"ts": datetime.now().isoformat(timespec="seconds"),
               "seat": seat, "scenario": cname, "correct": bool(correct),
               "objected": objected, "secs": r["secs"], "suite": "redteam"}
        rows.append(row)
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{cname:20} truth={c['truth']:8} correct={correct} ({r['secs']}s)")
    print(f"\n{sum(r['correct'] for r in rows)}/{len(rows)} correct")


if __name__ == "__main__":
    main()
