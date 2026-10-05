#!/usr/bin/env python3
"""The experiment factory: validate -> run -> score -> artifact.

Deterministic core of the sprint. Agents propose specs (JSON); this module
refuses anything violating the charter, executes registered specs on the
charter's splits, applies the fee ladder, and writes a full artifact for
red-team review. It also runs the two harness controls (C0/C1) that
calibrate both the factory AND the red team.

A spec (qr/sprint/specs/<name>.json):
{
  "name": "sprint01_xxx", "family": "sprint-intraday",
  "hypothesis": "...", "power_argument": "...",
  "symbols": ["BTCUSDT", ...], "signal": "<key in SIGNALS>",
  "params": {...}, "hold_minutes": 30, "fee_tier": "T2",
  "entry_per_day_cap": 10
}
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from qr.sprint.data import load                       # noqa: E402

SPLITS = {"research": ("2024-01-01", "2026-05-01"),
          "validation": ("2026-05-01", "2026-08-01")}
SEALED_FROM = "2026-08-01"
FEES_BPS = {"T0": 0.0, "T1": 2.0, "T2": 10.0, "T3": 25.0}
MIN_VALIDATION_TRADES = 300


# ------------------------------------------------------------ signal library
def sig_breakout(df, p):
    """+1 when close breaks the rolling high of the prior `window` minutes."""
    hi = df.close.rolling(p["window"]).max().shift(1)
    return (df.close > hi).astype(int)

def sig_meanrev(df, p):
    """+1 when the z-score of `ret_window`-minute return < -threshold."""
    r = df.close.pct_change(p["ret_window"])
    z = (r - r.rolling(p["z_window"]).mean()) / r.rolling(p["z_window"]).std()
    return (z < -p["threshold"]).astype(int)

def sig_volspike_drift(df, p):
    """+1 after volume > k * rolling median volume with positive bar."""
    med = df.volume.rolling(p["window"]).median().shift(1)
    up = df.close > df.open
    return ((df.volume > p["k"] * med) & up).astype(int)

def sig_taker_imbalance(df, p):
    """+1 when taker-buy share of volume is extreme (order-flow imbalance).
    Literature: order flow predicts crypto returns in-sample up to multi-day
    lags (EFMA 2025). imb = taker_buy/volume - 0.5, z-scored over z_window."""
    imb = (df.taker_buy / df.volume.replace(0, np.nan)) - 0.5
    r = imb.rolling(p.get("smooth", 30)).mean()
    zw = p.get("z_window", 1440)
    z = (r - r.rolling(zw).mean()) / r.rolling(zw).std()
    return (z > p.get("threshold", 2.0)).astype(int)

def sig_qhour_open(df, p):
    """Quarter-hour boundary effect (arXiv 2607.09426): periodic algo flow
    clusters at :00/:15/:30/:45; enter at the boundary when the PRIOR
    interval's drift agrees with prior taker imbalance."""
    m = df.ts.dt.minute % 15
    boundary = (m == 0)
    ret_prior = df.close.pct_change(p["lookback"])
    imb = (df.taker_buy / df.volume.replace(0, np.nan) - 0.5).rolling(p["lookback"]).mean()
    return (boundary & (ret_prior > 0) & (imb > p["imb_min"])).astype(int)

def sig_random(df, p):
    """C0 control: coin-flip entries at the same average rate as ~10/day."""
    rng = np.random.default_rng(p.get("seed", 0))
    return pd.Series(rng.random(len(df)) < p.get("rate", 10 / 1440),
                     index=df.index).astype(int)

SIGNALS = {"breakout": sig_breakout, "meanrev": sig_meanrev,
           "volspike_drift": sig_volspike_drift, "random": sig_random,
           "taker_imbalance": sig_taker_imbalance, "qhour_open": sig_qhour_open}


# ------------------------------------------------------------------ engine
def run_split(spec, split, planted_edge_bps=0.0):
    s0, s1 = SPLITS[split]
    fee = FEES_BPS[spec["fee_tier"]]
    hold = spec["hold_minutes"]
    trades = []
    for sym in spec["symbols"]:
        df = load(sym, s0, s1)
        if len(df) < 10_000:
            continue
        sig = SIGNALS[spec["signal"]](df, spec["params"]).fillna(0)
        entries = df.index[sig == 1]
        # non-overlapping entries, capped per day
        last_exit, day_counts = -1, {}
        for i in entries:
            if i <= last_exit or i + hold >= len(df):
                continue
            day = df.ts.iloc[i].date()
            if day_counts.get(day, 0) >= spec["entry_per_day_cap"]:
                continue
            day_counts[day] = day_counts.get(day, 0) + 1
            last_exit = i + hold
            entry_px = df.open.iloc[i + 1]          # NEXT bar open: no lookahead
            exit_px = df.open.iloc[i + hold + 1] if i + hold + 1 < len(df) \
                else df.close.iloc[i + hold]
            slip = max(1.0, min(10.0, 1e4 * (df.high.iloc[i + 1] - df.low.iloc[i + 1])
                                / max(entry_px, 1e-9) / 2))
            gross = (exit_px / entry_px - 1) * 1e4 + planted_edge_bps
            net = gross - 2 * fee - 2 * slip
            trades.append({"symbol": sym, "ts": str(df.ts.iloc[i]),
                           "gross_bps": round(gross, 2), "net_bps": round(net, 2)})
    t = pd.DataFrame(trades)
    if t.empty:
        return {"n": 0}
    x = t.net_bps
    return {"n": int(len(t)),
            "mean_net_bps": round(float(x.mean()), 3),
            "t_stat": round(float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), 3)
                      if x.std() > 0 else 0.0,
            "hit_rate": round(float((x > 0).mean()), 3),
            "mean_gross_bps": round(float(t.gross_bps.mean()), 3),
            "per_symbol_n": t.groupby("symbol").size().to_dict()}


def validate_spec(spec) -> list[str]:
    errs = []
    for k in ["name", "hypothesis", "power_argument", "symbols", "signal",
              "params", "hold_minutes", "fee_tier", "entry_per_day_cap"]:
        if k not in spec:
            errs.append(f"missing field {k}")
    if spec.get("signal") not in SIGNALS:
        errs.append(f"unknown signal {spec.get('signal')}")
    if spec.get("fee_tier") not in FEES_BPS:
        errs.append("fee_tier must be T0..T3")
    if spec.get("fee_tier") == "T0":
        errs.append("T0 is mechanism-check only, cannot be the registered tier")
    if len(spec.get("power_argument", "")) < 80:
        errs.append("power_argument too thin (charter: expected trade count "
                    "and minimum detectable edge, stated before scoring)")
    return errs


def run_experiment(spec_name: str, planted_edge_bps=0.0) -> dict:
    spec = json.loads((HERE / "specs" / f"{spec_name}.json").read_text())
    errs = validate_spec(spec)
    if errs:
        return {"name": spec_name, "status": "rejected", "errors": errs}
    res = {sp: run_split(spec, sp, planted_edge_bps) for sp in SPLITS}
    v, r = res["validation"], res["research"]
    powered = v.get("n", 0) >= MIN_VALIDATION_TRADES
    same_sign = (r.get("mean_net_bps", 0) > 0) == (v.get("mean_net_bps", 0) > 0)
    verdict = "PASS" if (powered and v.get("t_stat", 0) >= 2.0
                         and v.get("mean_net_bps", 0) > 0 and same_sign) \
        else "FAIL"
    art = {"name": spec_name, "ts": datetime.now(timezone.utc).isoformat(
               timespec="seconds"), "spec": spec, "results": res,
           "powered": powered, "same_sign": same_sign, "verdict": verdict,
           "planted_edge_bps": planted_edge_bps,
           "status": "complete", "red_team": "pending"}
    (HERE / "artifacts" / f"{spec_name}.json").write_text(json.dumps(art, indent=1))
    if not spec_name.startswith("C"):        # controls stay out of the registry
        try:
            from qr.research.protocol import register, Experiment
            register(Experiment(
                name=spec_name, family="sprint-intraday",
                hypothesis=spec["hypothesis"], features=[spec["signal"]],
                split="research 2024-01..2026-04 / validation 2026-05..07; sealed 2026-08+",
                universe=",".join(spec["symbols"]), rebalance_days=0,
                holding_days=round(spec["hold_minutes"] / 1440, 4),
                n_observations=v.get("n", 0), metrics=v, alpha=None,
                outcome=verdict,
                failure_reason="" if verdict == "PASS" else
                    f"validation net {v.get('mean_net_bps')}bps t={v.get('t_stat')} at {spec['fee_tier']}"))
        except Exception as e:
            art["registry_error"] = str(e)
    return art


if __name__ == "__main__":
    print(json.dumps(run_experiment(sys.argv[1],
          float(sys.argv[2]) if len(sys.argv) > 2 else 0.0), indent=1))
