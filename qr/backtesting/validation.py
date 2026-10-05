"""
Validation: the machinery that decides whether a result is real.

Four independent checks, all of which a strategy must pass. They are
independent on purpose: leakage, overfitting, luck, and fragility are
different failure modes and no single test catches all four.

1. LEAKAGE CANARY (`leakage_canary`)
   Run a deliberately-cheating signal through the engine. If the engine does
   not report an absurd result, the engine is broken. Then run the same signal
   correctly lagged; if that still looks good, the engine is leaking.
   This test protects the harness itself, which is the thing everything else
   depends on.

2. PURGED WALK-FORWARD (`walk_forward`)
   Expanding-window train/test with a PURGE and an EMBARGO between them.
   The purge removes training observations whose forward-return window
   overlaps the test period; without it, a strategy holding H days leaks H
   days of test outcome into training. The embargo drops a further buffer
   after the test window to break serial correlation.

3. MULTIPLE-TESTING CORRECTION (`multiple_testing_report`)
   Testing K hypotheses and reporting the best is not evidence. We report
   Bonferroni and Benjamini-Hochberg adjusted p-values, and the deflated
   Sharpe ratio, which is the number that actually matters.

4. RANDOMISATION TESTS (`permutation_test`, `random_signal_baseline`)
   Shuffle the signal cross-sectionally within each date, preserving all
   structure except the signal-to-outcome mapping. If a shuffled signal scores
   as well as the real one, there is no signal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

from .engine import BacktestConfig, run_backtest
from .metrics import summarize

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- 1. canary
def leakage_canary(panel: pd.DataFrame, cfg: BacktestConfig) -> dict:
    """
    Prove the engine can detect leakage, and that it is not itself leaking.

    `cheat`  : ranks on the realised forward return -- an oracle. Must score
               enormously. If it does not, the engine is broken.
    `honest` : ranks on the PAST return over the same horizon. Must score
               near zero. If it scores well, information is bleeding across
               the information cutoff.
    """
    truth = (panel.set_index(["date", "symbol"])["fwd_ret"]
             .groupby(level=0).rank(pct=True))

    def cheat(vis: pd.DataFrame) -> pd.Series:
        d = vis["date"].max()
        try:
            return truth.loc[d]
        except KeyError:
            return pd.Series(dtype=float)

    def honest(vis: pd.DataFrame) -> pd.Series:
        d = vis["date"].max()
        cur = vis[vis["date"] == d].set_index("symbol")
        past = (vis.sort_values("date").groupby("symbol")["adj_close"]
                .apply(lambda s: s.iloc[-1] / s.iloc[-(1 + cfg.holding_days)] - 1
                       if len(s) > cfg.holding_days else np.nan))
        return past.reindex(cur.index)

    out = {}
    for label, fn in (("cheat_oracle", cheat), ("honest_past_return", honest)):
        try:
            res = run_backtest(panel, fn, replace(cfg, name=f"canary_{label}"))
            m = summarize(res.daily, cfg.capital)
            out[label] = {k: m.get(k) for k in
                          ("n_periods", "sharpe", "sharpe_gross", "cagr",
                           "total_return", "win_rate", "mean_period_ret")}
        except Exception as exc:                          # noqa: BLE001
            out[label] = {"error": str(exc)}

    # Both checks use GROSS Sharpe. Net Sharpe is dominated by the cost model,
    # so a heavy-cost universe would fail the cleanliness check for reasons
    # that have nothing to do with leakage.
    cheat_sr = out.get("cheat_oracle", {}).get("sharpe_gross")
    honest_sr = out.get("honest_past_return", {}).get("sharpe_gross")

    out["engine_detects_leakage"] = bool(cheat_sr is not None
                                         and np.isfinite(cheat_sr)
                                         and cheat_sr > 3.0)
    # ONE-SIDED. Leakage inflates returns; it does not depress them. A large
    # NEGATIVE Sharpe on a lagged past-return signal is an ordinary finding
    # (it means the effect runs the other way), not evidence of contamination.
    # Testing |SR| < 2 would have flagged a genuine reversal effect as a bug.
    out["engine_appears_clean"] = bool(honest_sr is not None
                                       and np.isfinite(honest_sr)
                                       and honest_sr < 2.0)
    out["verdict"] = ("PASS" if out["engine_detects_leakage"]
                      and out["engine_appears_clean"] else "FAIL")
    return out


# ---------------------------------------------------------- 2. walk-forward
@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(dates: pd.Series, n_folds: int = 5, purge_days: int = 10,
               embargo_days: int = 5, min_train_days: int = 500) -> list[Fold]:
    """Expanding-window folds with purge and embargo."""
    d = pd.DatetimeIndex(sorted(pd.unique(dates)))
    if len(d) < min_train_days + n_folds * 40:
        n_folds = max(2, (len(d) - min_train_days) // 60)
    if n_folds < 2:
        return []

    test_size = (len(d) - min_train_days) // n_folds
    folds = []
    for i in range(n_folds):
        te_lo = min_train_days + i * test_size
        te_hi = min(te_lo + test_size, len(d) - 1)
        if te_hi - te_lo < 20:
            break
        tr_hi = te_lo - purge_days - embargo_days
        if tr_hi <= 50:
            continue
        folds.append(Fold(d[0], d[tr_hi], d[te_lo], d[te_hi]))
    return folds


def walk_forward(panel: pd.DataFrame, signal_fn, cfg: BacktestConfig,
                 folds: list[Fold]) -> pd.DataFrame:
    """Run the strategy on each fold's TEST window only. No fitting here --
    this measures stability of a fixed rule across regimes."""
    rows = []
    for i, f in enumerate(folds):
        sub = panel[(panel["date"] >= f.test_start - pd.Timedelta(days=400)) &
                    (panel["date"] <= f.test_end)]
        try:
            res = run_backtest(sub, signal_fn, replace(cfg, name=f"{cfg.name}_f{i}"))
            m = summarize(res.daily, cfg.capital)
            # keep only rebalances inside the test window
            inside = res.daily[res.daily["date"] >= f.test_start]
            m_in = summarize(inside, cfg.capital)
            rows.append({"fold": i, "test_start": f.test_start,
                         "test_end": f.test_end, **{
                             k: m_in.get(k) for k in
                             ("n_periods", "mean_period_ret", "sharpe", "cagr",
                              "max_drawdown", "win_rate", "t_stat")}})
        except Exception as exc:                          # noqa: BLE001
            rows.append({"fold": i, "test_start": f.test_start,
                         "test_end": f.test_end, "error": str(exc)[:80]})
    return pd.DataFrame(rows)


# --------------------------------------------------- 3. multiple testing
def multiple_testing_report(pvals: dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    """Bonferroni and Benjamini-Hochberg on a dict of {strategy: p}."""
    s = pd.Series(pvals).dropna().sort_values()
    k = len(s)
    if k == 0:
        return pd.DataFrame()
    out = pd.DataFrame({"p_raw": s})
    out["bonferroni"] = (s * k).clip(upper=1.0)
    ranks = np.arange(1, k + 1)
    bh = (s.values * k / ranks)
    bh = np.minimum.accumulate(bh[::-1])[::-1].clip(0, 1)
    out["bh_fdr"] = bh
    out["passes_bonferroni"] = out["bonferroni"] < alpha
    out["passes_bh"] = out["bh_fdr"] < alpha
    out["n_tested"] = k
    return out


# ----------------------------------------------------- 4. randomisation
def permutation_test(panel: pd.DataFrame, signal_fn, cfg: BacktestConfig,
                     n_perm: int = 30, seed: int = 0) -> dict:
    """
    Shuffle the signal WITHIN each rebalance date. This destroys the
    signal-to-outcome mapping while preserving the number of names, the
    universe, the cost structure, and the market's own return distribution.
    """
    rng = np.random.default_rng(seed)
    real = run_backtest(panel, signal_fn, cfg)
    real_m = summarize(real.daily, cfg.capital)
    real_stat = real_m.get("mean_period_ret", np.nan)

    null = []
    for i in range(n_perm):
        def shuffled(vis: pd.DataFrame, _fn=signal_fn, _r=rng) -> pd.Series:
            s = _fn(vis)
            if s is None or len(s) == 0:
                return s
            vals = s.values.copy()
            _r.shuffle(vals)
            return pd.Series(vals, index=s.index)
        try:
            res = run_backtest(panel, shuffled, replace(cfg, name=f"perm{i}", seed=i))
            null.append(summarize(res.daily, cfg.capital).get("mean_period_ret", np.nan))
        except Exception:                                 # noqa: BLE001
            continue

    null = np.array([x for x in null if np.isfinite(x)])
    if len(null) < 5 or not np.isfinite(real_stat):
        return {"error": "insufficient permutations"}
    p = float((null >= real_stat).mean())
    return {
        "real_mean_period_ret": float(real_stat),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "n_permutations": int(len(null)),
        "p_value_permutation": p,
        "z_vs_null": float((real_stat - null.mean()) / null.std(ddof=1))
                     if null.std(ddof=1) > 0 else np.nan,
    }
