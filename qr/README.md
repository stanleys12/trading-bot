# qr — quantitative research system

A backtesting and research framework built to answer one question honestly:
**is there a tradeable edge here, or does it only look like there is?**

The previous system in `../src/` reported 738% returns and an 86% win rate.
Both were artefacts. This system is designed so that class of error cannot
happen silently. See `reports/RESEARCH_LOG.md` for what has been tested and
what it found.

---

## Architecture

```
qr/
├── data/            acquisition and cleaning. Knows nothing about strategies.
│   ├── universe.py      point-in-time universe from archived symbol directories
│   ├── prices.py        OHLCV panel; raw prices + separate adjustment factor
│   ├── raw/             immutable downloads
│   ├── interim/         parsed, not yet validated
│   └── processed/       the panel everything else reads
├── backtesting/     the engine. Knows nothing about any specific signal.
│   ├── costs.py         spread estimation, impact, commission, borrow
│   ├── engine.py        cross-sectional engine with a hard information cutoff
│   ├── metrics.py       performance/risk metrics, PSR, deflated Sharpe
│   └── validation.py    canary, purged walk-forward, permutation, FDR
├── strategies/      signal definitions. Pure functions, no I/O, no state.
│   └── signals.py       17 signals across 8 categories, each with a hypothesis
├── portfolio/       position sizing and construction rules
├── execution/       paper trading only
│   └── paper.py         LIVE_TRADING_ENABLED = False, by design
├── research/        experiment scripts. One question per script.
│   ├── screen.py            the pre-registered systematic screen
│   ├── liquidity_tiers.py   does any liquidity tier make costs survivable
│   └── deep_dive.py         four hurdles applied to a single candidate
├── tests/           unit tests for anything that can silently corrupt a result
├── configs/         every run is reproducible from one of these
└── reports/         outputs. RESEARCH_LOG.md is the human-readable record.
```

**Separation rule:** data processing never imports strategy code; strategy code
never imports the engine; the engine never knows what signal it is running.
A strategy is a function `signal(visible_panel) -> Series`. The engine calls it
with a frame that has been stripped of every forward-looking column and
truncated at the information cutoff. **A strategy cannot read the future
because the future is not in the object it is given.**

---

## The three design decisions that matter

### 1. The information cutoff is structural, not a convention

The prior research leaked because features and outcomes lived in the same row
of the same dataframe, and nothing prevented a filter from touching a column it
shouldn't. Here:

```
decision date  D   strategy sees data through D's close, and nothing after
entry              D+1 open
exit               D+1+H open
```

`prepare_panel` computes `fwd_ret` from `next_open` → `exit_open`. `run_backtest`
drops `fwd_ret`, `next_open`, `exit_open`, `raw_next_open` before calling the
signal. A test asserts this (`test_strategy_cannot_see_forward_columns`).

### 2. Costs are estimated from data, and reported under two models

No flat assumption anywhere. Spread comes from the Corwin-Schultz high-low
estimator; impact follows the square-root law scaled by ADV participation;
commission includes the per-order minimum, which matters enormously for small
positions. Shorts pay borrow and are restricted to a conservative
borrowability proxy.

Corwin-Schultz is upward-biased for liquid names (it returns ~55bps where the
truth is ~2bps), so a second literature-calibrated model is included and
`CostConfig.spread_model` selects between them. **Conclusions are reported
under both. Disagreement between them is itself a result.**

### 3. The harness is tested before its output is believed

`validation.leakage_canary` runs an oracle (must score enormously — proving the
engine can *see* leakage) and an honest lagged signal (must not score well —
proving the engine isn't *causing* leakage). If the canary fails, no result
from that run is reported as meaningful.

---

## Reproducing a run

```bash
python qr/data/universe.py            # point-in-time universe
python qr/data/prices.py              # price panel
python -m pytest qr/tests -q          # must be green
python qr/research/screen.py          # the screen
python qr/research/liquidity_tiers.py # cost-tier sweep
python qr/research/deep_dive.py mom_12_1
```

Every parameter lives in `configs/screen_v1.yaml`.

---

## Data sources and their limitations

| Source | Used for | Limitation |
|---|---|---|
| Internet Archive → `nasdaqtraded.txt` | point-in-time universe | starts 2018-11; irregular snapshots, worst gap 721 days; can't separate delisting from rename/merger |
| Yahoo Finance (yfinance) | daily OHLCV | unversioned, silently revised; **only 12% of delisted names recoverable**; no bid/ask; aggressive rate limiting |
| — | bid/ask spreads | **not available.** Estimated, never measured. Largest source of cost-model error. |
| — | short borrow rates | **not available.** Proxied by a price/liquidity exclusion. |
| — | earnings dates, float, index membership, sector | **not available.** Rules out event-conditioned and sector-neutral research. |

**What would change the answer:** Polygon/Massive Stocks Starter ($29/mo) gives
5 years of minute bars plus Flat Files — S3 dumps organised by *date* rather
than ticker, which are survivorship-free by construction. That single change
fixes the 12% recovery problem, enables intraday research, and lets spreads be
measured instead of estimated.

---

## Biggest risks of false discovery in this project

Ranked by how likely they are to fool us, given what we've already seen:

1. **Survivorship bias in the long leg.** 41.9% of the universe departed and we
   recover 12% of it. Mitigated by making everything long-short, which flips
   the bias to conservative — not by pretending it's gone.
2. **Multiple testing.** 34 trials in the screen alone. Mitigated by
   Bonferroni + BH + deflated Sharpe, and by a TEST window scored once.
3. **Cost-model error.** We estimate spreads rather than measure them, and the
   estimator is demonstrably wrong at the liquid end. Mitigated by dual models;
   genuinely resolved only by buying quote data.
4. **Regime-fitting.** 2018–2026 contains COVID and a meme-stock era. A signal
   that works only in 2021 is not a signal. Mitigated by purged walk-forward.
5. **Yahoo data revisions.** Results are not exactly reproducible over time
   because the source mutates. Mitigated by caching raw downloads immutably.
6. **The researcher.** Every "let me just try one more variant" is an untracked
   trial. Mitigated by `RESEARCH_LOG.md`, which records failures with reasons
   and forbids modifying a hypothesis until it works.

---

## Status

**No strategy has been promoted. The TEST window has never been scored.**
Live trading is not implemented and `execution/paper.py` refuses to run if
`LIVE_TRADING_ENABLED` is flipped.
