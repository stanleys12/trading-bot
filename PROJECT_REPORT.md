# Small-Cap Quantitative Trading Research — Project Report

**Repository:** `~/trading-bot`
**Period:** August 2026
**Status:** Research concluded. Paper forward-test running.
**Headline:** 212 pre-registered trials, 0 promoted candidates, sealed test set never opened.

---

## 1. The Goal

Find a small, genuine, economically plausible statistical edge in US small-cap
equities that survives:

- realistic transaction costs and slippage
- liquidity constraints and capacity limits
- market and sector exposure controls (alpha, not beta)
- survivorship bias
- look-ahead bias
- point-in-time data constraints
- multiple-testing / researcher overfitting
- completely unseen data

The objective was explicitly **not** the best-looking backtest. It was to
determine whether an edge exists, and to say so honestly either way.

---

## 2. Where This Started

The inherited system (`src/`) reported **738% returns** and an **86% win rate**
across three strategies. Both figures were artefacts.

| Defect | Detail |
|---|---|
| Look-ahead bias | `RVOL` used the day's **own total volume** to decide at 9:30am. Reproduced at 86.0% win rate; replacing the one unknowable input with its lagged equivalent gave **42.9%**, against a 43.7% no-filter baseline. |
| Hindsight position sizing | `simulate_strategy` sized positions as `1% / abs(low_from_open)` — the day's realised low. Correlation between position size and that day's return: **+0.578**. A correct sizer scores 0.00. |
| Survivorship | Universe was 80 hand-picked meme tickers; 12 no longer returned data. |
| Multiple testing | ~12,000 filter combinations searched, winners with n = 29–50, no correction, no holdout. |

Four live risk rails were also inert: the daily loss limit could never fire
(percentages accumulated into a dollar-denominated check), the circuit breaker
raised `NameError` on the line that triggered it, `record_trade_result` was
never called from anywhere, and premarket orders carried no stop.

---

## 3. What Was Built

**~7,000 lines, 23 unit tests, all passing.**

```
qr/
├── data/            acquisition; knows nothing about strategies
│   ├── universe.py           point-in-time universe from archived exchange directories
│   ├── prices.py             OHLCV panel, raw prices + separate adjustment factor
│   ├── corporate_actions.py  split history; true traded price reconstruction
│   ├── sec_fundamentals.py   SEC XBRL with filing-date point-in-time joins
│   ├── fundamentals.py       sector / size metadata
│   ├── etf_universe.py       cross-asset ETF panel, 2003–2026
│   └── spread_audit.py       measured bid/ask from consolidated SIP quotes
├── backtesting/     the engine; knows nothing about any specific signal
│   ├── costs.py              spread estimation, sqrt-law impact, borrow, commission
│   ├── engine.py             cross-sectional engine with a hard information cutoff
│   ├── metrics.py            Sharpe, Sortino, PSR, deflated Sharpe, CVaR, turnover
│   ├── validation.py         leakage canary, purged walk-forward, permutation, FDR
│   ├── bias_probes.py        behavioural probes adapted from freqtrade
│   └── evaluation.py         residual alpha, survivorship bounding, scorecard
├── strategies/      pure functions, no I/O, no state
│   ├── signals.py            17 signals across 8 categories, each with a hypothesis
│   └── allocation.py         vol targeting, TSMOM, risk parity, trend+RP
├── portfolio/       sizing, ADV caps, capacity
├── execution/       paper trading only
│   ├── paper.py              order simulation and journalling
│   ├── live_paper.py         Alpaca paper runner (live trading structurally blocked)
│   └── daily_run.py          scheduled daily forward test
├── research/        one question per script
│   ├── protocol.py           locked splits, frozen specs, sealed test set, registry
│   ├── screen.py             the pre-registered systematic screen
│   ├── exp01..exp08          individual experiments
├── tests/           23 tests including the leakage canary
├── configs/         every run reproducible from one file
└── reports/         RESEARCH_LOG.md + experiment_registry.jsonl
```

### Three design decisions that mattered

**1. The information cutoff is structural, not conventional.**
A strategy is a function handed a frame truncated at the decision date and
stripped of every forward-looking column. It cannot read the future because
the future is not in the object it receives.

```
decision date  D   sees data through D's close
entry              D+1 open
exit               D+1+H open
```

**2. Costs are estimated from data, then later measured.**
No flat assumption anywhere. Initially Corwin-Schultz (2012) high-low spread
estimation; ultimately replaced by 2 million real quotes (§7).

**3. The harness is tested before its output is believed.**
`leakage_canary` runs an oracle (must score enormously — proving the engine
can *see* leakage) and an honest lagged signal (must not — proving the engine
isn't *causing* it). Verdict: **PASS**, oracle Sharpe 19.5, honest −2.48.

### Protocol enforcement, verified working

| Attempt | Result |
|---|---|
| Unfrozen candidate reaching the test set | **refused** |
| Validation Sharpe 0.55 requesting the test set | **refused** ("not for triage") |
| Silently changing a frozen specification | **refused** |
| Logging a failure without a reason | **refused** |

---

## 4. Data Assets

| Asset | Detail |
|---|---|
| Point-in-time universe | 18 archived Nasdaq symbol directories (Internet Archive), 2018-11 → 2026-08. **9,317 symbols ever listed; 3,905 (41.9%) departed.** |
| Price panel | 14.4M rows, 5,137 symbols, 2010–2026. Raw OHLC + separate adjustment factor. Frozen: `panel_v5_full.parquet`, sha `a8ebf9e9`. |
| Corporate actions | 2,973 split events; **2,126 reverse splits across 1,157 symbols.** |
| Fundamentals | SEC XBRL companyfacts, 6.58M facts, 4,594 companies, joined **as-of filing date** — genuinely point-in-time. |
| Sector / size | 2,558 symbols labelled (98% of liquid names). |
| ETF panel | 15 cross-asset ETFs, 2007–2026, including the GFC. |
| **Measured spreads** | **2,052,834 consolidated SIP quotes**, 202 symbols stratified by ADV, 21 dates 2016–2026. |

### The binding data constraint

**Delisted price recovery: 10.7%.** Of 3,905 departed symbols, only 416 have
recoverable price history — versus 95.1% for active names.

Coverage by era, which governs how much to trust each period:

| Year | In panel | Actually listed | Coverage |
|---|---|---|---|
| 2010 | 2,390 | — | ~49% |
| 2018 | 3,372 | 4,883 | **69%** |
| 2025 | 5,047 | 5,336 | **95%** |

A rising coverage ratio **manufactures fake alpha decay** — early years are
more survivor-contaminated, so any strategy looks stronger early and weaker
late with no change in the market. Split boundaries were drawn where data
quality changes, not at round numbers:

```
RESEARCH    2010–2017   survivor-only, bias UNMEASURED — hypotheses only
VALIDATION  2018–2022   point-in-time universe, bias MEASURED
TEST        2023–2026   SEALED, never opened
```

---

## 5. Experiments and Results

### Phase 1 — small-cap stock selection

| Candidate | Peak result | Cause of death |
|---|---|---|
| RVOL Rocket (inherited) | 86% win rate | Look-ahead. Lagged: 42.9% |
| Dead Cat Fade (inherited) | 79% win rate | Same leak; lagged = the no-filter baseline exactly |
| Panic Bounce (inherited) | +2.2%/trade | 161% of profit from 3 of 50 symbols; negative 2024–26 |
| 17-signal screen | — | **0 of 34** passed Bonferroni/BH. Gross Sharpe 0.16 median |
| Momentum, microcap | Sharpe 4.50 | Split contamination (→1.74), then invented borrow rate (→0.19) |
| Low-price premium | t = 6.78 | Lottery payoff: median −0.30%, 64% of return from 1% of names |
| 5-day reversal, illiquid | t = 6.7 | **Real effect.** 85× turnover × 120bps = −60.7%/yr |
| Value / quality, annual | 18.2% CAGR | It was beta 1.13. **Alpha vs SPY −1.12%** |
| Sector-neutral ROE | t = 4.34 | Cleared Bonferroni at K=204. **Validation: t = 1.48** |
| Net share issuance | t = 3.73 | Real, survives value/quality controls — but lives in **large** caps |

### The two effects that are real but unusable

**Short-term reversal in illiquid names.** t = 6.7, and it survives the
bid-ask-bounce test decisively: its IC is *higher* in tradeable open-to-open
returns (+0.0238) than close-to-close (+0.0192) — the opposite of a quoting
artefact — and decays gradually across skip days rather than collapsing.
It loses money at **every** holding period from 5 to 42 days.

| Hold | Turnover | Sharpe | CAGR |
|---|---|---|---|
| 5d | 84.7× | −2.23 | −60.7% |
| 21d | 21.4× | −0.51 | −25.1% |
| 42d | 11.0× | −0.15 | −11.6% |

**Reverse-split distress.** Mean −2.80%, median −4.38%, 40% of names up,
1.7× delisting risk. Both mean and median agree, so not outlier-driven.
Touches 0.6% of the universe → moves portfolio Sharpe by ~0.01.
Permanent free hygiene, not a strategy.

### The alpha–cost gradient (thesis confirmed, and self-defeating)

5-day reversal by liquidity quintile:

| Quintile | Median ADV | IC | t | Gross spread | Est. cost | Net |
|---|---|---|---|---|---|---|
| Q1 | $1.1M | 0.0192 | **6.7** | 99bp | 85bp | −71bp |
| Q3 | $14.7M | 0.0096 | 3.0 | 46bp | 77bp | −108bp |
| Q5 | $200M | 0.0109 | 2.4 | 20bp | 54bp | −88bp |

**Alpha falls 5× from illiquid to liquid; cost falls only 1.6×.** The premise
that inefficiency concentrates where big money won't go is *correct*. The
problem is that the best place to look is also the most expensive to trade.

### Informative censoring — the deepest problem

Net share issuance should be strongest in microcaps. It measured **zero**
there (IC −0.003) and strongest in the most liquid quintile (+0.097).

| Cohort | n | Median NSI | Diluting >10%/yr |
|---|---|---|---|
| Survivors | 99,335 | +0.0038 | 13% |
| **Later delisted** | 270 | +0.0089 | **34%** |

In the illiquid quintile, names that later vanished diluted at **+0.0404 vs
survivors' +0.0092** — four times harder. Heavy dilution → delisting →
absent from panel, and 89% of departures are unpriced.

That is missingness correlated with **both** the predictor and the outcome.
The microcap question is not answered "no" — it is **unanswerable with this
data**, because the process under study is the process that deleted the
evidence.

### The terminal arithmetic

Two independent signals produced alpha stable across periods:

| Signal | Research | t | Validation | t |
|---|---|---|---|---|
| low volatility | +4.65% | 2.81 | **+4.82%** | **1.30** |
| net share issuance | +4.56% | 2.29 | **+4.97%** | **0.95** |

The alpha did not decay — it grew. The **t-statistic** collapsed.

```
information ratio = 4.8% / 17% = 0.28
t = IR × √years   →   years = (2 / 0.28)² = 51
```

**Fifty-one years** to distinguish this alpha from zero at t = 2. No data
purchase changes this. Anything with IR > 1 (provable in ~4 years) is exactly
what large funds compete away.

### Phase 2 — portfolio construction on liquid ETFs

Full cycle 2007–2020, 15 assets, vol-matched with financing charged:

| Strategy | CAGR | Vol | Sharpe | MaxDD | vs 60/40 |
|---|---|---|---|---|---|
| **Static 60/40** | 8.5% | 11.5% | **0.77** | −31.4% | — |
| Trend (TSMOM) | 5.3% | 8.0% | 0.69 | −20.4% | −1.30% |
| Trend + risk parity | 5.1% | 8.0% | 0.66 | −23.4% | −1.60% |
| Risk parity | 4.5% | 7.3% | 0.64 | −23.2% | −2.01% |
| Buy & hold SPY | 9.3% | 20.8% | 0.53 | −55.2% | −2.67% |
| Vol-targeted SPY | 5.9% | 12.9% | 0.51 | −30.8% | −2.94% |

**A static 60/40 beat every adaptive rule.** And volatility targeting was the
*worst* crisis performer, not the best — in 2020 it returned **−13.8% while
SPY gained +18.3%**, because trailing volatility told it to de-risk after the
March crash. It sold the bottom.

| Mean return 2008/2018/2020 | |
|---|---|
| 60/40 | **−0.6%** |
| Trend | −3.7% |
| SPY | −7.7% |
| **Vol targeting** | **−10.4%** |

**Even 60/40's advantage was a falling-rate era.** The static ratio grid
showed Sharpe rising monotonically as equity weight fell — then inverted:

| Equity weight | 2007–14 | 2015–20 | **2021–26** |
|---|---|---|---|
| 30% | **1.14** | **1.31** | **0.48** |
| 60% | 0.65 | 0.95 | 0.82 |
| 100% | 0.41 | 0.74 | **0.94** |

IEF returned −1.9%/yr and TLT −8.1% with a −44% drawdown in the recent era.
"Diversification beats concentration" describes a forty-year rate decline, not
a law.

### Levered low-vol (exp08) — the one adjustment with a real prior

Frazzini-Pedersen: the low-beta premium exists *because* investors are
leverage-constrained, so the implied trade is to lever low-beta.

| Window | Variant | CAGR | MaxDD | vs SPY |
|---|---|---|---|---|
| Research | SPY | 15.9% | −18.6% | — |
| | levered 1.27× @6.5% | 17.3% | −18.8% | **+1.47%** |
| Validation | SPY | 10.1% | −33.7% | — |
| | levered 1.15× @6.5% | 12.5% | **−41.6%** | **+2.39%** |

Beat SPY in both windows, and still beat it at 9% financing. **Rejected** on
drawdown: −41.6% vs −33.7%. Also used hindsight leverage ratios, and a 1–2%
excess against 17% tracking error is IR ≈ 0.1 — weaker than the 0.28 that
already needed 51 years.

---

## 6. Errors Caught — Including Four of Our Own

The harness caught the inherited 738%/86% mirage, then caught four errors we
introduced:

1. **Split contamination in our own pipeline.** Yahoo's `Close` is
   split-adjusted *even with* `auto_adjust=False` — that flag disables only
   dividends. ADTX, after seven reverse splits, appeared at a nominal
   **$115,450** and peaked at **$1.49bn per share**. 13% of panel rows carried
   fictitious prices. This was the exact defect flagged in the original audit,
   reproduced by us.
2. **A fabricated borrow rate.** A Sharpe-4.50 candidate drew 77% of its
   profit from shorting unborrowable microcaps at an assumed 5%/yr. At a
   realistic 50%, Sharpe fell to **0.19**.
3. **A validation kill at t = 4.34.** Sector-neutral ROE cleared Bonferroni at
   K=204 and collapsed to **t = 1.48** out of sample.
4. **A frequency-mismatched regression.** 21-day strategy returns regressed on
   1-day benchmark returns removed almost no beta and produced alphas of
   **+10% to +19%** — pure artefact.

Plus three **silent guards** that discarded data without complaining:
rate-limited download batches cached as empty; a half-covered, alphabetically
biased sector map; and a portfolio function that dropped all 20 variants.
All now fail loudly.

> **All of these would have looked like good news if reported unchecked.**
> That is the argument for the harness — not that it finds edges, but that the
> error rate in this work is high enough that an unverified positive should be
> assumed wrong.

---

## 7. The Cost Floor, Re-derived from Measured Data

Every cost conclusion originally rested on Corwin-Schultz. Alpaca's **free**
tier turned out to serve historical consolidated SIP quotes back to 2016 —
real NBBO, not the IEX subset.

| ADV bucket | **Measured** | Corwin-Schultz | Overstated by |
|---|---|---|---|
| <$2M | 89.9 bps | 169.6 | 1.9× |
| $2–10M | 30.6 bps | 79.1 | 2.6× |
| $10–50M | 16.2 bps | 60.5 | 3.7× |
| $50–200M | 7.5 bps | 53.5 | **7.1×** |
| $200M–1B | 4.6 bps | 46.1 | **10.0×** |
| >$1B | 2.8 bps | 47.5 | **17.1×** |

Fitted model: `log10(spread_bps) = 4.288 − 0.416 × log10(ADV)`, R² = 0.587.

Re-running the central screen at **23.7bps** instead of ~120bps changed
**nothing**: low volatility moved from +4.65%/+4.82% to +4.45%/+3.89%, and
**zero signals reached alpha t > 2 in both splits.**

**The null result was never about costs.** No cost correction reaches a signal
that isn't there. It also confirmed the microcap verdict for the right reason
(measured <$2M spread 89.9bps vs 85bps assumed), and removed the strongest
remaining argument for buying data.

---

## 8. What Is Running Now

**Strategy:** low-volatility, long-only, no leverage.

| Parameter | Value |
|---|---|
| Universe | US common stocks, NYSE/NASDAQ, ADV ≥ $50M (~1,400 names) |
| Signal | trailing 63-day realised volatility, annualised |
| Selection | lowest volatility, equal weighted |
| Rebalance | monthly (first trading day) |
| Capital | $100,000 **paper money** |
| Schedule | launchd, 06:40 PDT weekdays (survives sleep; cron does not) |

**Known specification drift:** the live runner holds **25 names**; the frozen
validated spec was **quarterly rebalance at the top 20% quantile** (~280
names). That is 11× the concentration and 3× the turnover. Flagged, not yet
corrected.

**Economic rationale:** leverage-constrained investors cannot borrow to lever
safe stocks, so they buy high-beta stocks instead — bidding those up and
leaving low-beta names cheap. The premium persists because the constraint
persists (Black 1972; Frazzini-Pedersen 2014).

**Expected behaviour** given beta ≈ 0.73:

| If SPY does | Expect roughly | |
|---|---|---|
| +30% | +26% | loses |
| +15% | +15% | ties |
| 0% | +4% | wins |
| −20% | −11% | wins big |

**Purpose is measurement, not profit.** The strategy is available as USMV for
15bps with more names and better optimisation — our version is strictly worse
as an investment. It exists to measure **realised slippage against the
modelled 3–5bps**, which is the last open methodological question.

The dry runs already caught two operational failures no backtest encounters:
a universe that would have silently traded bond ETFs (because "lowest
volatility" over a universe including ARCA selects bond funds), and
after-hours IEX quotes showing **3,000bps spreads** with BAC quoted 0.00/0.00.

**Commands:**
```bash
./check.sh                                      # account, positions, daily record
python3 -m qr.execution.live_paper dryrun       # signal only, no orders
python3 -m qr.execution.live_paper rebalance    # during 09:30–16:00 ET
python3 -m qr.execution.live_paper reconcile    # at T+2, vs historical SIP NBBO
```

Live trading is structurally impossible: `_assert_paper` gates every order path
and raises on a live endpoint.

---

## 9. Conclusions

1. **Free daily data on US equities does not contain a retail-tradeable edge.**
   212 pre-registered trials across six independent information families.

2. **The two effects that are genuinely real are unusable.** Short-term
   reversal in illiquid names is too expensive to harvest at any holding
   period. The low-volatility anomaly is too small to prove and purchasable as
   an ETF.

3. **The binding constraint is statistical, not informational.** An IR of 0.28
   needs ~51 years. Anything provable quickly is already arbitraged. We were
   not failing to find alpha — we were failing to find alpha *large enough to
   prove*, which the market actively enforces.

4. **The small-cap thesis is directionally correct and self-defeating.** Alpha
   is 5× larger in the least-liquid quintile. So is cost, and the alpha curve
   is steeper — but not steeply enough.

5. **Some questions are unanswerable with this data, not merely unanswered.**
   Microcap dilution is censored by the very process it describes.

6. **The instrument is the deliverable.** A harness that killed a t = 4.34
   finding, caught four of its own errors, and never opened its test set is
   worth more than a strategy found without one.

---

## 10. Reproducibility

```bash
python3 qr/data/universe.py              # point-in-time universe
python3 qr/data/prices.py                # price panel
python3 qr/data/corporate_actions.py     # splits, true prices
python3 qr/data/sec_fundamentals.py      # SEC XBRL, PIT joins
python3 -m pytest qr/tests -q            # must be green (23 tests)
python3 qr/research/screen.py            # the systematic screen
```

- Frozen panel: `panel_v5_full.parquet`, sha `a8ebf9e9`
- Config: `qr/configs/screen_v1.yaml`
- Registry: `qr/reports/experiment_registry.jsonl` (every trial, with reasons)
- Log: `qr/reports/RESEARCH_LOG.md`

**Data sources and limitations:** archived Nasdaq symbol directories via the
Internet Archive (coverage begins 2018-11); prices via yfinance (unversioned,
silently revised, 10.7% delisted recovery); fundamentals from SEC XBRL
companyfacts (tag mapping is a first-available heuristic; original filings
kept, not restatements); spreads measured from Alpaca consolidated SIP quotes
(free tier, back to 2016, blocked for the most recent ~24h).

---

*Final tally: 212 trials. 0 candidates. Test set never opened.*
