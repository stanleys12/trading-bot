# 30-Day Intraday Sprint — Charter (pre-registered 2026-08-26)

GOAL. In 30 days, find and validate — or definitively close — intraday
strategies in free, survivorship-clean data. One experiment per day minimum,
executed by the factory, reviewed by the agent research wing, adjudicated by
this charter. This charter is frozen at sprint start; amendments must be
appended, dated, and never retroactive.

WHY INTRADAY. Verdict speed scales with bet count, not calendar time. A
strategy with >=300 independent validation trades supports a t~2 verdict
inside this sprint; the old daily books could not. The cost of this speed is
fees: every result below is net of the registered fee tier or it is fiction.

## Data and splits (crypto 1m, data.binance.vision)
  RESEARCH     2024-01-01 .. 2026-04-30   explore freely
  VALIDATION   2026-05-01 .. 2026-07-31   ONE registered scoring per spec
  SEALED       2026-08-01 .. onward       opens only per protocol.open_test_set
                                          rules + live paper forward test
  SPY minute panel: same three-way split by date, same rules.
  Universe: top-20 USDT pairs by trailing volume AT EACH POINT IN TIME,
  including since-delisted pairs (the whole reason binance.vision was chosen).

## Cost model (registered tiers, per side)
  T0 = 0bps (mechanism check only, never a verdict)
  T1 = 2bps + slippage  (maker-feasible)
  T2 = 10bps + slippage (taker, binance.us-like)
  T3 = 25bps + slippage (Alpaca-crypto-like; deployment reality check)
  Slippage: max(1bp, half the bar's (high-low)/midpoint capped at 10bps),
  participation capped at 1% of bar volume. VERDICT TIER = T2 unless the
  spec registers maker execution with a fill-rate argument (exp17 lesson:
  maker fills are adversely selected — the argument must address it).

## Pre-registration requirements (every spec, enforced by validator)
  1. Hypothesis + family + exact signal definition, parameters FROZEN.
  2. Power argument: expected validation trade count >= 300 AND the
     minimum detectable per-trade edge at t=2 stated and >= registered fees.
  3. Criteria that CAN fail under the null (protocol's exp14 rule).
  4. One validation scoring per spec. No reruns with tweaked params under
     the same name (freeze-hash enforced).

## Verdicts
  PASS requires ALL: validation net-of-T2 per-trade t >= 2.0; research and
  validation effect same sign; survives red-team adjudication; controls
  current (see below). PASS -> paper deployment same week on the existing
  execution stack, sized <= $10k. FAIL -> registry with reason. Sprint ends
  day 30 with a written verdict: deployed candidate(s), or the intraday
  free-data door recorded closed, with the registry as evidence.

## Harness controls (calibration, run before first real experiment and
## after any factory change)
  C0 null-control: random entries, same pipeline -> must FAIL. A pass means
     the harness leaks; halt sprint until fixed.
  C1 planted-edge control: synthetic series with +25bp/trade injected ->
     must PASS at T2. A fail means the harness cannot detect a real edge
     of deployable size; halt sprint until fixed.

## Red team: accuracy rules (2026-08-26, at principal's direction)
  A red-team objection is VALID only if it (a) names a defect from the
  taxonomy below, (b) cites specific evidence in the experiment artifact,
  and (c) states what result the defect predicts if fixed. "This looks
  overfit" without mechanism is an INVALID objection and is logged as such.
    Taxonomy: LOOKAHEAD (uses information unavailable at decision time),
    SURVIVORSHIP (universe excludes the dead), COST-OMISSION (wrong tier /
    missing slippage), MULTIPLE-TESTING (undisclosed prior tries),
    POWER (criteria could not fail under null), DATA-DEFECT (splits,
    duplicates, gaps, stale prints), REGIME (effect confined to one
    sub-period and spec claims otherwise).
  Adjudication: deterministic checks where possible (the factory re-runs
  with the alleged fix); otherwise the principal decides. An experiment is
  killed ONLY by a CONFIRMED defect — never by an unadjudicated objection.
  Red-team ledger: every objection scored confirmed/refuted; precision and
  recall tracked against seeded calibration cases (redteam_cases/) exactly
  like other seats' evals. A red team that kills randomly loses tier
  standing the same way a scout that cries wolf does.

## Cadence
  Nightly (factory, mechanical): run registered specs; write artifacts.
  Nightly (agents): research_lead proposes next spec from backlog + results;
  red_team reviews yesterday's artifact under the rules above;
  backtest_auditor spot-checks pipeline integrity. Proposals become
  registered specs after the deterministic validator passes them (schema +
  power + split compliance); validator rejections return to the lead.
  Human principal: reads the desk note; adjudicates disputed kills; may veto
  any registration. Weekly: coach reviews red-team precision.

## Prior art constraint
  The registry's closed doors stay closed: no respecting-the-map violations
  (no daily-frequency retreads, no reopening sealed windows, no exp23/exp17
  reruns without a NEW mechanism registered as such).

## Amendment 2026-08-26a (before any real experiment; C1 not yet run)
C1 as originally written planted +25bp/trade and demanded a PASS at T2 —
but T2 round-trip costs measured ~26bp (C0 artifact), so a +25bp edge
SHOULD fail: that is the harness being honest, not insensitive. C1 now
plants +40bp (≈14bp net of measured costs, a genuinely deployable edge)
and must PASS at T2. C0's requirement is unchanged and its first run
FAILED as required (gross +0.06bp, net -26.2bp, n=1518).
