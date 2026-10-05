# Sprint backlog — REORDERED 2026-08-27 after sprint05 (funding carry)
sprint05 result: the carry MECHANISM survived validation (3/3 months positive,
t~15) — only magnitude failed (majors compressed to 1-2%/yr vs 4% gate).
Direction: follow the mechanism that works, not new price-prediction signals.

PRIORITY 1 — CROSS-SECTIONAL ALT CARRY (sprint06). Literature: funds report
  ~19%/yr via selective pair choice; majors are compressed but alt perp
  funding is richer. Design: each week hold the K highest-trailing-funding
  USDT perps (hedged), subject to spot availability + ADV floor; costs per
  rotation; same splits/gates as sprint05. Data: extend carry.py loader
  across the full perp universe (data.binance.vision fundingRate).
PRIORITY 2 — PREDICTION-MARKET SCANNER (Kalshi x Polymarket). Doors VERIFIED
  open 08-27 (public APIs, no auth). Documented: $40M/yr extracted (IMDEA);
  1.5-4.5% cross-venue spreads; settlement-definition risk is the killer —
  scanner must match resolution criteria text, not just titles. Build =
  monitoring first (measure real spread frequency/size net of fees), trade
  paper only after measured.
PRIORITY 3 — KALSHI WEATHER vs GFS ENSEMBLES. Documented retail bot profit
  ($1.8k) pricing weather markets from free NOAA GFS ensembles. Small,
  genuinely uncrowded, model-vs-market edge. Needs GFS pull + market map.
PRIORITY 4 — taker_imbalance / qhour_open (order-flow signals, data live).
  Literature-backed but still price-prediction; run when queue is idle.
CLOSED: breakout (sprint01), naive meanrev/volspike variants (sprint02-04
  pending verdicts), majors-only carry below gate (sprint05), all 249 daily
  registry doors.
RESEARCH LEAD DIRECTIVE (principal, 08-27): no more parameter permutations
  of closed families. Every proposal must cite a mechanism with outside
  evidence (literature/registry) for why money should exist there.
