# The Fund — a 20-seat AI desk over three paper books

Runs 24/7 via launchd (`com.stanleyshen.tradingfund`, hourly heartbeat).
The orchestrator decides each hour: full desk (12 seats) at 7am/10am/12pm PT
during market hours, light crew (3 seats) other market hours, skeleton crew
(2 seats) every 4h off-hours, research wing daily after the close, evals
nightly, coach on Sundays, scribe daily.

Ground rules (why this desk is shaped the way it is):
- 248 pre-registered experiments found zero tradeable edges in free daily
  data, and turnover killed most candidates. So agents do NOT trade — the
  validated runners keep their schedules, and agents supervise, verify,
  research, and report. Agent value here is reliability, not order flow.
- Agents are tool-less: briefing in (fund/briefing.py — deterministic
  code gathers all data), memo out. No credentials, no filesystem access.
- Trust is earned: seats start tier 0 (advisory). Promotion gates are
  PRE-REGISTERED in evals/criteria.json. Tier 2 + an enabled discretionary
  sleeve (roster.json, default OFF) is the only path to agent-queued
  paper orders.
- Evolution: eval failures and journal reviews become playbook-diff
  proposals in coach_inbox/ — applied by a human, with the failing eval
  cited. Recorded eval misses are never retro-scored.

Map: roster.json (seats/schedule) · playbooks/ (skills) · orchestrator.py ·
briefing.py · evals/ (scenarios, scorer, criteria) · floor/ (per-cycle
memos) · journal/ (machine-readable decisions+proposals) · reports/
(desk_note_latest.md, fund_cron.log) · coach_inbox/ (evolution queue).
Check on it: cat fund/reports/desk_note_latest.md
